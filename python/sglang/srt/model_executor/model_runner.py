# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ModelRunner runs the forward passes of the models."""

import datetime
import gc
import json
import logging
import os
import time
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from sglang.semi_pd.utils import (
    InstanceRole,
    IPCInfo,
    convert_ipc_handle_to_tensor,
    get_ipc_handle,
)
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.srt.distributed import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_size,
    initialize_dp_attention,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization import monkey_patch_isinstance_for_vllm_base_layer
from sglang.srt.layers.sampler import Sampler
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.mem_cache.memory_pool import (
    DoubleSparseTokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.paged_allocator import PagedTokenToKVPoolAllocator
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader import get_model
from sglang.srt.model_loader.loader import (
    DefaultModelLoader,
    device_loading_context,
    get_model_loader,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    MultiprocessingSerializer,
    enable_show_time_cost,
    get_available_gpu_memory,
    init_custom_process_group,
    is_cuda,
    is_hip,
    monkey_patch_p2p_access_check,
    monkey_patch_vllm_gguf_config,
    set_cpu_offload_max_bytes,
    set_cuda_arch,
)

logger = logging.getLogger(__name__)

SGLANG_CI_SMALL_KV_SIZE = os.getenv("SGLANG_CI_SMALL_KV_SIZE", None)
UNBALANCED_MODEL_LOADING_TIMEOUT_S = 300


class ModelRunner:
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[TokenToKVPoolAllocator] = None,
        bypass_load_weight: bool = False,
        instance_role: InstanceRole = InstanceRole.OTHER,
    ):
        # Parse args
        self.model_config = model_config
        self.mem_fraction_static = mem_fraction_static
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.is_generation = model_config.is_generation
        self.is_multimodal = model_config.is_multimodal
        self.should_log = tp_rank == 0
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.bypass_load_weight = bypass_load_weight
        self.instance_role = instance_role

        # Model-specific adjustment
        self.model_specific_adjustment()

        if server_args.show_time_cost:
            enable_show_time_cost()

        if server_args.disable_outlines_disk_cache:
            from outlines.caching import disable_cache

            disable_cache()

        # Global vars
        global_server_args_dict.update(
            {
                "attention_backend": server_args.attention_backend,
                "sampling_backend": server_args.sampling_backend,
                "triton_attention_reduce_in_fp32": server_args.triton_attention_reduce_in_fp32,
                "disable_mla": server_args.disable_mla,
                "torchao_config": server_args.torchao_config,
                "enable_nan_detection": server_args.enable_nan_detection,
                "enable_dp_attention": server_args.enable_dp_attention,
                "enable_ep_moe": server_args.enable_ep_moe,
                "device": server_args.device,
                "speculative_accept_threshold_single": server_args.speculative_accept_threshold_single,
                "speculative_accept_threshold_acc": server_args.speculative_accept_threshold_acc,
                "enable_flashinfer_mla": server_args.enable_flashinfer_mla,
                "disable_radix_cache": server_args.disable_radix_cache,
                "flashinfer_mla_disable_ragged": server_args.flashinfer_mla_disable_ragged,
                "debug_tensor_dump_output_folder": server_args.debug_tensor_dump_output_folder,
                "debug_tensor_dump_inject": server_args.debug_tensor_dump_inject,
            }
        )

        # CPU offload
        set_cpu_offload_max_bytes(int(server_args.cpu_offload_gb * 1024**3))

        # Get memory before model loading
        min_per_gpu_memory = self.init_torch_distributed()

        # If it is a draft model tp_group can be different.
        self.initialize(min_per_gpu_memory)

    def initialize(self, min_per_gpu_memory: float):
        """
        Semi-PD:
        - skip attn and cuda graph init for standalone instances
        - delay attn and cuda graph init for P & D instances
        """
        server_args = self.server_args
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        # Load the model
        self.sampler = Sampler()
        self.load_model()

        # Apply torchao quantization
        torchao_applied = getattr(self.model, "torchao_applied", False)
        # In layered loading, torchao may have been applied
        if not torchao_applied:
            apply_torchao_config_to_model(
                self.model, global_server_args_dict["torchao_config"]
            )

        # Apply torch TP if the model supports it
        supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        if self.tp_size > 1 and supports_torch_tp:
            self.apply_torch_tp()
            self.torch_tp_applied = True
        else:
            self.torch_tp_applied = False

        # Init lora
        if server_args.lora_paths is not None:
            self.init_lora_manager()

        # Init memory pool and attention backends
        self.init_memory_pool(
            min_per_gpu_memory,
            server_args.max_running_requests,
            server_args.max_total_tokens,
        )

        self.cuda_graph_runner = None
        if self.device == "cuda":
            self.init_cublas()
            if not self.server_args.enable_semi_pd:
                # Semi-PD
                self.init_attention_backend()
                self.init_cuda_graphs()
        else:
            self.init_attention_backend()

    def model_specific_adjustment(self):
        server_args = self.server_args

        if (
            self.model_config.attention_arch == AttentionArch.MLA
            and not server_args.disable_mla
        ):
            # TODO: add MLA optimization on CPU
            if server_args.device != "cpu":
                if server_args.enable_flashinfer_mla:
                    logger.info(
                        "MLA optimization is turned on. Use flashinfer mla backend."
                    )
                    server_args.attention_backend = "flashinfer_mla"
                else:
                    logger.info("MLA optimization is turned on. Use triton backend.")
                    server_args.attention_backend = "triton"

        if server_args.enable_double_sparsity:
            logger.info(
                "Double sparsity optimization is turned on. Use triton backend without CUDA graph."
            )
            server_args.attention_backend = "triton"
            server_args.disable_cuda_graph = True
            if server_args.ds_heavy_channel_type is None:
                raise ValueError(
                    "Please specify the heavy channel type for double sparsity optimization."
                )
            self.init_double_sparsity_channel_config(server_args.ds_heavy_channel_type)

        if self.is_multimodal:
            self.mem_fraction_static *= 0.95
            logger.info(
                f"Automatically reduce --mem-fraction-static to {self.mem_fraction_static:.3f} "
                f"because this is a multimodal model."
            )

            if self.model_config.hf_config.architectures == [
                "MllamaForConditionalGeneration"
            ]:
                logger.info("Automatically turn off --chunked-prefill-size for mllama.")
                server_args.chunked_prefill_size = -1

            if self.model_config.hf_config.architectures == [
                "Qwen2VLForConditionalGeneration"
            ]:
                # TODO: qwen2-vl does not support radix cache now, set disable_radix_cache=True automatically
                logger.info(
                    "Automatically turn off --chunked-prefill-size and disable radix cache for qwen2-vl."
                )
                server_args.chunked_prefill_size = -1
                server_args.disable_radix_cache = True

    def init_torch_distributed(self):
        logger.info("Init torch distributed begin.")

        torch.get_device_module(self.device).set_device(self.gpu_id)
        if self.device == "cuda":
            backend = "nccl"
        elif self.device == "xpu":
            backend = "xccl"
        elif self.device == "hpu":
            backend = "hccl"
        elif self.device == "cpu":
            backend = "gloo"

        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if not self.server_args.enable_p2p_check:
            monkey_patch_p2p_access_check()

        if self.server_args.dist_init_addr:
            dist_init_method = f"tcp://{self.server_args.dist_init_addr}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{self.dist_port}"
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)

        if not self.is_draft_worker:
            # Only initialize the distributed environment on the target model worker.
            init_distributed_environment(
                backend=backend,
                world_size=self.tp_size,
                rank=self.tp_rank,
                local_rank=self.gpu_id,
                distributed_init_method=dist_init_method,
                timeout=self.server_args.dist_timeout,
            )
            initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
            initialize_dp_attention(
                enable_dp_attention=self.server_args.enable_dp_attention,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                dp_size=self.server_args.dp_size,
            )

        min_per_gpu_memory = get_available_gpu_memory(
            self.device, self.gpu_id, distributed=self.tp_size > 1
        )
        self.tp_group = get_tp_group()
        self.attention_tp_group = get_attention_tp_group()

        # Check memory for tensor parallelism
        local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if self.tp_size > 1:
            if min_per_gpu_memory < local_gpu_memory * 0.9:
                raise ValueError(
                    "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                    f"{min_per_gpu_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                )

        logger.info(
            f"Init torch distributed ends. mem usage={(before_avail_memory - local_gpu_memory):.2f} GB"
        )
        return min_per_gpu_memory

    def get_ipc_info(self) -> IPCInfo:
        def check_duplicate_handle(handle_to_name_, handle_, name_):
            hashed = tuple(handle_[0]), handle_[1]
            if handle_to_name_.get(hashed, None) is not None and handle_ != "BYPASS":
                logger.warning(
                    f"Duplicate handle found, {handle_to_name_[hashed]} and {name_}"
                )
            handle_to_name_[hashed] = name_

        assert not self.bypass_load_weight

        handle_to_name = {}
        tensor_info = {}
        weight_handles = {}
        register_buffer_handles = {}

        # Get Parameter Handles
        source_params = dict(self.model.named_parameters())
        for name, _ in self.model.named_parameters():
            # Get the path to the parameter
            path = name.split(".")

            # Navigate to the parent module
            module = self.model
            for p in path[:-1]:
                if p.isdigit():
                    module = module[int(p)]
                else:
                    module = getattr(module, p)
            # Create a parameter that shares storage with source parameter
            source_param = source_params[name]
            param_tensor = source_param.view_as(source_param)
            
            # Bypass empty parameter
            if param_tensor.numel() == 0:
                ipc_handle = "BYPASS"
            else:
                ipc_handle = get_ipc_handle(param_tensor)
            check_duplicate_handle(handle_to_name, ipc_handle, name)

            weight_handles[name] = ipc_handle
            tensor_info[name] = (
                param_tensor.shape,
                param_tensor.dtype,
                param_tensor.device,
            )

        # Get Non-Parameter Buffers, eg. cos_sin_cache
        source_buffers = dict(self.model.named_buffers())
        for name, _ in self.model.named_buffers():
            # Get the path to the parameter
            path = name.split(".")

            # Navigate to the parent module
            module = self.model
            for p in path[:-1]:
                if p.isdigit():
                    module = module[int(p)]
                else:
                    module = getattr(module, p)

            # Create a parameter that shares storage with source parameter
            source_buffer = source_buffers[name]
            if source_buffer.numel() == 0:
                tensor_info[name] = (None, None, None)
                continue
            buffer_tensor = source_buffer.view_as(source_buffer)
            
            # Bypass empty parameter
            if buffer_tensor.numel() == 0:
                ipc_handle = "BYPASS"
            else:
                ipc_handle = get_ipc_handle(buffer_tensor)
            check_duplicate_handle(handle_to_name, ipc_handle, name)

            register_buffer_handles[name] = ipc_handle
            tensor_info[name] = (
                buffer_tensor.shape,
                buffer_tensor.dtype,
                buffer_tensor.device,
            )

        # Get KV Cache Handles
        if isinstance(self.token_to_kv_pool, MHATokenToKVPool):
            k_caches = self.token_to_kv_pool.k_buffer
            v_caches = self.token_to_kv_pool.v_buffer
            k_cache_handles = [get_ipc_handle(k_cache) for k_cache in k_caches]
            v_cache_handles = [get_ipc_handle(v_cache) for v_cache in v_caches]

            for i, (k_cache_handle, v_cache_handle) in enumerate(
                zip(k_cache_handles, v_cache_handles)
            ):
                check_duplicate_handle(handle_to_name, k_cache_handle, f"k_cache_{i}")
                check_duplicate_handle(handle_to_name, v_cache_handle, f"v_cache_{i}")

            kvcache_info = {
                "cache_shape": k_caches[0].shape,
                "cache_dtype": k_caches[0].dtype,
                "cache_device": k_caches[0].device,
            }
            kv_cache_handles = [k_cache_handles, v_cache_handles]
        elif isinstance(self.token_to_kv_pool, MLATokenToKVPool):
            kv_caches = self.token_to_kv_pool.kv_buffer
            kv_cache_handles = [get_ipc_handle(kv_cache) for kv_cache in kv_caches]
            for i, kv_cache_handle in enumerate(kv_cache_handles):
                check_duplicate_handle(handle_to_name, kv_cache_handle, f"kv_cache_{i}")
            kvcache_info = {
                "cache_shape": kv_caches[0].shape,
                "cache_dtype": kv_caches[0].dtype,
                "cache_device": kv_caches[0].device,
            }
        else:
            raise ValueError(
                f"Unsupported token to kv pool type: {type(self.token_to_kv_pool)}"
            )

        # Get ReqToToken Handles
        req_to_token_tensor = self.req_to_token_pool.req_to_token
        req_to_token_handles = [get_ipc_handle(req_to_token_tensor)]
        req_to_token_info = {
            "req_to_token_shape": req_to_token_tensor.shape,
            "req_to_token_dtype": req_to_token_tensor.dtype,
            "req_to_token_device": req_to_token_tensor.device,
        }

        return IPCInfo(
            params_info=tensor_info,
            weight_handles=weight_handles,
            register_buffer_handles=register_buffer_handles,
            kv_cache_handles=kv_cache_handles,
            kvcache_info=kvcache_info,
            req_to_token_handle=req_to_token_handles,
            req_to_token_info=req_to_token_info,
        )

    def share_params_from_ipc(self, ipc_info: IPCInfo):
        # Reconstruct parameters from IPC handles
        logger.info("🔍 [ORIGINAL SEMI-PD] Starting parameter sharing from IPC...")

        # 🔍 VERIFY: Count parameters before sharing
        param_count_before = sum(1 for _ in self.model.named_parameters())
        logger.info(f"🔍 [ORIGINAL SEMI-PD] Parameters before IPC sharing: {param_count_before}")

        # 🔍 VERIFY: Check some parameter checksums before sharing
        self._checksums_before = {}
        param_count = 0
        for name, param in self.model.named_parameters():
            if param_count >= 3:  # Only check first 3 parameters
                break
            if param.numel() > 0 and not param.is_meta:  # Skip meta tensors
                try:
                    checksum = torch.sum(param.data).item()
                    data_ptr = param.data.data_ptr()
                    self._checksums_before[name] = {'checksum': checksum, 'data_ptr': data_ptr}
                    logger.info(f"🔍 [ORIGINAL SEMI-PD] BEFORE - {name}: checksum={checksum:.6f}, ptr=0x{data_ptr:x}")
                    param_count += 1
                except Exception as e:
                    logger.info(f"🔍 [ORIGINAL SEMI-PD] BEFORE - {name}: meta tensor (no data yet)")
                    self._checksums_before[name] = {'checksum': None, 'data_ptr': None, 'is_meta': True}

        for name, _ in self.model.named_parameters():
            # Get the path to the parameter
            path = name.split(".")

            # Navigate to the parent module
            module = self.model
            for p in path[:-1]:
                if p.isdigit():
                    module = module[int(p)]
                else:
                    module = getattr(module, p)

            # Get the parameter name (last part of the path)
            param_name = path[-1]

            share_param_handle = ipc_info.weight_handles.get(name, None)
            shape, dtype, device = ipc_info.params_info[name]
            size = reduce(lambda x, y: x * y, shape)

            assert (
                share_param_handle is not None
            ), f"Parameter {name} not found in meta_info"
            
            try:
                if shape == torch.Size([0]):
                    share_param_tensor = torch.empty(0, dtype=dtype, device=device)
                else:
                    share_param_tensor = convert_ipc_handle_to_tensor(
                        share_param_handle, size, dtype, device
                    ).view(shape)
            except Exception as e:
                raise NotImplementedError(f"Parameter {name, size, dtype, device} is not supported in Semi-PD")
            
            new_param = nn.Parameter(share_param_tensor, requires_grad=False)
            setattr(module, param_name, new_param)

        # Reconstruct registered buffers from IPC handles
        for name, _ in self.model.named_buffers():
            # Get the path to the parameter
            path = name.split(".")

            # Navigate to the parent module
            module = self.model
            for p in path[:-1]:
                if p.isdigit():
                    module = module[int(p)]
                else:
                    module = getattr(module, p)

            # Get the parameter name (last part of the path)
            buffer_name = path[-1]

            share_buffer_handle = ipc_info.register_buffer_handles.get(name, None)
            shape, dtype, device = ipc_info.params_info[name]

            if shape is None:
                continue
            assert (
                share_buffer_handle is not None
            ), f"Buffer {name} not found in meta_info"

            # Shape can be [] when the buffer represents a scalar
            size = reduce(lambda x, y: x * y, shape) if shape else 1
            # For deepseek model
            if "w_kc" in name or "w_vc" in name:
                shape = [shape[0], shape[2], shape[1]]
                share_buffer_tensor = (
                    convert_ipc_handle_to_tensor(
                        share_buffer_handle, size, dtype, device
                    )
                    .view(shape)
                    .transpose(1, 2)
                )
            else:
                share_buffer_tensor = convert_ipc_handle_to_tensor(
                    share_buffer_handle, size, dtype, device
                ).view(shape)
            module.register_buffer(buffer_name, share_buffer_tensor, persistent=False)
            # setattr(module, buffer_name, share_buffer_tensor)

        # Reconstruct req_to_token from IPC handles
        req_to_token_shape = ipc_info.req_to_token_info["req_to_token_shape"]
        req_to_token_dtype = ipc_info.req_to_token_info["req_to_token_dtype"]
        req_to_token_device = ipc_info.req_to_token_info["req_to_token_device"]
        size = reduce(lambda x, y: x * y, req_to_token_shape)
        self.req_to_token_pool.req_to_token = convert_ipc_handle_to_tensor(
            ipc_info.req_to_token_handle[0],
            size,
            req_to_token_dtype,
            req_to_token_device,
        ).view(req_to_token_shape)

        # Reconstruct kv cache from IPC handles
        if isinstance(self.token_to_kv_pool, MHATokenToKVPool):
            k_buffer = []
            v_buffer = []
            for k_cache_handle, v_cache_handle in zip(
                ipc_info.kv_cache_handles[0], ipc_info.kv_cache_handles[1]
            ):
                cache_shape = ipc_info.kvcache_info["cache_shape"]
                cache_dtype = ipc_info.kvcache_info["cache_dtype"]
                cache_device = ipc_info.kvcache_info["cache_device"]
                size = reduce(lambda x, y: x * y, cache_shape)
                k_cache_tensor = convert_ipc_handle_to_tensor(
                    k_cache_handle, size, cache_dtype, cache_device
                ).view(cache_shape)
                v_cache_tensor = convert_ipc_handle_to_tensor(
                    v_cache_handle, size, cache_dtype, cache_device
                ).view(cache_shape)
                k_buffer.append(k_cache_tensor)
                v_buffer.append(v_cache_tensor)
            self.token_to_kv_pool.k_buffer = k_buffer
            self.token_to_kv_pool.v_buffer = v_buffer
        elif isinstance(self.token_to_kv_pool, MLATokenToKVPool):
            kv_buffer = []
            for kv_cache_handle in ipc_info.kv_cache_handles:
                cache_shape = ipc_info.kvcache_info["cache_shape"]
                cache_dtype = ipc_info.kvcache_info["cache_dtype"]
                cache_device = ipc_info.kvcache_info["cache_device"]
                size = reduce(lambda x, y: x * y, cache_shape)
                kv_cache_tensor = convert_ipc_handle_to_tensor(
                    kv_cache_handle, size, cache_dtype, cache_device
                ).view(cache_shape)
                kv_buffer.append(kv_cache_tensor)
            self.token_to_kv_pool.kv_buffer = kv_buffer
        else:
            raise ValueError(
                f"Unsupported token to kv pool type: {type(self.token_to_kv_pool)}"
            )

        # Reconstruct req_to_token from IPC handles
        req_to_token_shape = ipc_info.req_to_token_info["req_to_token_shape"]
        req_to_token_dtype = ipc_info.req_to_token_info["req_to_token_dtype"]
        req_to_token_device = ipc_info.req_to_token_info["req_to_token_device"]
        size = reduce(lambda x, y: x * y, req_to_token_shape)
        req_to_token_tensor = convert_ipc_handle_to_tensor(
            ipc_info.req_to_token_handle[0],
            size,
            req_to_token_dtype,
            req_to_token_device,
        ).view(req_to_token_shape)
        self.req_to_token_pool.req_to_token = req_to_token_tensor

        # 🔍 VERIFY: Check parameter checksums after sharing
        logger.info("🔍 [ORIGINAL SEMI-PD] Parameter sharing completed, verifying results...")

        checksums_after = {}
        param_count = 0
        for name, param in self.model.named_parameters():
            if param_count >= 3:  # Only check first 3 parameters
                break
            if param.numel() > 0 and not param.is_meta:  # Skip meta tensors
                try:
                    checksum = torch.sum(param.data).item()
                    data_ptr = param.data.data_ptr()
                    checksums_after[name] = {'checksum': checksum, 'data_ptr': data_ptr}
                    logger.info(f"🔍 [ORIGINAL SEMI-PD] AFTER - {name}: checksum={checksum:.6f}, ptr=0x{data_ptr:x}")
                    param_count += 1
                except Exception as e:
                    logger.warning(f"🔍 [ORIGINAL SEMI-PD] AFTER - {name}: Failed to get checksum: {e}")
                    checksums_after[name] = {'checksum': None, 'data_ptr': None, 'error': str(e)}

        # 🔍 VERIFY: Compare before and after (if checksums_before exists)
        if hasattr(self, '_checksums_before'):
            for name in self._checksums_before:
                if name in checksums_after:
                    before = self._checksums_before[name]
                    after = checksums_after[name]

                    if before['data_ptr'] != after['data_ptr']:
                        logger.info(f"✅ [ORIGINAL SEMI-PD] {name}: Memory pointer changed (IPC sharing worked)")
                        logger.info(f"✅ [ORIGINAL SEMI-PD] {name}: Before ptr=0x{before['data_ptr']:x}, After ptr=0x{after['data_ptr']:x}")
                    else:
                        logger.warning(f"⚠️  [ORIGINAL SEMI-PD] {name}: Memory pointer unchanged (IPC sharing may have failed)")

                    if abs(before['checksum'] - after['checksum']) < 1e-6:
                        logger.info(f"✅ [ORIGINAL SEMI-PD] {name}: Checksum preserved (data integrity maintained)")
                    else:
                        logger.warning(f"⚠️  [ORIGINAL SEMI-PD] {name}: Checksum changed! Before={before['checksum']:.6f}, After={after['checksum']:.6f}")

        logger.info("🔍 [ORIGINAL SEMI-PD] Parameter sharing verification completed!")

    def load_model(self):
        if not self.bypass_load_weight:
            before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
            )
        else:
            logger.info("Bypass loading weight")

        # This can reduce thread conflicts and speed up weight loading.
        if self.device != "cpu":
            torch.set_num_threads(1)
        if self.device == "cuda":
            if torch.cuda.get_device_capability()[0] < 8:
                logger.info(
                    "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                )
                self.server_args.dtype = "float16"
                self.model_config.dtype = torch.float16
                if torch.cuda.get_device_capability()[1] < 5:
                    raise RuntimeError("SGLang only supports sm75 and above.")

        set_cuda_arch()

        # Prepare the model config
        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
        )
        if self.server_args.load_format == "gguf":
            monkey_patch_vllm_gguf_config()

        # Load the model
        # Remove monkey_patch when linear.py quant remove dependencies with vllm
        monkey_patch_vllm_parallel_state()
        monkey_patch_isinstance_for_vllm_base_layer()

        with self.memory_saver_adapter.region():
            device_config = (
                DeviceConfig(self.device)
                if not self.bypass_load_weight
                else DeviceConfig("meta")
            )

            self.model = get_model(
                model_config=self.model_config,
                load_config=self.load_config,
                device_config=device_config,
                bypass_load_weight=self.bypass_load_weight,
            )

        monkey_patch_vllm_parallel_state(reverse=True)
        monkey_patch_isinstance_for_vllm_base_layer(reverse=True)

        if self.server_args.kv_cache_dtype == "fp8_e4m3":
            if self.server_args.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    self.model.load_kv_cache_scales(
                        self.server_args.quantization_param_path
                    )
                    logger.info(
                        "Loaded KV cache scaling factors from %s",
                        self.server_args.quantization_param_path,
                    )
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__,
                    )
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!"
                )

        # Parse other args
        self.sliding_window_size = (
            self.model.get_attention_sliding_window_size()
            if hasattr(self.model, "get_attention_sliding_window_size")
            else None
        )
        self.dtype = self.model_config.dtype

        if not self.bypass_load_weight:
            after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Load weight end. "
                f"type={type(self.model).__name__}, "
                f"dtype={self.dtype}, "
                f"avail mem={after_avail_memory:.2f} GB, "
                f"mem usage={(before_avail_memory - after_avail_memory):.2f} GB."
            )

        # Handle the case where some ranks do not finish loading.
        try:
            dist.monitored_barrier(
                group=get_tp_group().cpu_group,
                timeout=datetime.timedelta(seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S),
                wait_all_ranks=True,
            )
        except RuntimeError:
            raise ValueError(
                f"TP rank {self.tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node."
            ) from None

    def update_weights_from_disk(
        self, model_path: str, load_format: str
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk."""
        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(load_format=load_format)

        # Only support DefaultModelLoader for now
        loader = get_model_loader(load_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source(
                    config.model_path,
                    revision=config.revision,
                    fall_back_to_pt=getattr(
                        self.model, "fall_back_to_pt_during_load", True
                    ),
                )
            )
            return iter

        def model_load_weights(model, iter):
            model.load_weights(iter)
            for _, module in self.model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
            return model

        with set_default_torch_dtype(self.model_config.dtype):
            try:
                iter = get_weight_iter(self.model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.model, iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message

        self.model = model
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.load_config = load_config

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in the RLHF workflow, where rank
        0 is the actor model in the training engine, and the other ranks are
        the inference engine, which is used for rollout.

        In the RLHF workflow, the training engine updates the model
        weights/parameters online, and broadcasts them to the inference
        engine through the `_model_update_group` process group.
        """
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            self._model_update_group = init_custom_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            dist.barrier(group=self._model_update_group, device_ids=[rank])
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    def update_weights_from_distributed(self, name, dtype, shape):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
        """
        target_dtype = (
            dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        )

        assert (
            self._model_update_group is not None
        ), "model update group must be initialized"

        try:
            weights = torch.empty(shape, dtype=target_dtype, device=self.device)
            torch.distributed.broadcast(weights, src=0, group=self._model_update_group)
            self.model.load_weights([(name, weights)])
            return True, f"Succeeded to update parameter {name} online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, Union[torch.Tensor, "LocalSerializedTensor"]]],
        load_format: Optional[str] = None,
    ):
        named_tensors = [
            (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank))
            for name, tensor in named_tensors
        ]
        if load_format == "direct":
            _model_load_weights_direct(self.model, named_tensors)
        elif load_format is None:
            self.model.load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        return True, "Success"

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        # TODO: (chenyang) Add support for Qwen models.
        try:
            return self.model.get_weights_by_name(
                name, truncate_size, tp_size=self.tp_size
            )
        except Exception as e:
            logger.error(f"Error when getting parameter {name}: {e}")
            return None

    def init_lora_manager(self):
        self.lora_manager = LoRAManager(
            base_model=self.model,
            lora_paths=self.server_args.lora_paths,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
            lora_backend=self.server_args.lora_backend,
        )
        logger.info("LoRA manager ready.")

    def profile_max_num_token(self, total_gpu_memory: int):
        available_gpu_memory = get_available_gpu_memory(
            self.device, self.gpu_id, distributed=self.tp_size > 1
        )
        if (
            self.model_config.attention_arch == AttentionArch.MLA
            and not self.server_args.disable_mla
        ):
            cell_size = (
                (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
                * self.model_config.num_hidden_layers
                * torch._utils._element_size(self.kv_cache_dtype)
            )
        else:
            cell_size = (
                self.model_config.get_num_kv_heads(get_attention_tp_size())
                * self.model_config.head_dim
                * self.model_config.num_hidden_layers
                * 2
                * torch._utils._element_size(self.kv_cache_dtype)
            )
        rest_memory = available_gpu_memory - total_gpu_memory * (
            1 - self.mem_fraction_static
        )
        max_num_token = int(rest_memory * (1 << 30) // cell_size)
        return max_num_token

    def init_memory_pool(
        self,
        total_gpu_memory: int,
        max_num_reqs: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
    ):
        if self.server_args.kv_cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            if is_hip():  # Using natively supported format
                self.kv_cache_dtype = torch.float8_e5m2fnuz
            else:
                self.kv_cache_dtype = torch.float8_e5m2
        elif self.server_args.kv_cache_dtype == "fp8_e4m3":
            if is_cuda():
                self.kv_cache_dtype = torch.float8_e4m3fn
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )

        if self.instance_role == InstanceRole.OTHER or self.instance_role == InstanceRole.DECODE:
            self.max_total_num_tokens = self.profile_max_num_token(total_gpu_memory)
        else:
            assert (
                max_total_tokens is not None
            ), f"max_total_tokens is required for {self.instance_role} instance"
            self.max_total_num_tokens = max_total_tokens

        if max_num_reqs is None:
            max_num_reqs = min(
                max(
                    int(
                        self.max_total_num_tokens / self.model_config.context_len * 512
                    ),
                    2048,
                ),
                4096,
            )

        if SGLANG_CI_SMALL_KV_SIZE:
            self.max_total_num_tokens = int(SGLANG_CI_SMALL_KV_SIZE)

        if not self.spec_algorithm.is_none():
            if self.is_draft_worker:
                self.max_total_num_tokens = self.server_args.draft_runner_cache_size
                max_num_reqs = self.server_args.max_num_reqs
            else:
                # We are sharing the `token_to_kv_pool`, and both verify and draft tokens
                # can be concurrently allocated, so we should give a headroom for it.
                self.server_args.draft_runner_cache_size = (
                    self.max_total_num_tokens
                    # draft
                    + max_num_reqs
                    * self.server_args.speculative_num_steps
                    * self.server_args.speculative_eagle_topk
                    # verify
                    + max_num_reqs * self.server_args.speculative_num_draft_tokens
                    # buffer
                    + 100
                )
                # Target worker and draft worker shares the same indices for the
                # token_to_kv_pool, so we should make sure to match max_total_num_tokens.
                self.max_total_num_tokens = self.server_args.draft_runner_cache_size
                self.server_args.max_num_reqs = max_num_reqs

        if max_total_tokens is not None:
            if max_total_tokens > self.max_total_num_tokens:
                logging.warning(
                    f"max_total_tokens={max_total_tokens} is larger than the profiled value "
                    f"{self.max_total_num_tokens}. "
                    f"Use the profiled value instead."
                )
            self.max_total_num_tokens = min(self.max_total_num_tokens, max_total_tokens)

        self.max_total_num_tokens = (
            self.max_total_num_tokens
            // self.server_args.page_size
            * self.server_args.page_size
        )
        logger.info(f" Using max_total_num_tokens={self.max_total_num_tokens}")

        if self.max_total_num_tokens <= 0:
            raise RuntimeError(
                "Not enough memory. Please try to increase --mem-fraction-static."
            )

        if self.req_to_token_pool is None:
            self.req_to_token_pool = ReqToTokenPool(
                size=max_num_reqs + 1,
                max_context_len=self.model_config.context_len + 4,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                bypass_create_buffers=self.bypass_load_weight,
            )
        else:
            # Draft worker shares req_to_token_pool with the target worker.
            assert self.is_draft_worker

        if (
            self.model_config.attention_arch == AttentionArch.MLA
            and not self.server_args.disable_mla
        ):
            self.token_to_kv_pool = MLATokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                kv_lora_rank=self.model_config.kv_lora_rank,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                layer_num=self.model_config.num_hidden_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                bypass_create_buffers=self.bypass_load_weight,
            )
        elif self.server_args.enable_double_sparsity:
            assert (
                not self.server_args.enable_semi_pd
            ), "Double sparsity is not supported with semi-PD"

            self.token_to_kv_pool = DoubleSparseTokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
                head_dim=self.model_config.head_dim,
                layer_num=self.model_config.num_hidden_layers,
                device=self.device,
                heavy_channel_num=self.server_args.ds_heavy_channel_num,
                enable_memory_saver=self.server_args.enable_memory_saver,
            )
        else:
            self.token_to_kv_pool = MHATokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
                head_dim=self.model_config.head_dim,
                layer_num=self.model_config.num_hidden_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                bypass_create_buffers=self.bypass_load_weight,
            )

        if self.token_to_kv_pool_allocator is None:
            if self.page_size == 1:
                self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                    self.max_total_num_tokens,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                )
            else:
                self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                )
        else:
            assert self.is_draft_worker

        logger.info(
            f"Memory pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

    def init_cublas(self):
        """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later."""
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        c = a @ b
        return c

    def init_attention_backend(self):
        """Init attention kernel backend."""
        if self.server_args.attention_backend == "flashinfer":
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferAttnBackend,
            )

            # Init streams
            if self.server_args.speculative_algorithm == "EAGLE":
                self.plan_stream_for_flashinfer = torch.cuda.Stream()
            self.attn_backend = FlashInferAttnBackend(self)
        elif self.server_args.attention_backend == "triton":
            assert self.sliding_window_size is None, (
                "Window attention is not supported in the triton attention backend. "
                "Please use `--attention-backend flashinfer`."
            )
            assert not self.model_config.is_encoder_decoder, (
                "Cross attention is not supported in the triton attention backend. "
                "Please use `--attention-backend flashinfer`."
            )
            if self.server_args.enable_double_sparsity:
                from sglang.srt.layers.attention.double_sparsity_backend import (
                    DoubleSparseAttnBackend,
                )

                self.attn_backend = DoubleSparseAttnBackend(self)
            else:
                from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

                self.attn_backend = TritonAttnBackend(self)
        elif self.server_args.attention_backend == "torch_native":
            from sglang.srt.layers.attention.torch_native_backend import (
                TorchNativeAttnBackend,
            )

            self.attn_backend = TorchNativeAttnBackend(self)
        elif self.server_args.attention_backend == "flashinfer_mla":
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAAttnBackend,
            )

            self.attn_backend = FlashInferMLAAttnBackend(self)
        else:
            raise ValueError(
                f"Invalid attention backend: {self.server_args.attention_backend}"
            )

    def init_double_sparsity_channel_config(self, selected_channel):
        selected_channel = "." + selected_channel + "_proj"
        self.sorted_channels = []
        # load channel config
        with open(self.server_args.ds_channel_config_path, "r") as f:
            channel_config = json.load(f)

        for i in range(self.model_config.num_hidden_layers):
            key = "model.layers." + str(i) + ".self_attn" + selected_channel
            self.sorted_channels.append(
                torch.tensor(channel_config[key])[
                    :, : self.server_args.ds_heavy_channel_num
                ]
                .contiguous()
                .cuda()
            )

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            return

        if self.server_args.disable_cuda_graph:
            return

        tic = time.time()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        self.cuda_graph_runner = CudaGraphRunner(self)
        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture cuda graph end. Time elapsed: {time.time() - tic:.2f} s. "
            f"avail mem={after_mem:.2f} GB. mem usage={(before_mem - after_mem):.2f} GB."
        )

    def apply_torch_tp(self):
        logger.info(f"Enabling torch tensor parallelism on {self.tp_size} devices.")
        from sglang.srt.model_parallel import tensor_parallel

        device_mesh = torch.distributed.init_device_mesh(self.device, (self.tp_size,))
        tensor_parallel(self.model, device_mesh)

    def forward_decode(self, forward_batch: ForwardBatch):
        self.attn_backend.init_forward_metadata(forward_batch)
        return self.model.forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        )

    def forward_extend(
        self, forward_batch: ForwardBatch, skip_attn_backend_init: bool = False
    ):
        if not skip_attn_backend_init:
            self.attn_backend.init_forward_metadata(forward_batch)

        if self.is_generation:
            if forward_batch.input_embeds is None:
                return self.model.forward(
                    forward_batch.input_ids, forward_batch.positions, forward_batch
                )
            else:
                return self.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    input_embeds=forward_batch.input_embeds.bfloat16(),
                )
        else:
            # Only embedding models have get_embedding parameter
            return self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                get_embedding=True,
            )

    def forward_idle(self, forward_batch: ForwardBatch):
        return self.model.forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        )

    def forward(
        self, forward_batch: ForwardBatch, skip_attn_backend_init: bool = False
    ) -> LogitsProcessorOutput:
        if (
            forward_batch.forward_mode.is_cuda_graph()
            and self.cuda_graph_runner
            and self.cuda_graph_runner.can_run(forward_batch)
        ):
            return self.cuda_graph_runner.replay(
                forward_batch, skip_attn_backend_init=skip_attn_backend_init
            )

        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(forward_batch)
        elif forward_batch.forward_mode.is_extend():
            return self.forward_extend(
                forward_batch, skip_attn_backend_init=skip_attn_backend_init
            )
        elif forward_batch.forward_mode.is_idle():
            return self.forward_idle(forward_batch)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        # Apply logit bias
        if sampling_info.sampling_info_done:
            # Overlap mode: the function update_regex_vocab_mask was executed
            # in process_batch_result of the last batch.
            if sampling_info.grammars:
                sampling_info.sampling_info_done.wait()
        else:
            # Normal mode: Put CPU-heavy tasks here. They will be overlapped with the forward pass.
            sampling_info.update_regex_vocab_mask()
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids
        """
        # For duplex models with multiple output streams.
        if isinstance(logits_output, tuple):
            return torch.stack(
                [self.sample(values, forward_batch) for values in logits_output],
                axis=-1,
            )

        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Sample the next tokens
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
        )
        return next_token_ids

    @property
    def model_is_mrope(self) -> bool:
        """Detect if the model has "mrope" rope_scaling type.
        mrope requires keep "rope_deltas" between prompt and decoding phases."""
        rope_scaling = getattr(self.model_config.hf_config, "rope_scaling", {})
        if rope_scaling is None:
            return False
        return rope_scaling.get("type", None) == "mrope"


def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        default_weight_loader(params_dict[name], tensor)


def _unwrap_tensor(tensor, tp_rank):
    if isinstance(tensor, LocalSerializedTensor):
        return tensor.get(tp_rank)
    return tensor


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])
