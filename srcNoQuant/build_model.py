import os
import json
from functools import cache
from dataclasses import dataclass
import typing as tp
import subprocess
import torch
from torch import nn
import psutil
from transformers import AutoConfig
from transformers.models.mixtral import MixtralForCausalLM, MixtralConfig

from safetensors.torch import load_file

from torch import nn
from tqdm.auto import trange



from .expert_cache import ExpertCache
from .expert_wrapper import MixtralExpertWrapper
from .custom_layers import (
    MixtralBlockSparseTop2MLP,
    SparseMoeWrapper,
)
from .utils import with_default_dtype

def print_gpu_memory():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], encoding='utf-8')
        # 分割输出获取内存使用情况
        memory_info = result.strip().split("\n")
        for gpu, mem_info in enumerate(memory_info):
            used, total = mem_info.split(', ')
            print(f"GPU {gpu}: 使用内存 {used}MB / 总内存 {total}MB")
    except Exception as e:
        print(f"获取GPU内存信息失败: {e}")

@dataclass(frozen=True)
class OffloadConfig:
    main_size: int
    offload_size: int
    buffer_size: int
    offload_per_layer: int



def make_empty_expert(
    model_config: MixtralConfig
) -> MixtralBlockSparseTop2MLP:
    return MixtralBlockSparseTop2MLP(
        model_config,
    )


def make_and_load_expert_wrapper( #load parameters
    config: MixtralConfig,
    states_dir: str,
    expert_uid: tuple[int, int],
    device: torch.device,
) -> MixtralExpertWrapper:
    layer_idx, expert_idx = expert_uid

    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        module_idx = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"
        state_fpath = json.load(f)["weight_map"][f"{module_idx}.w1.weight"]

    state_dict = load_file(os.path.join(states_dir, state_fpath), device=str(device))
    expert = make_empty_expert(config)
    expert.load_state_dict(state_dict, strict=True)
    expert.half()
    return MixtralExpertWrapper(expert, device)

def replace_attn_layers(
    model: MixtralForCausalLM,
    config: MixtralConfig,
    device: torch.device,
) -> None:

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_key_value_heads = config.num_key_value_heads

    shapes = [
        (hidden_size, num_heads * head_dim),
        (hidden_size, num_key_value_heads * head_dim),
        (hidden_size, num_key_value_heads * head_dim),
        (num_heads * head_dim, hidden_size),
    ]


    for layer in model.model.layers:
        layer.block_sparse_moe.gate = nn.Linear(  #创建一个门控
            config.hidden_size,
            config.num_local_experts,
            dtype=torch.float16,
            device=device,
            bias=False,
        )




def load_00_expert_state_dict(states_dir: str, device: torch.device):
    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        module_idx = f"model.layers.0.block_sparse_moe.experts.0"
        state_fpath = json.load(f)["weight_map"][f"{module_idx}.w1.weight"]
    return load_file(os.path.join(states_dir, state_fpath), device=str(device))


def build_model(
    device: torch.device,
    offload_config: OffloadConfig,
    state_path: str,
):
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    state_dict_00 = load_00_expert_state_dict(state_path, device) #确定开头的参数

    def _make_module():
        config = AutoConfig.from_pretrained(model_name)
        expert = make_empty_expert(config)
        expert.load_state_dict(state_dict_00)
        expert.half()
        return MixtralExpertWrapper(expert, device=device)

    with device, with_default_dtype(torch.float16):
        model = MixtralForCausalLM(
            AutoConfig.from_pretrained(
                model_name,
                num_local_experts=0,
                torch_dtype=torch.float16,
                device_map=device,
            ),
        )

    model_config = AutoConfig.from_pretrained(model_name)
    replace_attn_layers(model, model_config, device)
    state_index_path = os.path.join(state_path, "model.safetensors.index.json")
    with open(state_index_path) as f:
        weight_map = json.load(f)["weight_map"]

    trunk_state_path = os.path.join(
        state_path,
        weight_map["model.embed_tokens.weight"],
    )
    print("////load state dict////")
    model.load_state_dict(load_file(trunk_state_path, device=str(device)), strict=True)
    print_gpu_memory()
    print("////expert cache////")
    memory = psutil.virtual_memory()
    print(f"Available memory before expert catch: {memory.available / (1024**3):.2f} GB")
    expert_cache = ExpertCache(
        make_module=_make_module,
        main_size=offload_config.main_size,
        offload_size=offload_config.offload_size,
        buffer_size=offload_config.buffer_size,
    )
    memory = psutil.virtual_memory()
    print(f"Available memory after expert catch: {memory.available / (1024**3):.2f} GB")
    print_gpu_memory()
    print("////do layer////") 
    memory = psutil.virtual_memory()
    print(f"Available memory before do layer: {memory.available / (1024**3):.2f} GB")
    for layer_idx in trange(model_config.num_hidden_layers, desc="Loading experts"): #处理每一层
        curr_layer = model.model.layers[layer_idx]
        curr_layer.block_sparse_moe = SparseMoeWrapper(
            model_config,
            layer_idx,
            curr_layer.block_sparse_moe.gate,
            expert_cache,
        )

        for expert_idx in range(model_config.num_local_experts):
            do_offload = expert_idx < offload_config.offload_per_layer

            expert_wrapper = make_and_load_expert_wrapper(
                config=model_config,
                states_dir=state_path,
                expert_uid=(layer_idx, expert_idx),
                device=device,
            )

            expert_cache.add_expert(
                uid=(layer_idx, expert_idx),
                module=expert_wrapper,
                eviction_group=layer_idx,
                offload=do_offload,
            )

            del expert_wrapper
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    memory = psutil.virtual_memory()
    print(f"Available memory after do layer: {memory.available / (1024**3):.2f} GB")
    return model
