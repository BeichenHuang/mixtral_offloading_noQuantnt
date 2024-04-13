import torch
from safetensors import safe_open
from safetensors.torch import save_file

tensors = {}
path = "/scratch/bcjw/bhuang4/huggingface_model/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/rearranged"
with safe_open(f"{path}/model-00001-of-00257.safetensors", framework="pt", device="cuda") as f:
   for key in f.keys():
       tensors[key] = f.get_tensor(key)
       print("loaded")