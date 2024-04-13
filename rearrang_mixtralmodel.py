import torch
from safetensors import safe_open
from safetensors.torch import save_file
import json


path = "/scratch/bcjw/bhuang4/huggingface_model/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/1e637f2d7cb0a9d6fb1922f305cb784995190a83"
savepath = "/scratch/bcjw/bhuang4/huggingface_model/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/rearranged"
jsonfile = "/scratch/bcjw/bhuang4/huggingface_model/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/rearranged/model.safetensors.index.json"
# with open(jsonfile, 'r') as file:
#     json_data = json.load(file)

#save all experts
for layer in range(0,32): 
    print(f"==layer:{layer}==")
    for expert in range(0,8):
        print(f"expert:{expert}")
        expert_tensors = {}
        next_expert = False
        index = layer*8+expert+2
        save_name = f"model-{index:05}-of-00257.safetensors"
        for fileIndex in range(1,20):
            fname = f"{path}/model-{fileIndex:05}-of-00019.safetensors"
            with safe_open(fname, framework="pt", device="cuda") as f:
                for key in f.keys():
                    if f"model.layers.{layer}.block_sparse_moe.experts.{expert}" in key:
                        expert_tensors[key.replace(f"model.layers.{layer}.block_sparse_moe.experts.{expert}.",'')] = f.get_tensor(key)
                        # json_data['weight_map'][key] = save_name
                    if len(expert_tensors) == 3:
                        next_expert = True
                        break
            if next_expert:                
                save_file(expert_tensors, f"{savepath}/{save_name}")       
                break


#save other components
# other_tensor = {}
# other_list = ['gate.weight','input_layernorm.weight','post_attention_layernorm.weight','self_attn.k_proj.weight','self_attn.o_proj.weight','self_attn.q_proj.weight','self_attn.v_proj.weight','lm_head.weight','model.embed_tokens.weight','model.norm.weight']
# print("==other tensor begin==")
# for fileIndex in range(1,20):
#     print(f"in file {fileIndex}")
#     fname = f"{path}/model-{fileIndex:05}-of-00019.safetensors"
#     with safe_open(fname, framework="pt", device="cuda") as f:
#         for key in f.keys():
#             if any(s in key for s in other_list):
#                 other_tensor[key] = f.get_tensor(key)
#                 json_data['weight_map'][key] = f"model-{1:05}-of-00257.safetensors"
# save_file(other_tensor, f"{savepath}/model-{1:05}-of-00257.safetensors")

# with open(jsonfile, 'w') as file:
#     json.dump(json_data, file, indent=4)

# print("==other tensor down==")




# tensors = {}
# with safe_open("/scratch/bcjw/bhuang4/huggingface_model/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/rearranged/model-00001-of-000257.safetensors", framework="pt", device=0) as f:
#     for k in f.keys():
#         tensors[k] = f.get_tensor(k)
#     print("done")
