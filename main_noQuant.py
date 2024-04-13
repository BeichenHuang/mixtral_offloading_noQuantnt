import numpy
import torch
from torch.nn import functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import math
import pandas as pd
from tqdm import tqdm
import pickle
import eval_mmlu
import psutil
import time
from huggingface_hub import snapshot_download
from tqdm.auto import trange
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import logging as hf_logging
from srcNoQuant.build_model import OffloadConfig,  build_model


torch.cuda.empty_cache()

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
quantized_model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
state_path = "/scratch/bcjw/bhuang4/huggingface_model/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/rearranged"

config = AutoConfig.from_pretrained(model_name)

device = torch.device("cuda:0")
##### Change this to 5 if you have only 12 GB of GPU VRAM #####
offload_per_layer = 6
print(f"offload_per_layer is {offload_per_layer}")
# offload_per_layer = 5
###############################################################

# df = pd.read_parquet('//home/huangbc/mixtral_offloading/mixtral-offloading/dataset/test-00000-of-00001.parquet')
# texts = df['text'].tolist()

# test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

memory = psutil.virtual_memory()
print(f"Total memory: {memory.total / (1024**3):.2f} GB")
print(f"Available memory: {memory.available / (1024**3):.2f} GB")

num_experts = config.num_local_experts

offload_config = OffloadConfig(
    main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
    offload_size=config.num_hidden_layers * offload_per_layer,
    buffer_size=4,
    offload_per_layer=offload_per_layer,
)


model_begin = time.time()
model = build_model(
    device=device,
    offload_config=offload_config,
    state_path=state_path,
)
model_end = time.time()
print(f"build model time:{model_end - model_begin}s")
tokenizer = AutoTokenizer.from_pretrained(model_name)
##test wikitext2
fname = "/scratch/bcjw/bhuang4/dataset/dataset_mixtraloffloading/wikitext2_test-00000-of-00001.parquet"
df = pd.read_parquet(fname)
texts = df['text'].tolist()
encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
max_length = 2048
stride = 512
seq_len = encodings.input_ids.size(1)

test_begin = time.time()
nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break
test_end = time.time()
print(f"calculate perplexity time:{test_begin - test_end}s")
ppl = torch.exp(torch.stack(nlls).mean())
savename = f"wikitext2_noQuant_offload_{offload_per_layer}_perplexity.pickle"
with open(savename, 'wb') as file:
        pickle.dump(ppl, file)
print('\n ppl is: ', ppl.item())
##test mmlu
# eval_mmlu.eval_mmlu_main(model,tokenizer)
