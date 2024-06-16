import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random
import torch
import numpy as np

from loader.dataset import Task, LifeLongDataset, Examplar, generate_task
from construct_examplars import construct_examplars
from model import UnifiedQG
import copy
from ewc import EWC
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time
import config1 as config
import pandas as pd

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def test(model, out_dir, data_file):
    df = pd.read_csv(data_file)
    src = df["input"].tolist()
    tgt = df["target"].tolist()
    src = [i.strip() + " </s>" for i in src]
    # tgt = [t.strip() + " </s>" for t in tgt]

    f_ori = open(out_dir + "/golden_s6_norp.txt", "w", encoding="utf-8")
    f_inf = open(out_dir + "/generated_s6_norp.txt", "w", encoding="utf-8")
    for i, data in enumerate(src):
        data = tokenizer.encode_plus(data, max_length=512, pad_to_max_length=True, return_tensors="pt", truncation=True)
        beam_outputs = model.generate(
            input_ids=data["input_ids"].cuda(), attention_mask=data["attention_mask"].cuda(),
            do_sample=True,
            max_length=32,
            top_k=4,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=1
        )
        sent = tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
       # sent = sent[:-1] + " " + sent[-1]
        print("inference:", sent)
        # print(data["target_ids"].size())
        original = tgt[i].strip()
       # original = original[:-1] + " " + original[-1]
        print("original:", original)
        f_ori.write(original.strip() + "\n")
        f_inf.write(sent.strip() + "\n")
    f_inf.close()
    f_ori.close()

if __name__ == '__main__':

    set_seed(42)

    tokenizer = T5Tokenizer.from_pretrained(config.tokenizer_name_or_path)
   # state = torch.load('/home/xg/life_long_NER2/save/multitask/life_long/train_model_1125014534/per_3_0.183.pt')["model_state_dict"]
   # state = torch.load('/home/xg/life_long_NER2/save/life_long_all_diff/life_long/train_model_1125023312/card_1_0.155.pt', map_location=torch.device('cuda:0'))["model_state_dict"]
   # state = torch.load('/home/xg/life_long_NER2/save/life_long_without_er/life_long/train_model_1125101353/gpe_7_0.087.pt', map_location=torch.device('cuda:0'))["model_state_dict"]
    state = torch.load('/home/xg/life_long_NER2/save/life_long/life_long/train_model_1125101725/norp_2_0.053.pt', map_location=torch.device('cuda:0'))["model_state_dict"]

    model = UnifiedQG(config)
    model.load_state_dict(state)
    model.to(config.device)
    test_file = os.path.join(config.data_dir, "test_norp" + ".csv")
    out_dir = "result_for_generate/ontonote_wewc"
    test(model.model, out_dir, test_file)

