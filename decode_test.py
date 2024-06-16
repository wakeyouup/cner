import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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
import config
import pandas as pd

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':

    set_seed(42)

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
   # state = torch.load('/home/xg/life_long_NER2/save/multitask/life_long/train_model_1103202152/misc_4_0.154.pt',  map_location=torch.device('cpu'))["model_state_dict"]
    state = torch.load('/home/xg/life_long_NER2/save/multitask/life_long/train_model_1103202152/misc_4_0.154.pt')["model_state_dict"]
    model = UnifiedQG(config)
    model.load_state_dict(state)
    model.to(config.device)
    input = "Instruction: please extract predefined entities and their types from the input sentence. Predefined entity type: person. Input sentence:Stefano Bordon is out through illness and Coste said he had dropped back row Corrado Covi , who had been recalled for the England game after five years out of the national team ."
    data = tokenizer.encode_plus(input, max_length=512, pad_to_max_length=True, return_tensors="pt", truncation=True)
    beam_outputs = model.model.generate(
        input_ids=data["input_ids"].cuda(), attention_mask=data["attention_mask"].cuda(),
        do_sample=True,
        max_length=32,
        top_k=4,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=1
    )
    sent = tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    sent = sent[:-1] + " " + sent[-1]
    print("inference:", sent)

    # test(model.model, out_dir, test_file)

