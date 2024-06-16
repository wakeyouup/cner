import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import random
import torch
from model import UnifiedQG
import copy
from ewc import EWC
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time
import config_for_finetune as config
import pandas as pd
from run import find_best_model

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def test(model, out_dir, data_file, tokenizer):
    df = pd.read_csv(data_file)
    src = df["input"].tolist()
    tgt = df["target"].tolist()
    src = [i.strip() + " </s>" for i in src]
    # tgt = [t.strip() + " </s>" for t in tgt]

    f_ori = open(out_dir + "/golden_base.txt", "w", encoding="utf-8")
    f_inf = open(out_dir + "/generated_base.txt", "w", encoding="utf-8")
    for i, data in enumerate(src):
        data = tokenizer.encode_plus(data, max_length=512, pad_to_max_length=True, return_tensors="pt", truncation=True)
        beam_outputs = model.generate(
            input_ids=data["input_ids"].to(config.device), attention_mask=data["attention_mask"].to(config.device),
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
        # print(data["target_ids"].size())
        original = tgt[i].strip()
        original = original[:-1] + " " + original[-1]
        print("original:", original)
        f_ori.write(original.strip() + "\n")
        f_inf.write(sent.strip() + "\n")
    f_inf.close()
    f_ori.close()


def auto_test(models_dir, out_path):
    task_sequence = config.task_sequence
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    for index, task in enumerate(task_sequence):
        # 每个task找到最佳的那个model，然后生成这个task以及之前的所有task的生成文件
        save_dir = os.path.join(out_path, task)  # 比如/root/result/task_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = find_best_model(models_dir, task)
        state = torch.load(model_path)["model_state_dict"]
        model = UnifiedQG(config)
        model.load_state_dict(state)
        model.to(config.device)
        model.eval()
        with torch.no_grad():
            for cur_i, cur_task in enumerate(task_sequence[:index+1]):
                print(f"**********进行到了{index}, {cur_i}***task name: {task}, cur_task_name: {cur_task}*********")
                test_file = os.path.join(config.data_dir + cur_task, "test_new" + ".csv")
                cur_save_dir = os.path.join(save_dir, cur_task)
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                test(model.model, cur_save_dir, test_file, tokenizer)
        if index == 3:  # FIX ME: 在这里，我们只运行前4个
            break



# auto_test("/root/life_long_QG_copy/save/life_long_ewc_100000/life_long/train_model_813162156", "/root/life_long_QG_copy/result/life_long_ewc_100000")
auto_test("/root/life_long_QG/save/finetune/finetune/train_model_907111041", "/root/life_long_QG/result/finetune_new_series")



# if __name__ == '__main__':
#
#     set_seed(42)
#
#     tokenizer = T5Tokenizer.from_pretrained('t5-base')
#     state = torch.load('/root/life_long_QG/save/life_long/upper_bound/train_model_710143304/squad-v1_1_1_1.279.pt')["model_state_dict"]
#     print(state.keys())
#     model = UnifiedQG(config)
#     model.load_state_dict(state)
#     model.to(config.device)
#     test_file = os.path.join(config.data_dir + "squad-v1_1", "test_new" + ".csv")
#     out_dir = "result/life_long/squad-v1_1"
#     test(model.model, out_dir, test_file)

# 0
# 0, 1
# 0, 1, 2
# 0, 1, 2, 3
# 0, 1, 2, 3, 4
# 0, 1, 2, 3, 4, 5
# 0, 1, 2, 3, 4, 5, 6
# 0, 1, 2, 3, 4, 5, 6, 7  总共是 36