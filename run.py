import os
import random
import torch
import numpy as np

import sys
import traceback
import time

import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from loader.dataset import Task, LifeLongDataset, Examplar, generate_task
from construct_examplars import construct_examplars
from model import UnifiedQG
import copy
from ewc import EWC
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
#from transformers import BartTokenizer, BartForConditionalGeneration
import time
import config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read(task_sequence, config, tokenizer, debug=False):
    """读取所有task的dataset, 初始化model"""
    # model = UnifiedQG(config)
    # model.to(config.device)

    train_dataset_list = []
    valid_dataset_list = []
    train_examplars_list = []
    valid_examplars_list = []
    for task_name in task_sequence:
        train_dataset = LifeLongDataset(config.data_dir + task_name, "train", tokenizer,
                        config.batch_size, config.device, config.src_max_length,
                        config.tgt_max_length, shuffle=True, debug=debug)
        valid_dataset = LifeLongDataset(config.data_dir + task_name, "dev_new", tokenizer,
                        config.batch_size, config.device, config.src_max_length,
                        config.tgt_max_length, shuffle=True, debug=debug)
        train_dataset_list.append(train_dataset)
        valid_dataset_list.append(valid_dataset)
        train_examplars_list.append(Examplar({"input":[], "target":[], "input_ori":[], "target_ori":[]}, config.device)) # 刚开始没有examplar，用空代替
        valid_examplars_list.append(Examplar({"input":[], "target":[], "input_ori":[], "target_ori":[]}, config.device))

    task = generate_task(train_dataset_list, valid_dataset_list, train_examplars_list, valid_examplars_list, task_sequence)
    # optimizer = configure_optimizers(model, config.weight_decay, config.learning_rate, config.adam_epsilon)
    return task

def configure_optimizers(model, weight_decay, learning_rate, adam_epsilon):
    "Prepare optimizer and schedule (linear warmup and decay)"

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    return optimizer

def get_model_and_solver(config):
    model = UnifiedQG(config)
    model.to(config.device)
    optimizer = configure_optimizers(model, config.weight_decay, config.learning_rate, config.adam_epsilon)
    return model, optimizer

# 一个参数代表当前任务的数据集，一个是从过去选出来的很小一部分代表性的数据集
# def calculate_ewc_weight(current_dataset:LifeLongDataset, current_examplars:Examplar):
#     cur_data = copy.deepcopy(current_dataset.data)
#     old_data = copy.deepcopy(current_examplars.data)
#     cur_data = cur_data["input_ori"].extend(cur_data["target_ori"])
#     old_data = old_data["input_ori"].extend(old_data["target_ori"])
#     cur_data = " ".join(cur_data)
#     old_data = " ".join(old_data)
#     corpus = [old_data, cur_data]
#     from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
#     from sklearn.metrics.pairwise import cosine_similarity
#     vectorizer = CountVectorizer()  # Tf搭配余弦距离变动小，Countvectorizer搭配W距离，
#     text_counts = vectorizer.fit_transform(corpus)
#     text_counts = text_counts.todense()
#     return cosine_similarity(text_counts[0], text_counts[1])  # 一般在0.9左右

def evaluation(config, task_name, model:UnifiedQG, dataset:LifeLongDataset):
    model.eval()
    total_loss = 0.
    if len(dataset) % config.batch_size == 0:
        num_batches = len(dataset) // config.batch_size
    else:
        num_batches = len(dataset) // config.batch_size + 1
    with torch.no_grad():
        for i in range(num_batches):
            inp_ids_tensor, inp_attn_tensor, tgt_ids_tensor, tgt_attn_tensor, input_ori, target_ori = dataset.next_batch(i)
            loss = model.get_loss(inp_ids_tensor, inp_attn_tensor, tgt_ids_tensor, tgt_attn_tensor)
            total_loss += loss.detach().item()
            if i % config.print_iters_eval == 0:
                print(f"eval the task of {task_name} in iteration {i}/{num_batches}, original loss: {loss.item()}")
    total_loss /= num_batches
    print(f"eval the task of {task_name}, total average loss: {total_loss}")
    return total_loss


def train_epoch(config, task_name, model:UnifiedQG, optimizer, train_dataset:LifeLongDataset,
                examplars:Examplar, ewc_rate=None, schedule=None):
    """pytorch提供一种模式来切换模型训练和推断的模式（一般在训练开始前写上model.train()-用来启用batch normalization和dropout,推断开始前写上model.eval()）"""
    model.train()
    if config.experiment_type != "upper_bound" and examplars.size() != 0:
        print(type(examplars))
       # examplar = Examplar(data=examplars, device="cuda:0")
        ewc = EWC(model, examplars, config)

    total_loss = 0.
    if len(train_dataset) % config.batch_size == 0:
        num_batches = len(train_dataset) // config.batch_size
    else:
        num_batches = len(train_dataset) // config.batch_size + 1
    for i in range(num_batches):
        inp_ids_tensor, inp_attn_tensor, tgt_ids_tensor, tgt_attn_tensor, input_ori, target_ori = train_dataset.next_batch(i)

        """feed forward"""

        loss = model.get_loss(inp_ids_tensor, inp_attn_tensor, tgt_ids_tensor, tgt_attn_tensor)
        original_loss = loss.detach().item()
        if config.experiment_type != "upper_bound" and examplars.size() != 0:
            ewc_loss = ewc.penalty(model)
            if config.adaptive:
                loss = model.ewc_loss(loss, ewc_loss, ewc_rate * config.ewc_importance)
            else:
                loss = model.ewc_loss(ewc_loss, config.ewc_importance)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.max_grad_norm)
        optimizer.step()
        if schedule is not None:
            schedule.step()
        model.zero_grad()
        if i % config.print_iters == 0:
            if config.experiment_type != "upper_bound" and examplars.size() != 0:
                print(f"******the task of {task_name} in iteration {i}/{num_batches}, original loss: {original_loss}, ewc loss: {ewc_loss}****")
            else:
                print(f"******the task of {task_name} in iteration {i}/{num_batches}, original loss: {original_loss}******")
        total_loss += loss.detach().item()
    total_loss /= num_batches
    print(f"the task of {task_name}'s total average loss is {total_loss}")
    return total_loss

def find_best_model(model_dir, task_name):
    import glob
    # 列表形式返回所有的文件路径
    all_models = sorted(glob.glob(os.path.join(model_dir, task_name + "*.pt")))
    best_model = None
    best_loss = 100000
    print("all model:", all_models)
    # 找loss最小的那个，因为loss已经在模型保存的时候被保存进去了，返回的是最小loss的那个路径
    for model in all_models:
        loss = float(model.strip().split("/")[-1].split("_")[-1][:-3])
        if loss < best_loss:
            best_loss = loss
            best_model = model
    print("best model:", best_model)
    return best_model

def train(config):
    # 模型保存的主路径，同级目录下的save/multitask/lifelong
    train_dir = os.path.join("save/multitask", config.experiment_type)
    model_dir = os.path.join(train_dir, "train_model_%d" % int(time.strftime("%m%d%H%M%S")))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    examp_dir = os.path.join(train_dir, "train_examplar_%d" % int(time.strftime("%m%d%H%M%S")))
    if not os.path.exists(examp_dir):
        os.makedirs(examp_dir)

    task_sequence = config.task_sequence   # sequence of task name, python list
    print("start training")
    # 无论是upper_bound还是其他的，我都用这种task，只是说upper_bound用merged dataset，life-long用merged examplar
    tokenizer = T5Tokenizer.from_pretrained(config.tokenizer_name_or_path)
    start_time = time.time()
    task = read(task_sequence, config, tokenizer, config.debug)  #用一个自定义的task类来管理各个数据集（字典的形式保存每个数据集dataset对象和其所挑选出的examplar）
    end_time = time.time()
    print(f"@@@@@@@@@@@ 加载所有数据花了{(end_time - start_time)/60.} 时间")
    state = None
    ewc_weight = None
   # cur_exmaplars = Examplar({"input":[], "target":[], "input_ori":[], "target_ori":[]}, config.device)
    for index, task_name in enumerate(task_sequence):
        best_loss = 100000
        check_point_path = None
        model, optimizer = get_model_and_solver(config)  # 在这里会将加载的模型转移到显存里
        # get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=t_total)
        if config.experiment_type != "upper_bound" and index!=0:  # 如果不是multitask且不是第一个任务，那么需要加载前面的model
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])

        # 在拿到task之后，按照upper_bound或者其他，来得到一轮的训练数据
        if config.experiment_type == "upper_bound":
            # 如果是upper_bound，则使用dataset merge数据
            cur_dataset = task.get_merged_task_dataset(task_sequence[:index+1])  # {dtype:dataset}
        else:
            # 如果是life-long,则使用merged examplar以及当前的dataset
            # 这里的cur_dataset代表当前任务的train和valid集，cur_examplars应当也是过去所有任务的train和valid集合，分别是两个字典
            # 数据进行计算的时候也会转移到显存中
            cur_dataset = task.get_current_task_dataset(task_sequence[index])
            cur_exmaplars = task.get_merged_task_examplar(task_sequence[:index+1])
            print(type(cur_exmaplars))
            cur_train_examplar = cur_exmaplars["train"]

            if index!= 0:
                # 计算出当前的ewc weight
                # ewc_weight = calculate_ewc_weight(cur_dataset["train"], cur_exmaplars["train"])
                ewc_weight = 0.9
            # cur_dataset = {k: v.merge_with_examplars(copy.deepcopy(cur_exmaplars[k])) for k, v in cur_dataset.items()}  # 混合了examplars的dataset
            # 两部分，{"train": LifeLongDataset对象, "valid":examplar对象}
            print("在合并之前:", [len(v) for k, v in cur_dataset.items()])
            for k, v in cur_dataset.items():
                v.merge_with_examplars(copy.deepcopy(cur_exmaplars[k]))
            print("在合并之后:", [len(v) for k, v in cur_dataset.items()])
        # t_total = len(cur_dataset) // config.batch_size * config.train_epochs
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=0, num_training_steps=t_total
        # )
        # 设置save point path
        for epoch in range(config.train_epochs):
            train_dataset = cur_dataset["train"]
            train_dataset.shuffle()
            print(f"===========begin train {task_name} for {epoch}/{config.train_epochs}========")
            train_loss = train_epoch(config, task_name, model, optimizer, train_dataset, cur_train_examplar, ewc_weight)

            """validation"""
            valid_dataset = cur_dataset["valid"]
            valid_loss = evaluation(config, task_name, model, valid_dataset)
            # 只有在验证集上的损失小于目前的最优损失时，才会去保存模型的节点
            if valid_loss < best_loss:
                # 保存节点
                check_point_path = os.path.join(model_dir, f"{task_name}_{epoch}_{str(round(valid_loss, 3))}.pt")
                torch.save({"model_state_dict":model.state_dict(),
                            "optimizer_state_dict":optimizer.state_dict()},
                           check_point_path)
                print(f"===successfully save model in {task_name}, epoch{epoch} with loss {valid_loss}==")
                best_loss = valid_loss
                early_stop = 0
            else:
                print(f"^^^^^^So sad. Loss does not decrease in {task_name}, epoch: {epoch}^^^^^")
                early_stop += 1
            if early_stop == 3:
                print("%%%%%%%%%%%%%early_stop%%%%%%%%%%%%%%%")
                break
        # finish one task, load the best model of the current task from early stop to compute exemplar
        best_model = copy.deepcopy(model)
        best_optimizer = copy.deepcopy(optimizer)
        state = torch.load(find_best_model(model_dir, task_name))  # 让下一个任务也是从best model开始
        best_model.load_state_dict(state["model_state_dict"])
        best_optimizer.load_state_dict(state["optimizer_state_dict"])
        best_model.to(config.device)
        """construct examplars if required"""
        if config.experiment_type != "upper_bound" and index != len(task_sequence) -1:
            print("&&&&&& construct examplars &&&&&&&&&&")
            examplars = construct_examplars(best_model, task, task_name, config)  # {dtype:examplars}
            task.examplars[task_name] = copy.deepcopy(examplars)
            print(f"{task_name}'s examplars constructed!")
        del best_model
        del best_optimizer



if __name__ == '__main__':
    set_seed(42)
    try:
        train(config)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=None, file=open('Error_Infos_'+str(time.time())[:8]+'.txt'))


