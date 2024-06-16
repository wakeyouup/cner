"""
1、能够读取一个任务的数据
2、能够合并另一个任务的数据
3、能够合并examplar的数据
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

import copy
import os
import json
import math


class LifeLongDataset(Dataset):
    def __init__(self, data_dir, type_path, tokenizer, batch_size, device,
                 src_max_length=256, tgt_max_length=256, shuffle=False, debug=False):
        super(LifeLongDataset, self).__init__()
        self._src_max_length = src_max_length
        self._tgt_max_length = tgt_max_length
        self._path = os.path.join(data_dir, type_path + ".csv")   # the path of certain data file
        print(self._path)
        self.tokenizer = tokenizer
        self.data = {"input":[], "target":[], "input_ori":[], "target_ori":[]}
        self._source_column = "input"
        self._target_column = "target"
        self._shuffle = shuffle
        self._batch_size = batch_size
        self.device = device
        self.debug = debug

        # 创建实例时会自动执行下列语句
        self.read_corpus()

        if debug:
            self.data["input"] = self.data["input"][:100]
            self.data["target"] = self.data["target"][:100]
            self.data["input_ori"] = self.data["input_ori"][:100]
            self.data["target_ori"] = self.data["target_ori"][:100]

        if shuffle:
            self.shuffle()

    def shuffle(self):
        # 创建一个随机序列，然后根据这个序列将数据打乱
        indexes = np.random.permutation(len(self.data["input"]))
        # print(indexes)
        # print(self.data["input"])
        tmp = copy.deepcopy(self.data)
        self.data = {"input": [tmp["input"][i] for i in indexes],
                     "target":[tmp["target"][i] for i in indexes],
                     "input_ori":[tmp["input_ori"][i] for i in indexes],
                     "target_ori":[tmp["target_ori"][i] for i in indexes]}

    def read_corpus(self):
        corpus = pd.read_csv(self._path)
        inp_src = corpus[self._source_column].tolist()
        tgt_src = corpus[self._target_column].tolist()
        if self.debug:
            inp_src = inp_src[:100]
            tgt_src = tgt_src[:100]
        # 添加</s>
        inp = [i.strip() + " </s>" for i in inp_src]
        tgt = [t.strip() + " </s>" for t in tgt_src]
        # tokenize
        inp_tokenized = [self.tokenizer.batch_encode_plus(
            [i], max_length=self._src_max_length, pad_to_max_length=True, return_tensors="pt", truncation=True
        ) for i in inp]
        tgt_tokenized = [self.tokenizer.batch_encode_plus(
            [t], max_length=self._tgt_max_length, pad_to_max_length=True, return_tensors="pt", truncation=True
        ) for t in tgt]

        self.data["input"].extend(inp_tokenized)
        self.data["target"].extend(tgt_tokenized)
        self.data["input_ori"].extend(inp_src)
        self.data["target_ori"].extend(tgt_src)

        print("finish reading from ", self._path)

    def __getitem__(self, item):
        source_ids = self.data["input"][item]["input_ids"].squeeze()
        target_ids = self.data["target"][item]["input_ids"].squeeze()

        source_mask = self.data["input"][item]["attention_mask"].squeeze()
        target_mask = self.data["target"][item]["attention_mask"].squeeze()

        input_ori = self.data["input_ori"][item]
        target_ori = self.data["target_ori"][item]

        return source_ids.to(self.device), source_mask.to(self.device), target_ids.to(self.device), target_mask.to(self.device), input_ori, target_ori

    def __len__(self):
        return len(self.data["input"])

    def next_batch(self, index):  # 由于我们时常要变换dataset的大小，所以我们不使用dataloader了
        assert index <= self.__len__() // self._batch_size, "index out of range"
        if index < math.floor(self.__len__() // self._batch_size):
            inp_ids = [item["input_ids"].squeeze() for item in self.data["input"][index * self._batch_size: (index + 1) * self._batch_size]]
            inp_attn = [item["attention_mask"].squeeze() for item in self.data["input"][index * self._batch_size: (index + 1) * self._batch_size]]
            tgt_ids = [item["input_ids"].squeeze() for item in self.data["target"][index * self._batch_size: (index + 1) * self._batch_size]]
            tgt_attn = [item["attention_mask"].squeeze() for item in self.data["target"][index * self._batch_size: (index + 1) * self._batch_size]]
            input_ori = [item for item in self.data["input_ori"][index * self._batch_size: (index + 1) * self._batch_size]]
            target_ori = [item for item in self.data["target_ori"][index * self._batch_size: (index + 1) * self._batch_size]]
        else:
            inp_ids = [item["input_ids"].squeeze() for item in self.data["input"][index * self._batch_size: ]]
            inp_attn = [item["attention_mask"].squeeze() for item in self.data["input"][index * self._batch_size: ]]
            tgt_ids = [item["input_ids"].squeeze() for item in self.data["target"][index * self._batch_size: ]]
            tgt_attn = [item["attention_mask"].squeeze() for item in self.data["target"][index * self._batch_size: ]]
            input_ori = [item for item in self.data["input_ori"][index * self._batch_size: ]]
            target_ori = [item for item in self.data["target_ori"][index * self._batch_size: ]]

        # 向量化
        # print("input_ids:", len(inp_ids), " current index: ",
        #       index, "total batch_size", math.floor(self.__len__() // self._batch_size),
        #       "total item:", self.__len__())

        inp_ids_tensor = torch.stack(inp_ids).to(self.device)
        # print(inp_ids_tensor.size())
        inp_attn_tensor = torch.stack(inp_attn).to(self.device)
        tgt_ids_tensor = torch.stack(tgt_ids).to(self.device)
        tgt_attn_tensor = torch.stack(tgt_attn).to(self.device)

        return inp_ids_tensor, inp_attn_tensor, tgt_ids_tensor, tgt_attn_tensor, input_ori, target_ori

    def merge_with_dataset(self, dataset, shuffle=False):
        self.data["input"].extend(dataset.data["input"])
        self.data["target"].extend(dataset.data["target"])
        self.data["input_ori"].extend(dataset.data["input_ori"])
        self.data["target_ori"].extend(dataset.data["target_ori"])

        if shuffle:
            self.shuffle()

    def merge_with_examplars(self, examplar, shuffle=False):
        self.merge_with_dataset(examplar, shuffle)  # 直接调用merge_with_dataset

    def get_data_by_indexes(self, indexes):
        selected_data = {k:[] for k, _ in self.data.items()}
        keys = list(self.data.keys())
        for i in indexes:
            for k in keys:
                selected_data[k].append(self.data[k][i])
        return copy.deepcopy(selected_data)

class Examplar(object):

    def __init__(self, data, device):
        """
        :param data: {input:..., target:...}
        """
        self.data = data
        self.device = device

    def merge(self, input_examplars):  # 此处无需打乱，反正后面和dataset融合时候打乱
        self.data["input"].extend(input_examplars.data["input"])
        self.data["target"].extend(input_examplars.data["target"])
        self.data["input_ori"].extend(input_examplars.data["input_ori"])
        self.data["target_ori"].extend(input_examplars.data["target_ori"])

    def size(self):
        return len(self.data["input"])

    def save(self, data_file):
        with open(data_file, "w") as f:  # 到底用w+还是w需要商讨
            json.dump(self.data, f)

    def load(self, data_file):
        with open(data_file, "r") as f:
            self.data = json.load(f)

    def shuffle(self):

        indexes = np.random.permutation(len(self.data["input"]))
        tmp = copy.deepcopy(self.data)
        self.data = {"input": [tmp["input"][i] for i in indexes],
                     "target":[tmp["target"][i] for i in indexes],
                     "input_ori":[tmp["input_ori"][i] for i in indexes],
                     "target_ori":[tmp["target_ori"][i] for i in indexes]}

    def __getitem__(self, item):
        source_ids = self.data["input"][item]["input_ids"].squeeze()  # 去掉维度为一的维度
        target_ids = self.data["target"][item]["input_ids"].squeeze()

        source_mask = self.data["input"][item]["attention_mask"].squeeze()
        target_mask = self.data["target"][item]["attention_mask"].squeeze()

        input_ori = self.data["input_ori"][item]
        target_ori = self.data["target_ori"][item]
        # print("source ids张这个样子", source_ids)
        # source_ids_tensor = torch.tensor(source_ids, device=self.device)
        # target_ids_tensor = torch.tensor(target_ids, device=self.device)
        # source_mask_tensor = torch.tensor(source_mask, device=self.device)
        # target_mask_tensor = torch.tensor(target_mask, device=self.device)

        return source_ids.to(self.device), source_mask.to(self.device), target_ids.to(self.device), target_mask.to(self.device), input_ori, target_ori



# class Task(object):
#     """用于管理各个人物的dataset、examplar，以及合并dataset"""
#     def __init__(self, train_dataset:LifeLongDataset, val_dataset:LifeLongDataset,
#                  # test_dataset:LifeLongDataset,
#                  task_name, train_examplars:Examplar, valid_examplars:Examplar):
#         self.datasets = {task_name: {"train": train_dataset, "valid": val_dataset,
#                                      # "test":test_dataset
#                                      }}
#         self.task_name_set = [task_name]
#         self.task_name = task_name
#         self.examplars = {task_name: {"train": train_examplars, "valid": valid_examplars}}
#         self._merged_examplars = {"train": train_examplars, "valid": valid_examplars}
#
#         self._merged_dataset = {"train": copy.deepcopy(train_dataset), "valid": copy.deepcopy(val_dataset),
#                                 # "test": copy.deepcopy(test_dataset)
#                                 }
#
#     def merge_dataset(self, train_dataset, val_dataset,
#                       # test_dataset,
#                       shuffle=False):
#         self._merged_dataset["train"].merge_with_dataset(train_dataset, shuffle)
#         self._merged_dataset["valid"].merge_with_dataset(val_dataset, shuffle)
#         # self._merged_dataset["test"].merge_with_dataset(test_dataset, shuffle)
#
#     def add_dataset(self, task_name, train_dataset, val_dataset,
#                     # test_dataset
#                     ):
#         self.datasets[task_name] = {"train": train_dataset, "valid": val_dataset,
#                                     # "test":test_dataset
#                                     }
#
#     def add_task_name(self, task_name):
#         self.task_name_set.append(task_name)
#         self.task_name = " ".join(self.task_name_set)
#
#     def merge_examplars(self, train_examplar, valid_examplar):
#         self._merged_examplars["train"].merge(train_examplar)
#         self._merged_examplars["valid"].merge(valid_examplar)
#
#     def add_examplars(self, task_name, train_examplar, valid_examplar):
#         self.examplars[task_name] = {"train":train_examplar, "valid": valid_examplar}
#
#     def merge_task(self, task, shuffle=False):
#         """the task must be single task"""
#         self.add_dataset(task.task_name, task.datasets[task.task_name]["train"],
#                          task.datasets[task.task_name]["valid"],
#                          # task.datasets[task.task_name]["test"]
#                          )
#         self.add_task_name(task.task_name)
#         self.merge_dataset(copy.deepcopy(task.datasets[task.task_name]["train"]),
#                            copy.deepcopy(task.datasets[task.task_name]["valid"]),
#                            # copy.deepcopy(task.datasets[task.task_name]["test"]),
#                            shuffle)
#         self.merge_examplars(copy.deepcopy(task.examplars[task.task_name]["train"]),
#                              copy.deepcopy(task.examplars[task.task_name]["valid"]))
#         self.add_examplars(task.task_name, task.examplars[task.task_name]["train"],
#                            task.examplars[task.task_name]["valid"])
#
#     def get_merged_dataset_with_examplars(self):  # 此处默认shuffle
#         """返回当前所有任务以及所有examplars的融合了的dataset"""
#         train_set = copy.deepcopy(self._merged_dataset["train"]).merge_with_examplars(self._merged_examplars["train"], True)  # 使用deepcopy,这样merged_dataset不会影响
#         dev_set = copy.deepcopy(self._merged_dataset["valid"]).merge_with_examplars(self._merged_examplars["valid"], True)
#         # test_set = copy.deepcopy(self._merged_dataset["test"])
#         # return train_set, dev_set, test_set
#         return train_set, dev_set
#
#
#     def get_current_task_dataset(self, task_name):
#         return copy.deepcopy(self.datasets[task_name])

class Task(object):
    """用于管理各个任务的dataset、examplar，以及合并dataset"""
    def __init__(self, train_dataset:LifeLongDataset, val_dataset:LifeLongDataset,
                 task_name, train_examplars:Examplar, valid_examplars:Examplar):
        self.datasets = {task_name: {"train": train_dataset, "valid": val_dataset}}
        self.examplars = {task_name: {"train": train_examplars, "valid": valid_examplars}}

    def add_dataset(self, task_name, train_dataset, val_dataset):
        self.datasets[task_name] = {"train": train_dataset, "valid": val_dataset}

    def add_examplars(self, task_name, train_examplar, valid_examplar):
        self.examplars[task_name] = {"train":train_examplar, "valid": valid_examplar}

    def merge_task(self, task_name, task):
        """the task must be single task"""
        self.add_dataset(task_name, task.datasets[task_name]["train"],
                         task.datasets[task_name]["valid"])
        self.add_examplars(task_name, task.examplars[task_name]["train"],
                           task.examplars[task_name]["valid"])

    def get_current_task_dataset(self, task_name):
        return copy.deepcopy(self.datasets[task_name])

    def get_merged_task_dataset(self, task_name_set):
        """将所有dataset合并，一般用于multi-task learning"""
        for task_name in task_name_set:
            if task_name not in self.datasets.keys():
                assert False, "出错了，task{}中不包含要取的task name {}的dataset".format(list(self.datasets.keys()), task_name_set)

        # 合格，开始合并
        merged_dataset = None
        # merged_dataset = Examplar(device="cuda:0")
        for index, task_name in enumerate(task_name_set):
            if index == 0:
                merged_dataset = copy.deepcopy(self.datasets[task_name])  # {"train": train_dataset, "valid": val_dataset}
            else:
                merged_dataset["train"].merge_with_dataset(copy.deepcopy(self.datasets[task_name]["train"]))
                merged_dataset["valid"].merge_with_dataset(copy.deepcopy(self.datasets[task_name]["valid"]))
        return merged_dataset

    def get_merged_task_examplar(self, task_name_set):
        """将所有examplars合并，一般用于life long learning"""
        for task_name in task_name_set:
            if task_name not in self.examplars.keys():
                assert False, "出错了，task{}中不包含要取的task name {}的examplars".format(list(self.examplars.keys()), task_name_set)

        # 合格，合并
        merged_examplars = None
        # merged_examplars = Examplar()
        for index, task_name in enumerate(task_name_set):
            if index == 0:
                merged_examplars = copy.deepcopy(self.examplars[task_name])
                # merged_examplars.merge(self.examplars[task_name])
            else:
                merged_examplars["train"].merge(copy.deepcopy(self.examplars[task_name]["train"]))
                merged_examplars["valid"].merge(copy.deepcopy(self.examplars[task_name]["valid"]))
        return merged_examplars


def generate_task(train_datasets, valid_datasets, train_examplars, valid_examplars, task_names):
    assert len(task_names) == len(train_datasets), "task num is not equal to train set num"
    assert len(task_names) == len(valid_datasets), "task num is not equal to valid set num"
    assert len(task_names) == len(train_examplars), "task num is not equal to train examplars num"
    assert len(task_names) == len(valid_examplars), "task num is not equal to valid examplars num"

    main_task = Task(train_datasets[0], valid_datasets[0], task_names[0], train_examplars[0], valid_examplars[0])

    if len(task_names) > 1:
        # 说明是用在multi-task learning上面
        for i in range(1, len(task_names)):
            task = Task(train_datasets[i], valid_datasets[i], task_names[i], train_examplars[i], valid_examplars[i])
            main_task.merge_task(task_names[i], task)  # do not shuffle

    return main_task





