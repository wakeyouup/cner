import sys
import math
import numpy as np
import torch

from loader.dataset import Examplar
from loader.dataset import Task
from loader.dataset import LifeLongDataset
from model import UnifiedQG

def construct_exemplar_indices_loss(loss, m):
    """
    :param loss: is calculated weights
    :param m: the max number of examplars
    :return: selected ids (list)
    """
    sorted_loss, sorted_index = torch.sort(loss)  # 默认升序

    number = min(m, len(loss))  # 如果总共没多少，那么就按少的选
    selected_ids = sorted_index.tolist()[:number]

    return selected_ids


def construct_examplar_indices_random(m, num_data, selected_ids):
    """ 在我们的实验里，是先调用loss construction再调用random进行填充
    :param m: the max number of examplars
    :param num_data: number of data
    :param selected_ids: already selected ids
    :return: selected ids
    """
    if m == 0:
        return []
    seen_ids = []
    seen_ids.extend(selected_ids)
    ids = []
    number = min(m, num_data - len(selected_ids))  # 如果要的m比剩下的数据还多，则选择少的
    while len(ids) < number:
        tmp = np.random.choice(num_data, 1, replace=False).tolist()[0]
        if tmp not in seen_ids:
            ids.append(tmp)
            seen_ids.append(tmp)  # 放入到seen里面

    return ids


def construct_examplars(model:UnifiedQG, task:Task, taskname, config):
    """
    :param model: unifiedQG model
    :param task: Task instance
    :param taskname: the name of task
    :param config: configuration
    :return: dict {dtype: Examplar}
    """
    model.eval()

    with torch.no_grad():
        dataset = task.get_current_task_dataset(taskname)
        feature_map = dict({"train": [], "valid": []})
        for dtype in ["train", "valid"]:
            corpus = dataset[dtype]  # LifeLongDataset
            if len(corpus) % config.batch_size == 0:
                num_batches = len(corpus) // config.batch_size
            else:
                num_batches = len(corpus) // config.batch_size + 1  # 总共有num_batches个batch
            if config.shuffle:
                corpus.shuffle()

            for i in range(num_batches):
                inp_ids_tensor, inp_attn_tensor, tgt_ids_tensor, tgt_attn_tensor, input_ori, target_ori = corpus.next_batch(i)
                # input_ids, input_attn_mask, target_ids, target_attn_mask
                loss = model.get_loss_for_examplars(inp_ids_tensor, inp_attn_tensor, tgt_ids_tensor, tgt_attn_tensor)
                # 我们的计算法则由loss和target sentence的length一同构成, 但是我们先不用length，只用loss试试看，后面再探索
                feature_map[dtype].append(loss)
            # print("feature map", dtype, feature_map[dtype])
            # print("feature map length", len(feature_map[dtype]))  # 用get_loss是7
            feature_map[dtype] = torch.cat(feature_map[dtype])

        """ ------ Get exemplars data for train and validation ------ """
        m = dict({"train": math.ceil(config.train_examplar_size / config.batch_size) * config.batch_size,
                  "valid": math.ceil(config.valid_examplar_size / config.batch_size) * config.batch_size})
        #print("我们计算出来的m", m)
        print(f"construct {m['train']} train example and {m['valid']} valid example for {taskname}")
        # 2:1
        m_for_loss = {k: math.ceil(v * config.examplar_method_percentage) for k, v in m.items()}
        m_for_random = {k: v - m_for_loss[k] for k, v in m.items()}

        examplars = dict()
        selected_ids = dict()

        for dtype in ["train", "valid"]:
            print(f"getting examplars for {taskname}")
            # 先用loss来筛一遍
            dtype_selected_ids_by_loss = construct_exemplar_indices_loss(feature_map[dtype], m_for_loss[dtype])
            dtype_selected_ids_by_random = construct_examplar_indices_random(m_for_random[dtype], len(dataset[dtype]), dtype_selected_ids_by_loss)

            del feature_map[dtype]
            dtype_selected_ids = []
            dtype_selected_ids.extend(dtype_selected_ids_by_loss)
            dtype_selected_ids.extend(dtype_selected_ids_by_random)

            # get examplar data
            dtype_exemplar_data = dataset[dtype].get_data_by_indexes(dtype_selected_ids)
            examplars[dtype] = Examplar(dtype_exemplar_data, config.device)
            selected_ids[dtype] = dtype_selected_ids

        if config.return_selected_ids:
            return examplars, selected_ids
        else:
            return examplars





