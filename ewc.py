"""
Implementation of Elastic weight consolidation object
"""

import torch
from torch import nn
from torch.autograd import Variable
import copy
from model import UnifiedQG
from loader.dataset import Examplar


# 要求导只需要做到两点：
#
# 变量tensor是float或者其他复杂类型；
# 将requires_grad指定为True；
# 设置Variable，因为只有Variable是可以变的，而tensor则是不可以变的。
# 求导的步骤：
# 对需要求导的变量设置requires_grad设置以及Variable设置；
# 对结果公式进行反向求导，那么那些requires_grad为True的就都会被求导
# 导出结果，看你需要那个变量的求导结果，那么就可以直接用变量.grad
def variable(t: torch.Tensor, device, **kwargs):

    t = t.to(device)
    return Variable(t, **kwargs)

class EWC(object):
    # 传入的参数（这个模型因为还没有进行训练，所以也是针对过去数据的），模型、之前数据组成的dataset
    def __init__(self, model: UnifiedQG, dataset:Examplar, config):

        self.config = config
        self.model = model
        # the data we use to compute fisher information of ewc (old_exemplars)
        self.dataset = dataset
        self.dataset.shuffle()

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {} # previous parameters
        self._precision_matrices = self._diag_fisher() # approximated diagnal fisher information matrix

        for n, p in copy.deepcopy(self.params).items():
            self._means[n] = variable(p.data, config.device)

    def _diag_fisher(self):

        self.model.train()
        precision_matrices = {}  # 字典，初始化为参数的data(以variable包裹)
        for n, p in copy.deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data, self.config.device)

        # self.dataset.batch_size = 1  # set batch_size to 1 in ewc 我们这只有一个一个的取

        for i in range(self.dataset.size()):
            self.model.zero_grad()
            source_ids_tensor, source_mask_tensor, target_ids_tensor, target_mask_tensor, _, _ = self.dataset[i]
            # print("size大小：", target_ids_tensor.size())
            source_ids_tensor = source_ids_tensor.unsqueeze(dim=0)
            source_mask_tensor = source_mask_tensor.unsqueeze(dim=0)
            target_ids_tensor = target_ids_tensor.unsqueeze(dim=0)
            target_mask_tensor = target_mask_tensor.unsqueeze(dim=0)
            loss = self.model.get_loss(source_ids_tensor, source_mask_tensor, target_ids_tensor, target_mask_tensor)
            # feedforward and calculate loss
            # if self.model.model_type == "lm":
            #     decoded_words, _ = self.model(input_var, self.dataset, feats_var)
            # else:
            #     self.model.set_prior(False)
            #     target_var = input_var.clone()
            #     decoded_words, _ = self.model(input_var, input_lengths = lengths, target_seq = target_var, target_lengths = lengths, conds_seq = feats_var, dataset = self.dataset)
            #
            # length = Variable(torch.LongTensor(lengths)).cuda()
            #
            # # empirical Fisher if we provide ground truth label
            # loss = masked_cross_entropy(
            #     self.model.output_prob.contiguous(),  # -> batch x seq
            #     label_var.contiguous(),  # -> batch x seq
            #     length)

            loss.backward()  # 并没有进行optimization(针对该条数据计算梯度，但并不进行优化，对于每一条数据，计算完梯度之后都要进行更新)

            for n, p in self.model.named_parameters():

                # Jump over layers that is not trained
                if p.grad is None:
                    continue
                precision_matrices[n].data += p.grad.data ** 2 / self.dataset.size()

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()

        return loss
