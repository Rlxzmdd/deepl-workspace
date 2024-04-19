import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from layers.decoder import *

class Batch:
    """
    用于在训练过程中持有一批数据和掩码的对象。
    """
    def __init__(self, src, trg=None, pad=0):
        self.src = src  # 源数据
        self.src_mask = (src != pad).unsqueeze(-2)  # 源数据掩码，如果源数据不等于pad，则掩码位置为True
        if trg is not None:  # 如果目标数据存在
            self.trg = trg[:, :-1]  # 目标数据，除了最后一个单词
            self.trg_y = trg[:, 1:]  # 目标数据，除了第一个单词
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)  # 创建目标数据掩码
            self.ntokens = (self.trg_y != pad).data.sum()  # 计算非pad的目标数据单词数量
    
    @staticmethod
    def make_std_mask(tgt, pad):
        """
        创建一个掩码以隐藏填充和未来的单词。
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)  # 如果目标数据不等于pad，则掩码位置为True
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))  # 创建一个掩码，用于隐藏未来的单词
        return tgt_mask  # 返回掩码
    
def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()  # 记录开始时间
    total_tokens = 1e-9  # 初始化总token数为0
    total_loss = 1e-9  # 初始化总损失为0
    tokens = 0  # 初始化token数为0
    for i, batch in enumerate(data_iter):  # 遍历数据迭代器
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)  # 对每个批次的数据进行前向传播
        loss = loss_compute(out, batch.trg_y, batch.ntokens)  # 计算损失
        total_loss += loss  # 累加总损失
        total_tokens += batch.ntokens  # 累加总token数
        tokens += batch.ntokens  # 累加token数
        if i % 50 == 1:  # 每50个批次打印一次日志
            elapsed = time.time() - start  # 计算经过的时间
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))  # 打印当前步骤，损失和每秒token数
            start = time.time()  # 重置开始时间
            tokens = 0  # 重置token数
    return total_loss / total_tokens  # 返回平均损失

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))