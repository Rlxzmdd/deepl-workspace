# !pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl numpy matplotlib spacy torchtext seaborn 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from layers.encoder import *
from layers.decoder import *
from layers.attention import *
from layers.layer import *
from embedding import *
from positional import *
from utils.dialogue_dataset import *
from utils.train import *
from opter.noamopt import *
from utils.tokenizer import *
from torch.utils.data import DataLoader

seaborn.set_context(context="talk")

def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

def data_gen(train_loader):
    "从train_loader获取数据"
    for batch_idx, batch in tqdm(
            enumerate(train_loader),
            total=int(len(train_loader.dataset) / 2) + 1,
        ):
        src_batch, tgt_batch = (
                batch["src_ids"],
                batch["tgt_ids"]
            )

        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        yield Batch(src_batch, tgt_batch, 0)  # 产生一个批次的数据

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm
    
# 定义贪婪解码函数，用于生成预测序列
def greedy_decode(model, tokenizer,src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)  # 编码输入序列
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)  # 初始化输出序列
    o_res = []
    for i in range(max_len-1):  # 对于每一个位置
        # 解码生成预测
        out = model.decode(memory, src_mask, 
                Variable(ys), 
                Variable(subsequent_mask(ys.size(1))
                        .type_as(src.data)))
        prob = model.generator(out[:, -1])  # 获取最后一个位置的输出概率
        _, next_word = torch.max(prob, dim = 1)  # 获取概率最大的词
        next_word = next_word.data[0]  # 获取词的索引
        o_res.append(next_word.item())
        # 将预测的词添加到输出序列中
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_((next_word))], dim=1)
        if next_word.item() == "[end]": break
    return tokenizer.detokenize(o_res)  # 返回输出序列




# 分词、样本加载
tokenizer = basic_tokenizer("/home/iszhous/projects/deepl-workspace/transformer/data/self_vocab.txt")
trainset = dialogue_dataset("/home/iszhous/projects/deepl-workspace/transformer/data/data/self_train.txt", tokenizer)
print("训练集样本数：%d" % (trainset.__len__()))

V = tokenizer.vocab_size

# 创建标签平滑对象，用于计算损失
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
# 创建模型实例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(
    trainset,
    batch_size=2,
    num_workers=4,
    shuffle=True,
    collate_fn=collate_func,
    drop_last=True,
)
model = make_model(V, V, N=2)
model.to(device=device)
# 创建优化器实例
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def chat(query):
    src_ids = tokenizer.tokenize(list(query))
    src = torch.tensor(src_ids).unsqueeze(0)
    # src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )     # 创建输入序列
    src_len = src.shape[-1]  # 获取输入序列的长度
    src_mask = Variable(torch.ones(1, 1, src_len))
    src = src.to(device)
    src_mask = src_mask.to(device)
    # 打印贪婪解码的结果
    print(greedy_decode(model,tokenizer, src, src_mask, max_len=10, start_symbol=1))

for epoch in range(10):
    print("epoch->"+str(epoch))
    model.train() 
    run_epoch(data_gen(train_loader), model, SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()  # 设置模型为评估模式
    # 打印评估结果
    print("lose->"+ str(run_epoch(data_gen(train_loader), model, SimpleLossCompute(model.generator, criterion, None))))
    # torch.save(model.state_dict(), str(epoch)+".pth")
    
    chat("你好")
    chat("我是谁？")
    chat("你是谁？")
    chat("我是")
    chat("你是")
    chat("鲁迅是谁？")

