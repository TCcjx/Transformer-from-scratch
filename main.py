import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Hyperparameters 
batch_size = 64 # 批量大小
context_length = 32 # 上下文长度
d_model = 128 # 嵌入维度
num_blocks = 16 # transformer 块的数量
num_heads = 8 # 多头注意力的头数
learning_rate = 1e-3 # 学习率
head_size = d_model // num_heads
dropout_rate = 0.1 # dropout概率
max_iters = 1 # 最大训练批次数
eval_interval = 50 # 评估间隔
eval_iters = 20 # Number of iterations to average for evalution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 设备
TORCH_SEED = 1317 # 随机数种子
torch.manual_seed(TORCH_SEED)

# Load data
# 读取数据集,如果数据集不存在的话，就从huggingface上下载
if not os.path.exists('data/sales_train.csv'): 
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true'
    with open('data/sales_textbook.txt', 'wb') as f:
        f.write(requests.get(url).content)

with open('data/sales_textbook.txt', 'r') as f:
    text = f.read()


# Tokenize the text
# 使用openai开源库 tiktoken分词器
encoding = tiktoken.get_encoding('cl100k_base')
tokenized_text = encoding.encode(text)
max_token_value = max(tokenized_text) # 分词结果的最大数字,也就是说词表中的分词数就是 max_token_value + 1

# split train data and test data
# 按照 9:1 的比例划分训练数据 和 测试数据
train_size = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[ : train_size]
valid_data = tokenized_text[train_size : ]

# Feed Forword
class FeedforwardNetwork(nn.Module): # FeedforwardNetwork Module
    
    def __init__(self, d_model):
        super(FeedforwardNetwork,self).__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.linear2(F.relu(self.linear1(x))))

# 单头注意力机制
class Attention(nn.Module): # 注意力机制的实现

    def __init__(self, head_size: int):
        super(Attention,self).__init__()
        self.Wq = nn.Linear(d_model, head_size)
        self.Wk = nn.Linear(d_model, head_size)
        self.Wv = nn.Linear(d_model, head_size)
        # apply mask
        # tril 下三角掩码矩阵 , triu 上三角掩码矩阵
        self.register_buffer('mask',torch.tril(torch.ones(context_length,context_length).bool()))
        self.dropout = nn.Dropout(dropout_rate)
        self.output_linear = nn.Linear(head_size,head_size)

    def forward(self, x):
        B, T, C = x.shape # Batch_size, Time steps,embedding sieze（嵌入维度大小）
        assert T <= context_length
        assert C == d_model
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        attention_score = Q @ K.transpose(-1,-2) / math.sqrt(d_model // num_heads) # 进行scale化,保证训练的稳定，同时防止梯度爆炸
        attention_score = attention_score.masked_fill(self.mask == 0,float('-inf')) # 下三角掩码实现
        attention_score = F.softmax(attention_score,dim=-1)
        attention_score = self.dropout(attention_score)

        output = attention_score @ V
        output = self.output_linear(output) # 最后再做一个投影变换，这部分是可有可无的

        return output

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.d_model = d_model
        self.context_length = context_length
        
        self.heads = nn.ModuleList([Attention(head_size=head_size) for _ in range(num_heads)]) # num_heads 多头数
        self.projection_layer = nn.Linear(self.d_model, self.d_model)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.projection_layer(out)
        out = self.dropout_layer(out)
        return out


class TransformerBlock(nn.Module):

    def __init__(self, num_heads:int):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.head_size = d_model // num_heads
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout_rate)

        self.multi_head_attention = MultiHeadAttention(head_size=self.head_size)
        self.feed_forward_layer = FeedforwardNetwork(d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.multi_head_attention(self.layer_norm1(x))
        x = x + self.feed_forward_layer(self.layer_norm2(x))
        return x

class TransformerLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout_rate
        self.max_token_value = max_token_value

        # token Embedding
        self.token_embedding_lookup_table = nn.Embedding(self.max_token_value+1, self.d_model)
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock(num_heads=self.num_heads) for _ in range(self.num_blocks)] + 
                [nn.LayerNorm(self.d_model)]
        ))
        self.language_model_out_linear_layer = nn.Linear(d_model, max_token_value+1)

    def forward(self, idx, target = None):
        B, T = idx.shape # batch_size , context_length

        # 位置编码信息
        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model)
        position = torch.arange(0, self.context_length,dtype = torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000) / self.d_model))
        position_encoding_lookup_table[:,0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:,1::2] = torch.cos(position * div_term)

        # change position_encoding_lookup_table from (context_length, d_model) to (T, d_model)
        position_embedding = position_encoding_lookup_table[:T,:].to(device) # 这里想想为什么是取T个序列呢? 其实这里不必要，因为context_lenth = T
        x = self.token_embedding_lookup_table(idx) + position_embedding # 多个transformer块实现
        x = self.transformer_blocks(x)
        logits = self.language_model_out_linear_layer(x)

        if target is not None:
            B,T,C = logits.shape
            logits_reshaped = logits.view(B*T, C)
            targets_reshaped = target.view(B*T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens): # 生成预测 max_new_tokens个字符
            idx_crop = idx[:,-self.context_length:]
            # get prediciton
            logits, loss = self.forward(idx_crop)
            logits_last_timestep = logits[:,-1,:] # 续写的最后一个字
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            idx_next = torch.multinomial(input=probs, num_samples=1)
            idx = torch.cat((idx,idx_next),dim=-1) # 将续写预测到的那个词给拼接到之前生成字符串的末尾
        return idx


# 初始化model,并加载到GPU
model = TransformerLanguageModel()
model = model.to(device)


# 加载批次数据，用于训练和验证模型的性能
def get_batch(split: str): # 'train' or 'val'
    data = train_data if split == 'train' else valid_data
    data = torch.tensor(data)
    idxs = torch.randint(0,len(data) - context_length,(batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x,y


# Calculate loss
@torch.no_grad() # 装饰器,表示以下模块不会计算梯度信息
def estimate_loss():
    out = {}
    model.eval() # 评估模式
    for split in ['train','val']:
        losses = torch.zeros(eval_iters) # eval_iters 表示一共要评估多少批数据，然后再取平均
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch,y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # 恢复成训练模式
    return out


# 统计模型参数量
def cal_model_num(model) -> None:
    num = sum([p.numel() for p in model.parameters()])
    print(f'model total parameters:{num/1e6} M')


# Use AdamW 优化器
def train():
    optimizer = torch.optim.AdamW(params=model.parameters(), lr = learning_rate)
    tracked_losses = list()
    print('开始训练......')
    cal_model_num(model = model)
    for step in range(max_iters): # max_iters训练epoch数
        if step % eval_iters == 0 or step == max_iters -1:
            losses = estimate_loss()
            tracked_losses.append(losses)
            # 打印训练集 loss 和 验证集 loss,保留三位小数
            print('Step:',step,'Training Loss:',round(losses['train'].item(),3),'Validation Loss:',round(losses['val'].item(),3))

        xb , yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none = True) # 梯度清零
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

    print('训练完成!')

# Save the model state dicitonary 保存模型
torch.save(model.state_dict(),'model-ckpt.pt')
def predict():
    model.eval() # 评估模式
    start = "you create a sense of familiarity, comfort, and shared interests, making it easier to communicate and influence their decision-making process. \
    Your level of enthusiasm can be contagious and can greatly impact the customer's perception of you and your offering. By displaying genuine enthusiasm and confidence in what you are selling, you inspire trust and make the customer more inclined to believe in the value of your product or service.\
    Creating a positive first impression is not just about superficial charm or manipulation; it is about genuinely caring for the customer and their needs. It is about establishing a strong foundation of trust and credibility, setting the stage for a successful sales journey. By mastering the art of creating a positive first impression, you lay the groundwork for building lasting relationships and achieving sales success. Chapter 1: Building Rapport and Capturing Attention \
    Subpoint: Active Listening and Demonstrating Empathy"
    # 开始续写
    start_ids = encoding.encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device))
    x = x.unsqueeze(0)
    y = model.generate(x,max_new_tokens = 200)
    print('--------------------')
    print(encoding.decode(y[0].tolist()))
    print('--------------------')

if __name__ == '__main__':
    train()
    predict()
