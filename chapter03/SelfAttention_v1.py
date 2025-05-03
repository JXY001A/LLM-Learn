import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self,d_in,d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in,d_out))
        self.W_key =  nn.Parameter(torch.rand(d_in,d_out))
        self.W_value = nn.Parameter(torch.rand(d_in,d_out))

    def forward(self,x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        # 注意力分数
        attn_scores = queries @ keys.T;
        # 注意力权重
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)
        # 上下文向量
        # 自注意力机制的目标是为每个输入元素计算一个上下文向量，该向量结合了其他所有输入元素的信息。
        # 上下文向量（context vector）可以被理解为一种包含了序列中所有元素信息的嵌入向量
        context_vec = attn_weights @ values
        return context_vec





