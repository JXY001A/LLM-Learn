# 在因果注意力中，获得掩码后的注意力权重矩阵的一种方法是对注意力分数应用 softmax
# 函数，将对角线以上的元素清零，并对所得矩阵进行归一化
import torch
from SelfAttention_v2 import SelfAttention_v2

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],
    ]  # step     (x^6)
)

d_in = inputs.shape[1] # 3
# 输出嵌入向量数
d_out = 2

torch.manual_seed(789)

sa_v2 = SelfAttention_v2(d_in,d_out)
queries = sa_v2.W_key(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.05,dim=-1)


# PyTorch 的 tril 函数可是实现对角线以上元素为 0 的掩码
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length,context_length))

# 相乘，就是相同位置的元素相乘，实现对角线以上元素为 0 的作用，从而实现掩码能力
attn_weight_mask = attn_weights * mask_simple; 

print('attn_weight_mask',attn_weight_mask)
