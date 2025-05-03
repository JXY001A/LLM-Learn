import torch
import torch.nn as nn

# nn.Module 是 PyTorch 模型的一个基本构建块，它为模型层的创建和管理提供了必要的功能
# 重点：调用模型时（如 model(input)），会自动调用 forward。
class SelfAttention_v1(nn.Module):
    # __init__方法初始化了可训练的权重矩阵（W_query、W_key 和 W_value），这些矩阵用
    # 于查询向量、键向量和值向量，每个矩阵将输入维度 d_in 转换为输出维度 d_out
    def __init__(self,d_in,d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in,d_out))
        self.W_key =  nn.Parameter(torch.rand(d_in,d_out))
        self.W_value = nn.Parameter(torch.rand(d_in,d_out))
    # forward 方法将查询向量和键向量相乘来计算注意力分
    # 数（attn_scores），然后使用 softmax 对这些分数进行归一化。最后，我们通过使用这些归一
    # 化的注意力分数对值向量进行加权来创建上下文向量
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

# 输入嵌入向量维度数
d_in = inputs.shape[1] # 3
# 输出嵌入向量数
d_out = 2

torch.manual_seed(123)

sa_v1 = SelfAttention_v1(d_in,d_out)

print(sa_v1(inputs));

