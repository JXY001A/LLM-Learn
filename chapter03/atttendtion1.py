from importlib.metadata import version
import math
import torch

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


query = inputs[0];
# print(inputs);
# print("inputs.shape",inputs.shape)

# 创建一个结构与 inputs.shape[0] 相同的张量 tensor
atten_scores_2 = torch.empty(inputs.shape[0])

for i,x_i in enumerate(inputs):
    # 通过点积计算两个向量的方向对齐程度：点积越大，对齐程度越高或者叫越相似（也就是角度越解决）
    atten_scores_2[i] = torch.dot(query,x_i) 
# print("atten_scores_2",atten_scores_2) 
# atten_scores_2  tensor([0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310])
# 第一个值最大：因为 query 就是 inputs[0]

# TODO: 简单归一化实现
# atten_weights_2_temp = atten_scores_2 / atten_scores_2.sum()
# print('attention weights:', atten_weights_2_temp)
# print('sum:',atten_weights_2_temp.sum())

# 相当于： [[e^1, e^2],[e^3, e^4]]，其中 e 为自然对数
# print(torch.exp(query)) # [(math.e ** 0.43),4),(math.e ** 0.15),4),(math.e ** 0.89),4)]

# print("inputs:",inputs);
# print('torch.exp(inputs)',torch.exp(inputs))
# print(torch.exp(inputs).sum(dim=1))

#TODO: 正式归一化实现
# softmax 函数可以保证注意力权重总是正值，这使得输出可以被解释为概率或相对重要性，其中权重越高表示重要程度越高
def softmax_naive(x):
    return torch.exp(x)/torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(atten_scores_2)
print("Attention weights:", attn_weights_2_naive) # tensor([0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452])
print("Sum:", attn_weights_2_naive.sum())

# 一般推荐直接使用 torch 官方提供的 softmax 
atten_weight_2 = torch.softmax(atten_scores_2,dim=0);

print('='*60)
print('atten_weight_2',atten_weight_2)
print('atten_weight_2_sum',atten_weight_2.sum())

print('='*60)
#TODO: 计算上下文向量 Z : 是所有输入向量的加权总和
# 1. 注意力分数：计算目标向量与所有输入向量的点积
# 2. 注意力权重：注意力分数做归一化 softmax 操作
# 3. 计算上下文向量：输入的加权和 (atten_weight_2[i] * x_i)++

# 创建一个与 query 结构相同的 context_vec_2 张量
context_vec_2  = torch.zeros(query.shape);
for i,x_i in enumerate(inputs):
    context_vec_2 += atten_weight_2[i] * x_i

print(context_vec_2)


# TODO: 计算所有输入次元注意力权重
print('='*60)
atten_scores = torch.empty(6,6);

for i,x_i in enumerate(inputs):
    for j,x_j in enumerate(inputs):
        atten_scores[i,j] = torch.dot(x_i,x_j)

print("atten_scores",atten_scores);

# TODO: 矩阵乘法计算注意力
print('='*60)
# demo_tensor = torch.tensor([[1,2,3],[4,5,6]])
# 将矩阵转置例如：tensor([[1,2,3],[4,5,6]]) 转换为 tensor([[1,4],[2,5],[3,6]])
# print(demo_tensor.T)

attn_scores = inputs @ inputs.T;

print("attn_scores",attn_scores);

# a = [0.4300, 0.1500, 0.8900]
# b = [0.4300,0.150,0.8900 ]

# score = 0
# for i in range(0,len(a)):
#     print(i)
#     score+=a[i]*b[i]
# print(score)   

# 归一化
print('='*60)
# 对张量的最内层数据（通常是特征维度）进行 Softmax 转换，使其成为概率分布
attn_weights = torch.softmax(attn_scores,dim=-1);

print('attn_weights',attn_weights);

# 计算上下文向量
# 自注意力机制的目标是为每个输入元素计算一个上下文向量，该向量结合了其他所有输入元素的信息。
# 上下文向量（context vector）可以被理解为一种包含了序列中所有元素信息的嵌入向量
# attn_weights(5*6), inputs(6*3)
all_context_vecs = attn_weights @ inputs
# print('all_context_vecs',all_context_vecs)

#TODO: 实现带可训练权重的自注意力机制
# 查询向量（q）、键向量（k）、值向量（v）
print('='*60)

x_2 = inputs[1];
# 输入嵌入向量维度数
d_in = inputs.shape[1] # 3
# 输出嵌入向量数
d_out = 2

# 初始化权重矩阵 Wq,Wk,Wv
# 设置 requires_grad=False 以减少输出中的其他项，
# 但如果要在模型训练中使用这些权重矩阵，就需要设置 requires_grad=True，以便在训练中更新这些矩阵
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
# 在权重矩阵 W 中，“权重”是“权重参数”的简称，表示在训练过程中优化的神经网络参
# 数。这与注意力权重是不同的。正如我们已经看到的，注意力权重决定了上下文向量对输入
# 的不同部分的依赖程度（网络对输入的不同部分的关注程度）。
# 总之，权重参数是定义网络连接的基本学习系数，而注意力权重是动态且特定于上下文
# 的值。
query_2 = x_2 @ W_query
# key_2 = x_2 @ W_key
# value_2 = x_2 @ W_value
print('query_2',query_2)


# TODO: 输入向量 x 与权重矩阵 W 相乘（Wx+b）表示特征的线性组合。
keys = inputs @ W_key
values = inputs @ W_value

key_2 = keys[1]
atten_scores_22 = query_2.dot(key_2)
print(atten_scores_22)










