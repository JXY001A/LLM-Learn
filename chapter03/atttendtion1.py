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

print('*'*40)
print('atten_weight_2',atten_weight_2)
print('atten_weight_2_sum',atten_weight_2.sum())

print('*'*40)
#TODO: 计算上下文向量 Z : 是所有输入向量的加权总和
# 1. 计算目标向量与所有输入向量的点积
# 2. 对注意力权重做归一化 softmax 操作
# 3. 计算上下文向量：atten_weight_2[i] * x_i 

# 创建一个与 query 结构相同的 context_vec_2 张量
context_vec_2  = torch.zeros(query.shape);
for i,x_i in enumerate(inputs):
    context_vec_2 += atten_weight_2[i] * x_i

print(context_vec_2)