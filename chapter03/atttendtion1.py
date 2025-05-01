from importlib.metadata import version
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
atten_weights_2_temp = atten_scores_2 / atten_scores_2.sum()
print('attention weights:', atten_weights_2_temp)
print('sum:',atten_weights_2_temp.sum())

