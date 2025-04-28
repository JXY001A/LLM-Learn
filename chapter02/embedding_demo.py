import torch 
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)


print(embedding_layer.weight)

# tensor([[ 0.3374, -0.1778, -0.1690],
#         [ 0.9178,  1.5810,  1.3010],
#         [ 1.2753, -0.2010, -0.1606],
#         [-0.4015,  0.9666, -1.1481],
#         [-1.1589,  0.3255, -0.6315],
#         [-2.8400, -0.7849, -1.4096]]

input_ids = torch.tensor([0,1,5])
print(embedding_layer(input_ids))

# TODO: 所谓嵌入实际上就是首先生成指定数量和维度数的随机权重，然后再将输入的 ids 映射出来
# example：0 映射第1行，1 映射第2行 ，5 映射第 6 行 
# tensor([[ 0.3374, -0.1778, -0.1690],
#         [ 0.9178,  1.5810,  1.3010],
#         [-2.8400, -0.7849, -1.4096]], grad_fn=<EmbeddingBackward0>)