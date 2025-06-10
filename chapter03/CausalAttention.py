from torch import nn
import torch
class CausalAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,qkv_bias=False):
        super().__init__()
        self.d_out = d_out;
        self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)
        # 掩码比例，（0～1)
        self.dropout = nn.Dropout(dropout)
        # 虽然在 PyTorch 中使用 register_buffer 并非所有情况下都是必
        # 需的，但在这里具有一些优势。例如，当我们在大语言模型中使用 CausalAttention 类时，缓
        # 冲区会与模型一起自动移动到适当的设备（CPU 或 GPU），这在训练大语言模型时非常重要。这
        # 意味着我们无须手动确保这些张量与模型参数在同一设备上，从而避免了设备不匹配的错误
        # torch.triu 将矩阵对角线以上全部置为 1 的掩码矩阵
        self.register_buffer('mask',torch.triu(torch.ones(context_length, context_length),diagonal=1)) 

    def forward(self,x):
        b,num_tokens,d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens,:num_tokens],-torch.inf)

        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=1)

        attn_weights = self.dropout(attn_weights)

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
d_in = inputs.shape[1]
d_out = 2 
batch = torch.stack((inputs, inputs), dim=0)
# print(batch.shape) 
# print(batch)
torch.manual_seed(123)
context_length = batch.shape[1]

ca = CausalAttention(d_in, d_out, context_length, 0.0) 
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)