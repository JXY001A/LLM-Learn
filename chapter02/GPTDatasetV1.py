import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]
    


def create_dataloader_v1(txt,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True,num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt,tokenizer,max_length,stride)

    dataLoader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataLoader

with open("/Users/jinxianyu/code/LLM-Learn/chapter02/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# stride : 跨度
# batch_size： 批量
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)

data_iter = iter(dataloader)
inputs,target = next(data_iter)
# print('input:\n',inputs)
# print('\n target :\n',target)
# x = torch.tensor([[1, 2], [3, 4]]);
# print(x)

# inputs1,target2 = next(data_iter)
# print('inputs1:\n',inputs1)
# print('\n target2 :\n',target2)

vocab_size = 50257
output_dim = 256

# 词汇量 和 嵌入纬度数 
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

token_embedding = token_embedding_layer(inputs);
print(token_embedding.shape)

