import re
with open("./the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
# print("Total number of character:", len(raw_text))
print(raw_text[:100])
print(len(raw_text))
text_part = raw_text[:40]
# 使用空格 | [,.] 分割
result = re.split(r'([,.:;?_()\']|--|\s)',text_part)
result = [item for item in result if item.strip()]
print(result)

preprocessed = re.split(r'([,.:;?_()\']|--|\s)',raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("preprocessed:",preprocessed[:30])
print("preprocessed len",len(preprocessed))

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = { token:integer for integer,token in enumerate(all_words)}
for i,item in enumerate(vocab.items()):
    print(item)
    if i>50:
        break