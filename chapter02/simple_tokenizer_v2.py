import re
class SimpleTokenizerV2 :
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self,text):
        preprocessed = re.split(r'([,.:;?_!()\']|--|\s)',text)
        preprocessed = [
            item.strip()  for item in preprocessed if item.strip()
        ]
        preprocessed = [ item  if item in self.str_to_int else "<|unk|>"  for item in preprocessed ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids;
    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!()\'])',r'\1',text)
        return text


with open("/Users/jinxianyu/code/LLM-Learn/chapter02/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_()\']|--|\s)',raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_words = sorted(set(preprocessed))
vocab = { token:integer for integer,token in enumerate(all_words)}

tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text)
# print(vocab)