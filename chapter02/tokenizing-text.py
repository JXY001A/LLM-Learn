import re
with open("./the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
# print("Total number of character:", len(raw_text))
print(raw_text[:100])
print(len(raw_text))
text_part = raw_text[:40]
# 使用空格分割
result = re.split(r'\s',text_part)
print(result)