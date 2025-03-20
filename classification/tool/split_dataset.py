import json
import pandas as pd

# 假设jsonl内容存储在一个文件中
jsonl_file = '/home/kara/classification/dataset/prompts.jsonl'

# 读取并解析jsonl文件
def extract_texts(jsonl_file):
    df = pd.read_json(jsonl_file, lines=True)

    # 提取prompt和continuation的text
    df['prompt_text'] = df['prompt'].apply(lambda x: x['text'])
    df['continuation_text'] = df['continuation'].apply(lambda x: x['text'])
    # 合并prompt_text和continuation_text，中间用空格分隔
    df['combined_text'] = df['prompt_text'] + df['continuation_text']

    # prompts = df['prompt'].apply(lambda x: x['text'])
    # continuations = df['continuation'].apply(lambda x: x['text'])
    toxicity = df['continuation'].apply(lambda x: x['toxicity'])
    combined = df['combined_text']
    return combined, toxicity


# 调用函数
text = []
toxicity = []
text, toxicity = extract_texts(jsonl_file)
count1 = (toxicity > 0.5).sum()

label = toxicity.apply(lambda x: 1 if x > 0.5 else 0)
count2 = (label == 1).sum()

df = pd.DataFrame({'txt': text, 'label': label})

print(count1)
print(count2)

# 将DataFrame保存为CSV文件
# df.to_csv('/home/kara/classification/dataset/dataset.csv', index=False)

# 或者保存为JSON文件
# df.to_json('output.json', orient='records', lines=True)