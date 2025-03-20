import pandas as pd
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

# 自定义Dataset类
class ToxicityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts.tolist()  # 将 Pandas Series 转换为列表
        self.labels = labels.tolist()  # 将 Pandas Series 转换为列表
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float).unsqueeze(0)  # Add an extra dimension
        }
# 定义tokenization函数
def tokenize_texts(texts):
    return tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors='pt')

class ToxicityClassifier(nn.Module):
    def __init__(self):
        super(ToxicityClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits
    
df = pd.read_csv('/home/kara/classification/dataset/test_data.csv')
df = df.drop_duplicates()
test_texts, test_labels = df['txt'], df['label']
# # 数据分割
# train_texts, test_texts, train_labels, test_labels = train_test_split(df['txt'], df['label'], test_size=0.2, random_state=42)

# 初始化BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
test_dataset = ToxicityDataset(test_texts, test_labels, tokenizer, max_len=128)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# 模型
model = ToxicityClassifier()
model.load_state_dict(torch.load('/home/kara/classification/model/7.21-2layer/model_epoch_16.pth'))

#gpu
device = torch.device('cuda:0')
model.to(device)

model_path = '/home/kara/classification/model/7.21-2layer/result/'

# 测试模型并生成混淆矩阵
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.round(torch.sigmoid(outputs))

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='binary')  # 也可以使用'micro'或'macro'作为average参数

# 生成混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()
plt.savefig(model_path + 'matrix.png')
print(accuracy)
print(f1)
print(cm)

# 读取参数
# model.load_state_dict(torch.load('model_checkpoints/model_epoch_2.pth'))