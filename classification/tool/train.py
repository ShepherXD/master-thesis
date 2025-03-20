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
        # 冻结
        for param in self.bert.encoder.layer[:10].parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits
    
df = pd.read_csv('/home/kara/classification/dataset/dataset.csv')
df = df.drop_duplicates()
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# 保存训练集
train_df.to_csv('/home/kara/classification/dataset/train_data.csv', index=False)
# 保存测试集
test_df.to_csv('/home/kara/classification/dataset/test_data.csv', index=False)
# 打印数据集长度以验证分割是否正确
print(f"Original dataset size: {len(df)}")
print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")
print(f"Sum of training and test set sizes: {len(train_df) + len(test_df)}")
train_texts, train_labels = train_df['txt'], train_df['label']
test_texts, test_labels = test_df['txt'], test_df['label']
# # 数据分割
# train_texts, test_texts, train_labels, test_labels = train_test_split(df['txt'], df['label'], test_size=0.2, random_state=42)

# 初始化BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# # 创建DataLoader
# train_dataset = TextDataset(train_encodings, train_labels.tolist())
# test_dataset = TextDataset(test_encodings, test_labels.tolist())
train_dataset = ToxicityDataset(train_texts, train_labels, tokenizer, max_len=128)
test_dataset = ToxicityDataset(test_texts, test_labels, tokenizer, max_len=128)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# 模型
model = ToxicityClassifier()
# 优化器
# optimizer = AdamW(model.parameters(), lr=1e-5)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

#gpu
device = torch.device('cuda:1')
model.to(device)

for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f'Frozen layer: {name}')
#loss func
criterion = nn.BCEWithLogitsLoss()

model_path = '/home/kara/classification/model/7.21-2layer/'
model_architecture_file = 'model_architecture.txt'
print(model)
with open(model_path + model_architecture_file, 'w') as f:
    f.write(str(model))
f.close()
print("已经保存模型结构")
num_epochs = 1
train_losses = []


# 训练循环
for epoch in range(20):  # 假设训练3个epoch
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=epoch_loss / len(train_loader))


  
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f'Epoch {epoch + 1}')
    print(f'Training loss: {avg_train_loss}')
    # 每两个epoch保存一次模型
    if (epoch + 1) % 2 == 0:
        model_save_path = os.path.join(model_path, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')



plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')
plt.show()
plt.savefig(model_path + 'training_loss.png')

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