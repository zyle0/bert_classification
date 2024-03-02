import torch
import numpy as np
from torch import nn
from transformers import BertModel
from transformers import BertTokenizer
import pandas as pd
from torch.optim import AdamW
from tqdm import tqdm


# 加载预训练的BERT模型和分词器
model_name = r'.cache\huggingface\hub\models--bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert = BertModel.from_pretrained(model_name)

# 自定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [label for label in df['category']]
        self.texts = [tokenizer(text, 
                                padding='max_length', 
                                max_length = 512, 
                                truncation=True,
                                return_tensors="pt") 
                      for text in df['text']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # text
        batch_texts = self.texts[idx]
        # label
        batch_y = np.zeros(6)
        idex = int(self.labels[idx])
        batch_y[idex] = 1
        return batch_texts, batch_y

# 自定义模型
class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 6)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        output = self.relu(linear_output)
        return output
    
def train(model, train_dataloader, val_dataloader, learning_rate, epochs):
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
    
    # 开始进入训练循环
    for epoch_num in range(epochs):
  # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0
        count = 0
        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            # 计算损失
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            # 计算精度
            acc = (output.argmax(dim=1) == train_label.argmax(dim=1)).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            count += 1
            if count%100 == 0:
                print(
                    f'''Epochs: {count}| Train Loss: {total_loss_train/count/batch_size: .3f}| Train Accuracy: {total_acc_train/count/batch_size: .3f}''') 

def evaluate(model, val_dataloader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in val_dataloader:
              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)
              output = model(input_id, mask)
              acc = (output.argmax(dim=1) == test_label.argmax(dim=1)).sum().item()
              total_acc_test += acc   
    print(f'Test Accuracy: {total_acc_test / len(val_dataloader)/16: .3f}')

def predict(model, dataloader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    with torch.no_grad():
        result = []
        for test_input, test_label in dataloader:
              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)
              output = model(input_id, mask)
              result.append(output.argmax(dim=1))
    
    return torch.cat(result).cpu()

# 加载和预处理数据
train_data = pd.read_csv('train.txt', sep='\t', names=['text', 'category'])
train_data = train_data.fillna(5)
val_data = pd.read_csv('dev.txt', sep='\t', names=['text', 'category'])
val_data = val_data.fillna(5)
# 通过Dataset类获取训练和验证集
train_Dataset, val_Dataset = Dataset(train_data), Dataset(val_data)
# DataLoader根据batch_size获取数据，训练时选择打乱样本
batch_size = 16
train_dataloader = torch.utils.data.DataLoader(train_Dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_Dataset, batch_size=batch_size)

# 训练模型
model = BertClassifier()
EPOCHS = 5
LR = 5e-6
train(model, train_dataloader, val_dataloader, LR, EPOCHS)

# 评估模型
evaluate(model, val_dataloader)

# 保存模型
PATH = 'bert_5.pt'
torch.save(model.state_dict(), PATH)
# 读取模型
model.load_state_dict(torch.load(PATH))
# 预测
test_data = pd.read_csv('test.txt', names=['text'])
test_data['category'] = 0
test_Dataset = Dataset(test_data)
test_dataloader = torch.utils.data.DataLoader(test_Dataset, batch_size=100)

res = predict(model, test_dataloader)
test_data['category'] = res
test_data.to_csv('pred.txt',sep='\t', header=False)
