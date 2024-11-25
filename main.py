import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import torch
from torch import nn as nn
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tools import lstm as lstm
from tools import dataprocess as dataprocess
import sys

# 读取 CSV 文件
data = pd.read_csv(r"D:\PycharmProjects\intern\Data\individualMovebankData\2296102400-Branta leucopsis-RRK.csv", header=None, names=["Timestamp", "Latitude", "Longitude"])
# 将时间戳解析为日期时间对象
data["Timestamp"] = pd.to_datetime(data["Timestamp"])
#将timestamp转化为time然后方便模型预测
data["Timestamp"] = data["Timestamp"].apply(lambda x: datetime.timestamp(datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')))

x = np.array(data["Timestamp"].values)
y_lat = np.array(data["Latitude"].values)
y_long = np.array(data["Longitude"].values)

# 调整成前5个顺序，只调整x因为是根据后面多少个x预测y, 并筛选出合适范围的y
sequence_length = 5
x = dataprocess.prepare_dataframe_for_lstm(x, sequence_length)
y_lat = y_lat[sequence_length:].reshape(-1,1)

#将x, y归一化
scaler = MinMaxScaler(feature_range=(-1,1))
x = scaler.fit_transform(x)
scaler2 = MinMaxScaler(feature_range=(-1,1))
y_lat = scaler.fit_transform(y_lat)

# 分开测试和训练集
train_size = int(len(x) * 0.9)
test_size = len(x) - train_size
y_train_lat, y_test_lat = y_lat[:train_size], y_lat[train_size:]
x_train_time, x_test_time = x[:train_size], x[train_size:]

# 变成LSTM符合的维度
x_train_time = x_train_time.reshape(-1, 5, 1)
x_test_time = x_test_time.reshape(-1, 5, 1)
# 调整符合LSTM的float参数
y_train_lat = torch.tensor(y_train_lat, dtype=torch.float32)
y_test_lat = torch.tensor(y_test_lat, dtype=torch.float32)
x_train_time = torch.tensor(x_train_time, dtype=torch.float32)
x_test_time = torch.tensor(x_test_time, dtype=torch.float32)

# 定义hyperparameters
input_size = 1
hidden_size = 128
num_layers = 2
learning_rate = 0.001
batch_size = 32 # 一次训练的样本数量
num_epochs = 100

# 将数据变化成可迭代的tensor
train_dataset = dataprocess.TrajectoryDataset(x_train_time, y_train_lat)
test_dataset = dataprocess.TrajectoryDataset(x_test_time, y_test_lat)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 模型、损失函数和优化器
model = lstm.LSTMModel(input_size, hidden_size, num_layers)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 开始训练模型
for epoch in range(1,num_epochs+1,1):
    model.train(True)
    train_loss  = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = model(x_batch) #调用forward方法进行传播预测
        loss = loss_function(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch [{epoch}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}")

model.eval()  # 设置为评估模式
test_loss = 0.0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # 前向传播
        outputs = model(x_batch)
        loss = loss_function(outputs, y_batch)
        test_loss += loss.item()
print(f"Test Loss: {test_loss / len(test_loader):.4f}")


# 可视化训练结果
predictions = []
true_values = []

model.eval()
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        outputs = model(x_batch)
        predictions.extend(outputs.cpu().numpy())
        true_values.extend(y_batch.numpy())

# 转换为 NumPy 数组
predictions = np.array(predictions).flatten()
true_values = np.array(true_values).flatten()

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(true_values, label="True Values")
plt.plot(predictions, label="Predictions")
plt.legend()
plt.title("LSTM Predictions vs True Values")
plt.show()