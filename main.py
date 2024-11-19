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



# 调整数据使得其符合MinMaxScaler的维数
x = np.array(data["Timestamp"].values).reshape(-1, 1)
y_lat = np.array(data["Latitude"].values).reshape(-1, 1)
y_long = np.array(data["Longitude"].values).reshape(-1, 1)


#将x, y归一化
scaler = MinMaxScaler(feature_range=(-1,1))
x = scaler.fit_transform(x)
scaler2 = MinMaxScaler(feature_range=(-1,1))
y_lat = scaler.fit_transform(y_lat)

# 分开测试和训练集
train_size = int(len(x) * 0.8)
test_size = len(x) - train_size
y_train_lat, y_test_lat = y_lat[:train_size], y_lat[train_size:]
x_train_time, x_test_time = x[:train_size], x[train_size:]

# 变成LSTM符合的维度
x_train_time = x_train_time.reshape(-1, 1, 1)
x_test_time = x_test_time.reshape(-1, 1, 1)
# 调整符合LSTM的float参数
y_train_lat = torch.tensor(y_train_lat).float()
y_test_lat = torch.tensor(y_test_lat).float()
x_train_time = torch.tensor(x_train_time).float()
x_test_time = torch.tensor(x_test_time).float()

# plt.plot(x_train_time, y_train_lat)
# plt.show()

# 定义hyperparameters
input_size = 1
hidden_size = 50
num_layers = 1
learning_rate = 0.001
batch_size = 100 # 一次训练的样本数量
num_epochs = ceil(train_size/batch_size)

# 将数据变化成可迭代的tensor
train_dataset = dataprocess.TrajectoryDataset(x_train_time, y_train_lat)
test_dataset = dataprocess.TrajectoryDataset(x_test_time, y_test_lat)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 模型、损失函数和优化器
model = lstm.LSTMModel(input_size, hidden_size, num_layers)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 开始训练模型
for epoch in range(1,num_epochs+1,1):
    model.train(True)
    print(f'Epoch: {epoch}')
    running_los = 0.0
    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0], batch[1]
        output = model(x_batch) #调用forward方法进行传播预测
        loss = loss_function(output, y_batch)
        running_los+=loss.item()
        optimizer.zero_grad()  # 清除梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新模型参数
        if batch_index % 100 == 99:  # 每 100 个 batch 打印一次
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index, avg_loss_across_batches))
            running_loss = 0.0
    running_los = 0.0
    model.train(False)
    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0], batch[1]
        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_los += loss.item()
    avg_loss_across_batches = running_los / len(test_loader)
    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print("************************************************")