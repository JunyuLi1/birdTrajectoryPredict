import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# 读取数据
data = pd.read_csv("your_file.csv", header=None, names=["Timestamp", "Latitude", "Longitude"])
data["Timestamp"] = pd.to_datetime(data["Timestamp"])

# 提取时间特征（可以使用小时、分钟、秒，或者将时间转换为秒）
data["Seconds"] = data["Timestamp"].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

# 提取特征和目标
X = data["Seconds"].values.reshape(-1, 1)
y = data[["Latitude", "Longitude"]].values

# 归一化数据
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# 转换为 PyTorch 张量
X = torch.FloatTensor(X).unsqueeze(1)  # 增加一个维度，符合 LSTM 输入要求
y = torch.FloatTensor(y)

# 数据集分割
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_size)  # 初始隐藏状态
        c_0 = torch.zeros(1, x.size(0), self.hidden_size)  # 初始记忆单元状态

        out, _ = self.lstm(x, (h_0, c_0))  # LSTM 前向传播
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 超参数
input_size = 1
hidden_size = 50
output_size = 2
num_layers = 1
learning_rate = 0.001
num_epochs = 100

# 模型、损失函数和优化器
model = LSTMModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predictions = scaler_y.inverse_transform(predictions.numpy())  # 反归一化

# 输出预测结果
print("Predicted Latitudes and Longitudes:", predictions)
