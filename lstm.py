import torch
from torch import nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_size)  # 初始隐藏状态
        c_0 = torch.zeros(1, x.size(0), self.hidden_size)  # 初始记忆单元状态

        out, _ = self.lstm(x, (h_0, c_0))  # LSTM 前向传播
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

    def predict(self, timestamp):
        pass
        # 归一化后反归一化预测

if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)