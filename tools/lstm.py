import torch
from torch import nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) #完全连接神经网络模型，最后产出一个output

    def forward(self, x):
        bacth_size = x.size(0)
        h_0 = torch.zeros(self.num_stacked_layers, bacth_size, self.hidden_size)  # 初始隐藏状态
        c_0 = torch.zeros(self.num_stacked_layers, bacth_size, self.hidden_size) # 初始记忆单元状态
        out, _ = self.lstm(x, (h_0, c_0))  # LSTM 前向传播
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)