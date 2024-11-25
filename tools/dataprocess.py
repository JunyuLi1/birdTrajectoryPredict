from torch.utils.data import Dataset
from copy import deepcopy as dc
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def prepare_dataframe_for_lstm(data, n_steps):
    lagged_data = []

    # 遍历数据并创建滞后特征
    for i in range(n_steps, len(data)):
        lagged_data.append(data[i - n_steps:i])

    return np.array(lagged_data)