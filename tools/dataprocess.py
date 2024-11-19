from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
