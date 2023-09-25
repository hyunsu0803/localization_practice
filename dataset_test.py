import torch
from torch.utils.data import Dataset, DataLoader


class SimpleDataset(Dataset):
    def __init__(self, t):
        self.t = t

    def __len__(self):
        return self.t

    def __getitem__(self, idx):
        return torch.LongTensor([idx])
        
if __name__ == "__main__":
    dataset = SimpleDataset(t=5)
    print(len(dataset))
    it = iter(dataset)
    
    print(next(it))

    for i in range(10):
        print(i, next(it))
        
    print(dataset[100])
    
    