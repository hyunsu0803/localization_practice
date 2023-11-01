import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os

class PhaseMapDataset(Dataset):
    def __init__(self, path):
        self.traindata = os.listdir(path)[:-32*4]
        self.length = len(self.traindata)
        self.path = path

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.traindata[idx]
        with open(os.path.join(self.path, data), 'rb') as f:
            (phmp, target) = pickle.load(f)
        return phmp, target
    
    
class ValidationDataset(Dataset):
    def __init__(self, path):
        self.traindata = os.listdir(path)[-32*4:-32]
        self.length = len(self.traindata)
        self.path = path

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.traindata[idx]
        with open(os.path.join(self.path, data), 'rb') as f:
            (phmp, target) = pickle.load(f)
        return phmp, target
    
class TestDataset(Dataset):
    def __init__(self, path):
        self.traindata = os.listdir(path)[-16:]
        self.length = len(self.traindata)
        self.path = path

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.traindata[idx]
        with open(os.path.join(self.path, data), 'rb') as f:
            (phmp, target) = pickle.load(f)
        return phmp, target
        
        
def main():
    train_path = "./phasemap_samples"
    train_set = PhaseMapDataset(train_path)
    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True)
    
    for b_x, b_y in train_dataloader:
        print(b_x.shape)
        print(b_y.shape)
        break
    

if __name__ == '__main__':
    main()