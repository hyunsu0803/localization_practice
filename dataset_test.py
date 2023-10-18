import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class PhaseMapDataset(Dataset):
    def __init__(self, path):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
        
        
def main():
    signal_path = "./signal_samples"
    train_set = PhaseMapDataset(signal_path)
    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True)
    
    for b_x, b_y in train_dataloader:
        print(b_x.shape)
        print(b_y.shape)
        break
    

if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(a[:-5])
    # main()