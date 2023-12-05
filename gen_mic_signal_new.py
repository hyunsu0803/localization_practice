import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
import os
import glob
from gen_mic_signal import MicSignal

class MicSignalDataset(Dataset):
    def __init__(self, speeches, rooms, save_path):
        print("Preparing the data...")
        self.speeches = glob.glob(speeches, recursive=True)
        self.rirs = glob.glob(rooms)
        self.msg = MicSignal()
        self.save_path = save_path
        print("Data is prepared.")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
    
    def __len__(self):
        return len(self.speeches)
    
    def __getitem__(self, index):
        s = self.speeches[index]
        self.msg.conv_n_add(s, self.rirs, self.save_path)
        return 1


def main():
    
    # training data generation
    train_speeches = "/root/mydir/hdd/librispeech/LibriSpeech/train-clean-100/**/*.flac"
    train_rooms = "rir_dir/train/*.npy"
    save_path = "/root/mydir/hdd/training_data/mic_signal2"
    dataset = MicSignalDataset(train_speeches, train_rooms, save_path)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=24)

    print("start")
    num = 1
    for i in dataloader:
        print("batch", num)
        num += 1

    print("data generation finished")
    
    
    
if __name__ == '__main__':
    main()