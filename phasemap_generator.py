import numpy as np
import glob 
import os
import pickle
import random

signal_path = './signal_samples'
train_data = glob.glob(os.path.join(signal_path, '*.pickle'))
size = len(train_data)
fs = 16000

s_len = fs*2

for data_path in train_data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # 1. signal clipping    
    signals = data["signals"]
    signal_size = signals.shape[0]
    index = random.choice(range(signal_size - s_len))
    clipped_signals = signals[index : index + s_len, :]
    
    # 2. STFT
    
    
    
    exit()