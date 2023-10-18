import numpy as np
import glob 
import os
import pickle
import random
import scipy.signal as ss


signal_path = './signal_samples'
train_data = glob.glob(os.path.join(signal_path, '*.pickle'))
data_size = len(train_data)

fs = 16000
Nf = 512
win_size = 32 * 16  # 512

s_len = fs*2

for data_path in train_data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # 1. signal clipping    
    signals = data["signals"]
    signal_size = signals.shape[0]
    index = random.choice(range(signal_size - s_len))
    clipped_signals = signals[index : index + s_len, :]
    clipped_signals = clipped_signals.T
    
    # 2. STFT
    f, t, stft_signal = ss.stft(x=clipped_signals, fs=fs, nperseg=win_size, nfft=Nf)
    print(f, t)
    
    exit()