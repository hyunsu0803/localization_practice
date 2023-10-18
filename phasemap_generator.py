import numpy as np
import glob 
import os
import pickle
import random
import scipy.signal as ss
from torch_stft import STFT
import torch


signal_path = './signal_samples'
train_data = glob.glob(os.path.join(signal_path, '*.pickle'))
data_size = len(train_data)

fs = 16000
Nf = 512
win_size = 32 * 16  # 512 : 32ms * 16kHz
hop_size = win_size // 2

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
    f, t, stft_signal = ss.stft(x=clipped_signals,      # (4, 257, 126)
                                fs=fs, 
                                nperseg=win_size, 
                                nfft=Nf, 
                                noverlap=hop_size)   # f & t for plot (may be)
    
    # 3. phase map
    for t in range(stft_signal.shape[2]):
        initial_map = stft_signal[:, :, t]
        initial_map = torch.tensor(initial_map, dtype=torch.cfloat)
        phase_map = torch.angle(initial_map)
        
        
        # 4. target generation
        sigvad = np.array(data['vad'])
        sigvad = sigvad[index + hop_size * t : index + hop_size * (t+2)]
        if np.count_nonzero(sigvad) > len(sigvad) * 2//3:
            doa = data['doa']
            target = torch.zeros(37)
            target[doa//5] = 1
        else:
            target = torch.zeros(37)
        
        output = (phase_map, target)

        exit()
    
    
    exit()