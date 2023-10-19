import numpy as np
import glob 
import os
import pickle
import random
import scipy.signal as ss
from torch_stft import STFT
import torch
import matplotlib.pyplot as plt

signal_path = './signal_samples'
train_data = glob.glob(os.path.join(signal_path, '*.pickle'))
data_size = len(train_data)

fs = 16000
Nf = 512
win_size = 32 * 16  # 512 : 32ms * 16kHz
hop_size = win_size // 2
s_len = fs*2
n_doa_class = 37

name = 1
for data_path in train_data:    # for 1 set of signals
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # 1. signal clipping    
    signals = data["signals"]
    signal_size = signals.shape[0]
    index = random.choice(range(signal_size - s_len))
    clipped_signals = signals[index : index + s_len, :]
    clipped_signals = clipped_signals.T
    
    # 2. STFT
    f, t, stft_signal = ss.stft(x=clipped_signals,      # (4, 257, 126) (channel, freq, time)
                                fs=fs, 
                                nperseg=win_size, 
                                nfft=Nf, 
                                noverlap=hop_size)      # f & t for plot (may be)
    
    # 3. phase map
    phase_map = torch.angle(torch.tensor(stft_signal, dtype=torch.cfloat))  # (4, 257, 126)

    
    # 4. target generation
    n_time_frame = stft_signal.shape[2]
    doa = data['doa']
    
    target = torch.zeros((n_doa_class, n_time_frame))
    target[doa//5, :] = 1
    
    clipped_vad = data['vad'][index : index + s_len]
    frame_vad = torch.zeros(n_time_frame)
    for nt in range(n_time_frame):
        fv = np.array(clipped_vad[hop_size*nt : hop_size*(nt+2)])
        if np.count_nonzero(fv) > len(fv) * 2/3:
            frame_vad[nt] = 1
    frame_vad = torch.unsqueeze(frame_vad, dim=0)
    target = target * frame_vad     # (37, 126)
    
    
    # 5. phase map & target dump
    output = (phase_map, target)
    output_path = "".join(['./phasemap_samples/', str(name), '.pickle'])
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    
    name += 1
    
    
    # print(target.shape)
    # print(frame_vad.shape)
    # plt.subplot(2, 1, 1)
    # plt.imshow(target, aspect='auto')
    # plt.subplot(2, 1, 2)
    # plt.plot(torch.squeeze_copy(frame_vad))
    # plt.tight_layout()
    # plt.savefig('./files/targetNvad.png')
    # exit()
    
    # print(np.sum(clipped_signals**2, axis=0))
    # mag=np.abs(stft_signal[0])
    # plt.imshow(mag, aspect='auto')
    # plt.tight_layout()
    # plt.savefig('./files/mag.png', dpi=300)
    # exit(1)
