import numpy as np
import scipy.signal as ss
import soundfile as sf
import rir_generator as rir
import matplotlib.pyplot as plt


def rir_gen(mpos, source_pos, room_dim, rt60):
    h = rir.generate(
    c=340,
    fs=16000,
    r=mpos,             # receiver position
    s=source_pos,       # source position
    L=room_dim,            # room dim
    reverberation_time=0.2,
    nsample=4096
    )
    
    return h


signal, fs = sf.read("p287_001.wav", always_2d=True)
print(signal.shape)

# set up
room_dim = np.array([            # room dimensions
    [6, 6, 2.7],
    [8, 3, 2.7]
])
rt60 = np.array([0.3, 0.5])
mcenter = np.array([
    [3, 2, 1.0],
    [4, 0.8, 1.2]
])
marray = np.array([
    [-0.12, 0, 0],
    [-0.04, 0, 0],
    [0.04, 0, 0],
    [0.12, 0, 0]
])


# """
rirs = np.zeros((37, 2, 2, 4096, 4))    # DOA, room, dist, rir, channel
# generation
for i in [0]:
    mpos = marray + mcenter[i]
    # print(mpos)
    for d in [1]:
        for t in [5, 20]:
            # th = np.deg2rad(t)
            th = np.deg2rad(t*5)
            source_pos = np.array([d*np.cos(th), d*np.sin(th), 0]) + mcenter[i]
            h = rir_gen(mpos, source_pos, room_dim[i], rt60[i])
            
            # doa = int(t/5)
            doa = t
            rirs[doa, i, d-1, :, :] = h
            # print(h)
            # print(rirs[doa, i, d-1, :, :])
            # exit()

np.save('music_test_RIRs.npy', rirs)
# """


# rirs = np.load("training_RIRs.npy")
  
# h = rirs[0, 0, 0, :, :]
# print(h.shape)
# print(signal.shape)
# signal = ss.convolve(h[:, None, :], signal[:, :, None])
# print(signal.shape)     
# sf.write("test9.wav", signal[:, :, 0], fs)
