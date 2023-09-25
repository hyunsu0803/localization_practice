import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.signal import stft
from random import uniform, sample
from pyroomacoustics import doa, Room, ShoeBox
import pickle
import soundfile as sf
import scipy.signal as ss

# constants / config
fs = 16000
nfft = 1024
n = 5*fs
n_frames = 30
max_order = 10
doas_deg = np.linspace(start=0, stop=359, num=360, endpoint=True)
rs = [0.5, 1, 1.5]
mcenter = np.array([
    [3, 2, 1.0]
])
marray = np.array([
    [-0.12, 0, 0],
    [-0.04, 0, 0],
    [0.04, 0, 0],
    [0.12, 0, 0]
])
mic_locs = mcenter + marray
mic_locs = mic_locs.T
# mic_center = np.c_[[2, 2, 1]]   # column vector
# mic_locs = mic_center + np.c_[[0.04, 0.0, 0.0],
#                               [0.0, 0.04, 0.0],
#                               [-0.04, 0.0, 0.0],
#                               [0.0, -0.04, 0.0]]    # transpose 
snr_lb, snr_ub = 0, 30

# room simulation
data = []
"""
for r in rs:
    for i, doa_deg in enumerate(doas_deg):
        doa_rad = np.deg2rad(doa_deg)
        source_loc = mic_center[:, 0] + np.c_[r*np.cos(doa_rad), r*np.sin(doa_rad), 0][0]
        room_dim = [uniform(4, 6), uniform(4, 6), uniform(2, 4)]    # meters

        room = ShoeBox(room_dim, fs=fs, max_order=max_order)
        room.add_source(source_loc, signal=np.random.random(n))
        room.add_microphone_array(mic_locs)
        room.simulate(snr=uniform(snr_lb, snr_ub))
        signals = room.mic_array.signals    # (4, *****)
# ------------------------------------------------------------------------------------------------
        # calculate n_frames stft frames starting at 1 second
        stft_signals = stft(signals[:, fs:fs*n_frames*nfft], fs=fs, nperseg=nfft, noverlap=0, boundary=None)[2]
        data.append([r, doa_deg, stft_signals])
"""

with open('chapterlist.pickle', 'rb') as f:
    chapterlist = pickle.load(f)
#---------for MUSIC test-----------
speeches = []
for chapter in chapterlist[:2]:
    for utt in chapter.values():
        speech, fs = sf.read(utt, always_2d=True)
        speeches.append(speech)
        print(speech.shape)
        print(fs)
        break
rirs = np.load("music_test_RIRs.npy")
doas = [5, 20]
for d in doas:
    h = rirs[d, 0, 0, :, :]
    print(h.shape)
    print(speeches[0].shape)
    signals = ss.convolve(h[:, None, :], speeches[0][:, :, None])
    print(signals.shape)
    signals = signals.squeeze().T
    print(signals.shape)
    stft_signals = stft(signals[:, fs:fs*n_frames*nfft], fs=fs, nperseg=nfft, noverlap=0, boundary=None)[2]
    print(stft_signals.shape)
    data.append([1, d*5, stft_signals])
    
# exit()
# Prediction
kwargs = {'L' : mic_locs,
          'fs' : fs,
          'nfft' : nfft,
          'azimuth' : np.deg2rad(np.array([5*5, 20*5]))
}
algorithms = {
    'MUSIC' : doa.music.MUSIC(**kwargs),
    'NormMUSIC' : doa.normmusic.NormMUSIC(**kwargs)
}
columns = ["r", "DOA"] + list(algorithms.keys())    # r DOA MUSIC NormMUSIC

predictions = {n:[] for n in columns}
for r, doa_deg, stft_signals in data:
    predictions['r'].append(r)
    predictions['DOA'].append(doa_deg)
    for algo_name, algo in algorithms.items():
        algo.locate_sources(stft_signals)
        predictions[algo_name].append(np.rad2deg(algo.azimuth_recon[0]))
df = pd.DataFrame.from_dict(predictions)

print(df)

# Evaluation
MAE, MEDAE = {}, {}

def calc_ae(a, b):
    x = np.abs(a-b)
    return np.min(np.array((x, np.abs(360-x))), axis=0)

for algo_name in algorithms.keys():
    ae = calc_ae(df.loc[:, ["DOA"]].to_numpy(), df.loc[:, [algo_name]].to_numpy())
    MAE[algo_name] = np.mean(ae)
    MEDAE[algo_name] = np.median(ae)

print(f"MAE\t MUSIC: {MAE['MUSIC']:5.2f}\t NormMUSIC: {MAE['NormMUSIC']:5.2f}")
print(f"MEDAE\t MUSIC: {MEDAE['MUSIC']:5.2f}\t NormMUSIC: {MEDAE['NormMUSIC']:5.2f}")

# more
# fig = plt.figure(figsize=(14, 10))
# frequencies = sample(list(range(algorithms['MUSIC'].Pssl.shape[1])), k=10)
# for i, k in enumerate(frequencies):
#     plt.plot(algorithms['MUSIC'].Pssl[:, k])
# plt.xlabel("angle [* ]")
# plt.title("Multiple narrowband MUSIC pseudo spectra in one plot", fontsize=15)

# # calculation
# maxima = sorted(np.max(algorithms["MUSIC"].Pssl, axis=0))

# # plotting
# fig, ax = plt.subplots(1, 1, figsize=(14, 10))
# sns.swarmplot(data=maxima, ax=ax, size=6)

# ax.set_title("\nDistribution: Maxima of the MUSIC pseudo spectra of multiple frequency bins\n", fontsize=20)
# ax.set_xticks([1])

# plt.show()