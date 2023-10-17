import numpy as np
import os
import soundfile as sf
import scipy.signal as ss
import webrtcvad
import pickle
from glob import glob
import matplotlib.pyplot as plt

path_train = "/root/mydir/hdd/librispeech_360/LibriSpeech/train-clean-100"
path_test = "/root/mydir/hdd/librispeech_360/LibriSpeech/LibriSpeech/test-clean"

speeches = glob("./speech_samples/*.flac")


def gen_clean_vad(speech, fs):
    vad = webrtcvad.Vad()
    vad.set_mode(2)
    clean_vad = np.zeros_like(speech)
    vad_frame_len = int(10e-3 * fs)
    n_vad_frames = len(speech) // vad_frame_len
    
    clean_vad_start = 987654321
    for idx in range(n_vad_frames):
        index = idx * vad_frame_len
        frame = speech[index : index + vad_frame_len]
        frame_bytes = (frame * 32767).astype('int16').tobytes()
        clean_vad[index : index + vad_frame_len] = vad.is_speech(frame_bytes, fs)
        
        if clean_vad[index] == 1 and index < clean_vad_start:
            clean_vad_start = index

    return clean_vad, clean_vad_start


def gen_signal_vad(signal, fs, clean_vad, clean_vad_start):
    vad = webrtcvad.Vad()
    vad.set_mode(2)
    signal_vad = np.zeros_like(signal)
    vad_frame_len = int(10e-3 * fs)
    n_vad_frames = len(signal) // vad_frame_len
    
    for idx in range(n_vad_frames):
        index = idx * vad_frame_len
        frame = signal[index : index + vad_frame_len]
        frame_bytes = (frame * 32767).astype('int16').tobytes()
    
        if vad.is_speech(frame_bytes, fs):
            # print("clean vad start", clean_vad_start)
            # print("len(clean_vad[clean_vad_start:])", len(clean_vad[clean_vad_start:]))
            # print("len(signal vad)", len(signal_vad))
            # print("len(signal_vad[index:])", len(signal_vad[index:]))
            # print("index", index)
            if len(signal_vad[index:]) >= len(clean_vad[clean_vad_start:]):
                signal_vad[index : index + len(clean_vad[clean_vad_start:])] = clean_vad[clean_vad_start:]
            else:
                signal_vad[index:] = clean_vad[clean_vad_start: clean_vad_start + len(signal_vad[index:])]
            break

    return signal_vad


# main
for s in speeches:
    speech, fs = sf.read(s)
    clean_vad, clean_vad_start = gen_clean_vad(speech, fs)
    speech_num = 1
    for rn in [0, 1]:
        for dist in [0, 1]:
            for doa in range(37):
                signal_num = 1
                datadict = {"doa" : doa*5,
                            "room_num" : rn,
                            "dist" : dist+1}
                rirs = np.load("./files/training_RIRs.npy")

                # rir convolution
                h = rirs[doa, rn, dist, :, :]
                signals = ss.convolve(h[:, None, :], speech[:, None, None])
                signals = signals.squeeze()
                # gen signal vad
                signal_vad = gen_signal_vad(signals[:, 0], fs, clean_vad, clean_vad_start)

                datadict["signals"] = signals
                datadict["vad"] = signal_vad

                file_path = ["./signal_samples/", str(speech_num), "_", str(signal_num) ]
                with open(file_path)

