import numpy as np
import os
import soundfile as sf
import scipy.signal as ss
import webrtcvad
import pickle
from glob import glob
import matplotlib.pyplot as plt


def gen_anechoic_vad(speech, fs):
    vad = webrtcvad.Vad()
    vad.set_mode(2)
    anechoic_vad = np.zeros_like(speech)
    vad_frame_len = int(10e-3 * fs)
    n_vad_frames = len(speech) // vad_frame_len
    
    for idx in range(n_vad_frames):
        index = idx * vad_frame_len
        frame = speech[index : index + vad_frame_len]
        frame_bytes = (frame * 32767).astype('int16').tobytes()
        anechoic_vad[index : index + vad_frame_len] = vad.is_speech(frame_bytes, fs)

    return anechoic_vad    


def gen_signal_vad(rir, anechoic_vad, signal_len):
    peakidx = np.argmax(np.abs(rir))
    signal_vad = np.zeros(signal_len)
    signal_vad[peakidx : peakidx + len(anechoic_vad)] = anechoic_vad

    return signal_vad


# main
speeches = glob("./speech_samples/*.flac")
speech_num = 1
for s in speeches:
    speech, fs = sf.read(s)
    anechoic_vad = gen_anechoic_vad(speech, fs)
    
    signal_num = 1
    for rn in [0, 1]:
        for dist in [0, 1]:
            for doa in range(37):
                datadict = {"doa" : doa*5,
                            "room_num" : rn,
                            "dist" : dist+1}
                rirs = np.load("./files/training_RIRs.npy")

                # rir convolution
                h = rirs[doa, rn, dist, :, :]
                signals = ss.convolve(h[:, None, :], speech[:, None, None])
                signals = signals.squeeze()
                
                # gen signal vad
                signal_vad = gen_signal_vad(h[:, 0], anechoic_vad, len(signals[:, 0]))

                # save datadict
                datadict["signals"] = signals
                datadict["vad"] = signal_vad
                file_path = "".join(["./signal_samples/", str(speech_num), "_", str(signal_num), ".pickle"])
                with open(file_path, 'wb') as f:
                    pickle.dump(datadict, f)
                
                signal_num += 1
    speech_num += 1

