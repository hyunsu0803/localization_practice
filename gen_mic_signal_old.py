import numpy as np
import os
import soundfile as sf
import scipy.signal as ss
import webrtcvad
import pickle
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def gen_mic_signal_vad(rir, anechoic_vad, signal_len):
    peakidx = np.argmax(np.abs(rir))
    signal_vad = np.zeros(signal_len)
    signal_vad[peakidx : peakidx + len(anechoic_vad)] = anechoic_vad

    return signal_vad


# main
speeches = glob("/root/mydir/hdd/librispeech_360/LibriSpeech/train-clean-100/**/*.flac", recursive=True)
rooms = glob("data/train/*.npy")
save_path = "/root/mydir/hdd/training_data/mic_signal"
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

speech_num = 1
for s in tqdm(speeches):
    speech, fs = sf.read(s)
    anechoic_vad = gen_anechoic_vad(speech, fs)
    
    signal_num = 1
    for room in rooms:
        rirs = np.load(room)
        for dist in [1, 2]:
            for doa in range(37):
                # rir convolution
                h = rirs[doa, dist, :, :]
                signals = ss.convolve(h[:, None, :], speech[:, None, None])
                signals = signals.squeeze()
                
                # gen signal vad
                signal_vad = gen_mic_signal_vad(h[:, 0], anechoic_vad, len(signals[:, 0]))

                # save datadict
                datadict = {}
                datadict['doa'] = doa * 5
                datadict["signals"] = signals
                datadict["vad"] = signal_vad
                
                file_path = "%s/%d_%d.pickle" % (save_path, speech_num, signal_num)
                with open(file_path, 'wb') as f:
                    pickle.dump(datadict, f)
                
                signal_num += 1
    speech_num += 1

