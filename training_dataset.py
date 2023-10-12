import numpy as np
import os
from collections import namedtuple
from torch.utils.data import Dataset
import torch
import scipy
import scipy.io.wavfile
import soundfile as sf
import scipy.signal as ss
import webrtcvad
import pickle
from glob import glob
import matplotlib.pyplot as plt


path_train = "/root/mydir/hdd/librispeech_360/LibriSpeech/train-clean-100"
path_test = "/root/mydir/hdd/librispeech_360/LibriSpeech/LibriSpeech/test-clean"


def parsingDB(path, file_extension):
    directory_tree = {}
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            directory_tree[item] = parsingDB(os.path.join(path, item), file_extension)
        elif item.split(".")[-1] == file_extension:
            directory_tree[item.split(".")[0]] = os.path.join(path, item)
    return directory_tree


def gen_chapterlist():
    # db=glob(path_train+'**/*.flac', recursive=True)
    # print(db)
    # exit(1)
    db = parsingDB(path_train, "flac")
    nSpeakers = len(db)
    nChapters_per_speakers = {speaker : len(db[speaker]) 
                                    for speaker in db.keys()}
    nUtterances_per_chapters_per_speakers = {
        speaker : {
            chapter : len(db[speaker][chapter]) 
                    for chapter in db[speaker].keys()
            } for speaker in db.keys()
    }
    db_info = {"nSpeakers" : nSpeakers,     
            "nChapters_per_speakers" : nChapters_per_speakers,
            "nUtterances_per_chapters_per_speakers" : nUtterances_per_chapters_per_speakers
    }
    chapterlist = []
    for chapters in list(db.values()):  # chapters per speaker
        chapterlist += list(chapters.values())  # all the utterences

    with open('chapterlist.pickle', 'wb') as f:
        pickle.dump(chapterlist, f)
    with open('db_info.pickle', 'wb') as f2:
        pickle.dump(db_info, f2)
        

def vad_test():
    
    datadict = {"s"}
    
    speech, fs = sf.read("1334-135589-0016.flac")

    vad = webrtcvad.Vad()
    vad.set_mode(2)
    vad_out = np.zeros_like(speech)
    vad_frame_len = int(10e-3 * fs)
    n_vad_frames = len(speech) // vad_frame_len
    
    for idx in range(n_vad_frames):
        frame = speech[idx * vad_frame_len : (idx+1) * vad_frame_len]
        frame_bytes = (frame * 32767).astype('int16').tobytes()
        vad_out[idx * vad_frame_len : (idx+1) * vad_frame_len] = vad.is_speech(frame_bytes, fs)
        
    print(list(vad_out).count(1))
    plt.subplot(5, 1, 1)
    plt.plot(vad_out)
    
    rirs = np.load("training_RIRs.npy")
    doas = [5]
    for d in doas:
        h = rirs[d, 0, 0, :, :]
        signals = ss.convolve(h[:, None, :], speech[:, None, None])
        signals = signals.squeeze()
        # sf.write("vad_test.wav", signals[:, 0], 16000)
        # print(signals.shape)
    
    for i in range(4):
        speech = signals[:, i]
        vad_out = np.zeros_like(speech)
        vad_frame_len = int(10e-3 * fs)
        n_vad_frames = len(speech) // vad_frame_len
        
        for idx in range(n_vad_frames):
            frame = speech[idx * vad_frame_len : (idx+1) * vad_frame_len]
            frame_bytes = (frame * 32767).astype('int16').tobytes()
            vad_out[idx * vad_frame_len : (idx+1) * vad_frame_len] = vad.is_speech(frame_bytes, fs)
            
        print(list(vad_out).count(1))
        plt.subplot(5, 1, i+2)
        plt.plot(vad_out)
        
    plt.tight_layout()
    plt.savefig("vadtest2.png")
        


def main():
    # gen_chapterlist()
    
    # with open('chapterlist.pickle', 'rb') as f:
    #     chapterlist = pickle.load(f)
    
    vad_test()
    

if __name__ == '__main__':
    main()
    
# vad, rir, noise 


# with open('./dataset_target_test/chapterlist.pickle', 'rb') as f:
#     chapterlist = pickle.load(f)
# speeches = []
# for chapter in chapterlist[50:51]:
#     for utt in chapter.values():
#         speech, fs = sf.read(utt, always_2d=True)
#         speeches.append(speech)
#         print(speech.shape)
#         print(fs)
#         break
# rirs = np.load("training_RIRs.npy")
# doas = [5]
# for d in doas:
#     h = rirs[d, 0, 0, :, :]
#     signals = ss.convolve(h[:, None, :], speeches[0][:, :, None])
#     signals = signals.squeeze()
#     sf.write("test.wav", signals[:, 0], 16000)