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
        

def gen_signal_dicts(speech, fs, rir):
    
    speech, fs = sf.read("./files/1334-135589-0016.flac")

    vad = webrtcvad.Vad()
    vad.set_mode(2)
    speech_vad = np.zeros_like(speech)
    vad_frame_len = int(10e-3 * fs)
    n_vad_frames = len(speech) // vad_frame_len
    
    speech_vad_start = 987654321
    for idx in range(n_vad_frames):
        index = idx * vad_frame_len
        frame = speech[index : index + vad_frame_len]
        frame_bytes = (frame * 32767).astype('int16').tobytes()
        speech_vad[index : index + vad_frame_len] = vad.is_speech(frame_bytes, fs)
        
        if speech_vad[index] == 1 and index < speech_vad_start:
            speech_vad_start = index
        
    rirs = np.load("./files/training_RIRs.npy")
    doas = [5]
    for d in doas:
        h = rirs[d, 0, 0, :, :]
        signals = ss.convolve(h[:, None, :], speech[:, None, None])
        signals = signals.squeeze()
        # sf.write("vad_test.wav", signals[:, 0], 16000)
        # print(signals.shape)
    
    signals_vad = np.zeros_like(signals[:, 0])
    
    # we shift the clean vad only for the 0th channel
    signal = signals[:, 0]
    vad_frame_len = int(10e-3 * fs)
    n_vad_frames = len(signal) // vad_frame_len
    
    for idx in range(n_vad_frames):
        index = idx * vad_frame_len
        frame = signal[index : index + vad_frame_len]
        frame_bytes = (frame * 32767).astype('int16').tobytes()
    
        if vad.is_speech(frame_bytes, fs):
            signals_vad[index : index + len(speech_vad[speech_vad_start:])] = speech_vad[speech_vad_start:]
            break
    
    datadict = {"signals" : signals,
                "vad" : signals_vad,
                "doa" : doas[0]*5,
                "room_num" : 0,
                "dist" : 1}

    # with open('./files/test_data_1.pickle', 'wb') as f:
    #     pickle.dump(datadict, f)
        
    # with open('./files/test_data_1.pickle', 'rb') as f:
    #     testdict = pickle.load(f)
    #     print(testdict["signals"].shape)
    #     print("room num", testdict["room_num"])
    #     print(np.count_nonzero(testdict["vad"]))
        

def main():
    # gen_chapterlist()
    
    # with open('chapterlist.pickle', 'rb') as f:
    #     chapterlist = pickle.load(f)
    
    # vad_test()
    pass
    

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