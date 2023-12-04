import numpy as np
import os
import soundfile as sf
import scipy.signal as ss
import webrtcvad
import pickle
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm


class MicSignal:
    def __init__(self):
        print("Mic signal generator has started.")
    
    
    def _gen_anechoic_vad(self, speech, fs):
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


    def _gen_mic_signal_vad(self, rir, anechoic_vad, signal_len):
        peakidx = np.argmax(np.abs(rir))
        signal_vad = np.zeros(signal_len)
        signal_vad[peakidx : peakidx + len(anechoic_vad)] = anechoic_vad

        return signal_vad
    
    
    def _add_noise(self, signals):
        snr = np.random.randint(0, 31)
        noise_audio = np.random.standard_normal(signals.shape)
        
        clean_db = 10 * np.log10(np.mean(signals**2)+1e-4)
        noise_db = 10 * np.log10(np.mean(noise_audio**2)+1e-4)
        noise = np.sqrt(10**((clean_db - noise_db - snr) / 10)) * noise_audio
        noisy_signals = signals + noise
        
        return noisy_signals
    

    def conv_n_add(self, s, rooms, save_path):
        speech_name = s.split('/')[-1].split('.')[0]
        speech, fs = sf.read(s)
        anechoic_vad = self._gen_anechoic_vad(speech, fs)
        
        signal_num = 1
        for room in rooms:
            rirs = np.load(room)
            for dist in range(rirs.shape[1]):
                for doa in range(37):
                    # rir convolution
                    h = rirs[doa, dist, :, :]
                    signals = ss.convolve(h[:, None, :], speech[:, None, None])
                    signals = signals.squeeze()
                    # noise addition
                    signals = self._add_noise(signals)
                    # gen signal vad
                    signal_vad = self._gen_mic_signal_vad(h[:, 0], anechoic_vad, len(signals[:, 0]))

                    # save datadict
                    datadict = {}
                    datadict["doa"] = doa * 5
                    datadict["signals"] = signals
                    datadict["vad"] = signal_vad
                    
                    file_path = "%s/%d_%d.pickle" % (save_path, speech_name, signal_num)
                    with open(file_path, 'wb') as f:
                        pickle.dump(datadict, f)
                    
                    signal_num += 1
            
    
    
    # def conv_n_add(self, speeches, rooms, save_path):
    #     speech_num = 1
    #     for s in tqdm(speeches):
    #         speech, fs = sf.read(s)
    #         anechoic_vad = self._gen_anechoic_vad(speech, fs)
            
    #         signal_num = 1
    #         for room in rooms:
    #             rirs = np.load(room)
    #             for dist in range(rirs.shape[1]):
    #                 for doa in range(37):
    #                     # rir convolution
    #                     h = rirs[doa, dist, :, :]
    #                     signals = ss.convolve(h[:, None, :], speech[:, None, None])
    #                     signals = signals.squeeze()
    #                     # noise addition
    #                     signals = self._add_noise(signals)
    #                     # gen signal vad
    #                     signal_vad = self._gen_mic_signal_vad(h[:, 0], anechoic_vad, len(signals[:, 0]))

    #                     # save datadict
    #                     datadict = {}
    #                     datadict["doa"] = doa * 5
    #                     datadict["signals"] = signals
    #                     datadict["vad"] = signal_vad
                        
    #                     file_path = "%s/%d_%d.pickle" % (save_path, speech_num, signal_num)
    #                     with open(file_path, 'wb') as f:
    #                         pickle.dump(datadict, f)
                        
    #                     signal_num += 1
    #         speech_num += 1
            

def call_micsignal(speeches, rooms, path):
    micsignal = MicSignal()

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    print("Preparing the training data...")

    speeches = glob(speeches, recursive=True)
    rooms = glob(rooms)

    print("Training data is prepared.")
    
    micsignal.conv_n_add(speeches, rooms, path)



def main():
    
    # training data generation
    train_speeches = "/root/mydir/hdd/librispeech/LibriSpeech/train-clean-100/**/*.flac"
    train_rooms = "rir_dir/train/*.npy"
    save_path = "/root/mydir/hdd/training_data/mic_signal"
    call_micsignal(train_speeches, train_rooms, save_path)

    # validation data generation
    train_speeches = "/root/mydir/hdd/librispeech/LibriSpeech/dev-clean/**/*.flac"
    train_rooms = "rir_dir/validation/*.npy"
    save_path = "/root/mydir/hdd/validation_data/mic_signal"
    # call_micsignal(train_speeches, train_rooms, save_path)

    # test data generation
    
    
    
if __name__ == '__main__':
    main()