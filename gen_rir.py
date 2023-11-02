import numpy as np
import scipy.signal as ss
import soundfile as sf
import rir_generator
import os
import json


class RIR:
    def __init__(self, room):
        # 1 rir.npy per room
        self.room_dim = room['dim']
        self.rt60 = room['rt60']
        self.mcenter = room['mcenter']
        self.room_num = room['num']
        
        self.marray = np.array([
                    [-0.12, 0, 0],
                    [-0.04, 0, 0],
                    [0.04, 0, 0],
                    [0.12, 0, 0]
                    ])
        self.dist = np.array([1, 2])
        self.doa_range = range(0, 185, 5)
        self.rir_length = 4096
        self.rirs = None
        
    def generate_rir(self):
        # (DOA, dist, rir, channel)
        self.rirs = np.zeros((len(self.doa_range), 
                              len(self.dist)+1, 
                              self.rir_length, 
                              len(self.marray)))
        
        for d in self.dist:
            for deg in self.doa_range:
                th = np.deg2rad(deg)
                source_pos = np.array([d*np.cos(th), d*np.sin(th), 0]) + self.mcenter
        
                h = rir_generator.generate(
                    c=340,
                    fs=16000,
                    r=self.marray + self.mcenter,   # receiver position
                    s=source_pos,                   # source position
                    L=self.room_dim,                # room dim
                    reverberation_time=self.rt60,
                    nsample=self.rir_length
                    )
                
                doa_class = deg // 5
                self.rirs[doa_class, d, :, :] = h
        
        save_dir = './files'
        save_name = 'rir%d.npy' % self.room_num
        np.save(os.path.join(save_dir, save_name))
        
        
def main():
    
    with open('./files/config.json', 'r') as f:
        config = json.load(f)
        
    
    a = config['test'].split()
    a = [float(i) for i in a]
    print(a)
    
    # r1 = {'dim' : np.array([6, 6, 2.7]),
    #       'rt60' : 0.3,
    #       'mcenter' : ,
    #       'num': }
    

        
if __name__ == '__main__':
    main()