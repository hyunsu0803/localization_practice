import numpy as np
import scipy.signal as ss
import soundfile as sf
import rir_generator
import os
import json
import random


class RIR:
    def __init__(self, room):        
        self.marray = np.array([
                    [-0.12, 0, 0],
                    [-0.04, 0, 0],
                    [0.04, 0, 0],
                    [0.12, 0, 0]
                    ])
        self.doa_range = range(0, 185, 5)
        self.rir_length = 4096

        self.room_dim = room['dim']
        self.rt60 = room['rt60']
        self.mcenter = room['mcenter']
        self.room_num = room['num']
        self.dist = room['dist']
        self.rirs = np.zeros((len(self.doa_range), 
                              len(self.dist), 
                              self.rir_length, 
                              len(self.marray)))
        
    def _is_ok(self, x, y):
        if x <= 0 or y <= 0 or x >= self.room_dim[0] or y >= self.room_dim[1]:
            return False
        return True
        
    def generate_rir(self, save_dir):
        # (DOA, dist, rir, channel)
        # 1 rir.npy per room
        
        for i, d in enumerate(self.dist):
            for deg in self.doa_range:
                th = np.deg2rad(deg)

                source_pos = np.array([d*np.cos(th), d*np.sin(th), 0]) + self.mcenter
                if not self._is_ok(source_pos[0], source_pos[1]):
                    continue
        
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
                self.rirs[doa_class, i, :, :] = h
        
        save_name = 'rir%d.npy' % self.room_num
        np.save(os.path.join(save_dir, save_name), self.rirs)
        print("Saved", save_name)


def call_rir(dir):
    with open('%s/config.json' % dir, 'r') as f:
        config = json.load(f)
    
    for n in range(len(config)):

        r = config[str(n+1)]
        
        r["dim"] = np.array([float(i) for i in r['dim'].split()])
        x = random.uniform(2, r['dim'][0]-2)
        y = random.uniform(2, r['dim'][1]-2)
        z = random.uniform(1.5, 1.8)
        r["mcenter"] = np.array([x, y, z])
        r["num"] = n+1
        r["dist"] = np.array([float(i) for i in r["dist"].split()])
        
        rir = RIR(r)
        rir.generate_rir(dir)

        
def main():
    
    train_rir_dir = "./rir_dir/train"
    valid_rir_dir = "./rir_dir/validation"
    test_rir_dir = "./rir_dir/test"

    # call_rir(train_rir_dir)
    call_rir(valid_rir_dir)
    # call_rir(test_rir_dir)
    
        
if __name__ == '__main__':
    main()