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
