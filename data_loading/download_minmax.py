import torch
from torch import nn
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from urllib.request import urlretrieve
import pickle
import os
import math

def download(url_path, out_path):
    return urlretrieve(url_path, out_path)

minmax_path = "https://www.andrew.cmu.edu/user/afluo/required_metadata/minmax/apartment_1_minmax.pkl"

minmax_path_out = "D:/D/LNAF/3/Learning_Neural_Acoustic_Fields-master/metadata/minmax/apartment_1_minmax"

print("Downloading room bbox")
print(download(minmax_path, minmax_path_out))