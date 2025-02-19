import torchaudio
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.io import wavfile
from torchaudio.transforms import Spectrogram
import librosa
import os

import gc
import pickle


def load_audio(path_name, use_torch=True, resample=True, resample_rate=22050):
    # returns in shape (ch, num_sample), as float32 (on Linux at least)
    # by default torchaudio is wav_arr, sample_rate
    # by default wavfile is sample_rate, wav_arr
    if use_torch:
        loaded = torchaudio.load(path_name)
        wave_data_loaded = loaded[0].numpy()
        sr_loaded = loaded[1]
    else:
        loaded = wavfile.read(path_name)
        wave_data_loaded = np.clip(loaded[1], -1.0, 1.0).T
        sr_loaded = loaded[0]

    if resample:
        if wave_data_loaded.shape[1] == 0:
            print("len 0")
            assert False
        if wave_data_loaded.shape[1] < int(sr_loaded * 0.1):
            padded_wav = librosa.util.fix_length(wave_data_loaded, int(sr_loaded * 0.1))
            resampled_wave = librosa.resample(padded_wav, orig_sr=sr_loaded, target_sr=resample_rate)
        else:
            resampled_wave = librosa.resample(wave_data_loaded, orig_sr=sr_loaded, target_sr=resample_rate)
    else:
        resampled_wave = wave_data_loaded
    return np.clip(resampled_wave, -1.0, 1.0)


def if_compute(arg):
    unwrapped_angle = np.unwrap(arg).astype(np.single)
    return np.concatenate([unwrapped_angle[:, :, 0:1], np.diff(unwrapped_angle, n=1)], axis=-1)


class get_spec():
    def __init__(self, use_torch=False, power_mod=2, fft_size=512):
        self.n_fft = fft_size
        self.hop = self.n_fft // 4
        if use_torch:
            assert False
            self.use_torch = True
            self.spec_transform = Spectrogram(power=None, n_fft=self.n_fft, hop_length=self.hop)
        else:
            self.power = power_mod
            self.use_torch = False
            self.spec_transform = None

    def transform(self, wav_data_prepad):
        wav_data = librosa.util.fix_length(data=wav_data_prepad, size=wav_data_prepad.shape[-1] + self.n_fft // 2)
        if wav_data.shape[1] < 4410:
            wav_data = librosa.util.fix_length(wav_data, size=4410)
        if self.use_torch:
            transformed_data = self.spec_transform(torch.from_numpy(wav_data)).numpy()
        else:

            transformed_data = np.array([librosa.stft(wav_data[0], n_fft=self.n_fft, hop_length=self.hop),
                                         librosa.stft(wav_data[1], n_fft=self.n_fft, hop_length=self.hop)])[:, :-1]
        #         print(np.array([librosa.stft(wav_data[0],n_fft=self.n_fft, hop_length=self.hop),
        #                librosa.stft(wav_data[1],n_fft=self.n_fft, hop_length=self.hop)]).shape, "OLD SHAPE")

        real_component = np.abs(transformed_data)
        img_component = np.angle(transformed_data)
        gen_if = if_compute(img_component) / np.pi
        return np.log(real_component + 1e-3), gen_if, img_component


def get_wave_if(input_stft, input_if):
    # 2 chanel input of shape [2,freq,time]
    # First input is logged mag
    # Second input is if divided by np.pi
    padded_input_stft = np.concatenate((input_stft, input_stft[:, -1:]), axis=1)
    padded_input_if = np.concatenate((input_if, input_if[:, -1:] * 0.0), axis=1)
    unwrapped = np.cumsum(padded_input_if, axis=-1) * np.pi
    phase_val = np.cos(unwrapped) + 1j * np.sin(unwrapped)
    restored = (np.exp(padded_input_stft) - 1e-3) * phase_val
    wave1 = librosa.istft(restored[0], hop_length=512 // 4)
    wave2 = librosa.istft(restored[1], hop_length=512 // 4)
    return wave1, wave2


try:
    f_mag.close()
except:
    pass

try:
    f_phase.close()
except:
    pass

raw_path = "D:/D/LNAF/3/Learning_Neural_Acoustic_Fields-master/media/aluo/big2/soundspaces_full/binaural_rirs/replica"
mag_path = "D:/D/LNAF/3/Learning_Neural_Acoustic_Fields-master/metadata/magnitudes"
phase_path = "D:/D/LNAF/3/Learning_Neural_Acoustic_Fields-master/metadata/phases"
spec_getter = get_spec()
room_names = os.listdir(raw_path)
print(room_names)
for room_name in room_names:
    length_tracker = []
    print(room_name)
    room_path = os.path.join(raw_path, room_name)
    mag_object = os.path.join(mag_path, room_name)
    phase_object = os.path.join(phase_path, room_name)
    f_mag = h5py.File(mag_object + ".h5", 'w')
    f_phase = h5py.File(phase_object + ".h5", 'w')
    zz = 0
    for orientation in ["0", "90", "180", "270"]:
        ori_path = os.path.join(room_path, orientation)
        files = sorted(os.listdir(ori_path))
        files = [_ for _ in files if "wav" in _]
        if orientation == "0":
            print("Found {} files".format(str(len(files) * 4)))

        for ff in files:
            zz += 1
            if zz % 500 == 0:
                print(zz)
            cur_file = os.path.join(ori_path, ff)
            loaded_wav = load_audio(cur_file, use_torch=False)
            real_spec, img_spec, raw_phase = spec_getter.transform(loaded_wav)
            length_tracker.append(real_spec.shape[2])
            #             reconstructed_wave = get_wave_if(real_spec, img_spec)
            #             plt.imshow(real_spec[0])
            #             plt.show()
            #             plt.imshow(img_spec[0])
            #             plt.show()
            f_mag.create_dataset('{}_{}'.format(orientation, ff.split(".")[0]), data=real_spec.astype(np.half))
            f_phase.create_dataset('{}_{}'.format(orientation, ff.split(".")[0]), data=img_spec.astype(np.half))
    print("Max length {}".format(room_name), np.max(length_tracker))
    f_mag.close()
    f_phase.close()


def pad(input_arr, max_len_in, constant=np.log(1e-3)):
    return np.pad(input_arr, [[0, 0], [0, 0], [0, max_len_in - input_arr.shape[2]]], constant_values=constant)


raw_path = "D:/D/LNAF/3/Learning_Neural_Acoustic_Fields-master/metadata/magnitudes"
mean_std = "D:/D/LNAF/3/Learning_Neural_Acoustic_Fields-master/metadata/mean_std"

#"apartment_1": 101,
#max_len_dict = { "apartment_2": 86, "frl_apartment_2": 107, "frl_apartment_4": 103, "office_4": 78,
            #    "room_2": 84}
max_len_dict = {  "room_2": 84}

files = os.listdir(raw_path)
for f_name_old in sorted(list(max_len_dict.keys())):
    f_name = f_name_old + ".h5"
    print("Processing ", f_name)
    f = h5py.File(os.path.join(raw_path, f_name), 'r')
    keys = list(f.keys())
    max_len = max_len_dict[f_name.split(".")[0]]
    all_arrs = []
    for idx in np.random.choice(len(keys), 4000, replace=False):
        all_arrs.append(pad(f[keys[idx]], max_len).astype(np.single))
    all_arrs_2 = np.array(all_arrs, copy=False, dtype=np.single)
    print("Computing mean")
    mean_val = np.mean(all_arrs, axis=(0, 1))
    print("Computing std")
    std_val = np.std(all_arrs, axis=(0, 1)) + 0.1

    plt.imshow(all_arrs[0][0])
    plt.show()
    plt.imshow(mean_val)
    plt.show()
    plt.imshow(std_val)
    plt.show()
    print(mean_val.shape)
    del all_arrs
    f.close()
    gc.collect()

    with open(os.path.join(mean_std, f_name.replace("h5", "pkl")), "wb") as mean_std_file:
        pickle.dump([mean_val, std_val], mean_std_file)