#-*- coding: UTF-8 -*- 
import torch
import os
import random
import difflib, librosa
import wave
import numpy as np
import scipy
import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from augmentor.specaugment import specaug
from augmentor.libaug import speed_tune, pitch_tune, specaugment

sample_rate = 16000
window_size = 0.02
window_stride = 0.01
n_mels = 161
n_fft = int(sample_rate * window_size)
win_length = n_fft
hop_length = int(sample_rate * window_stride)
window = "hamming"

def load_audio(wav_path, normalize=True):  # -> numpy array
	if ".wav" in wav_path:
		with wave.open(wav_path) as wf:
			wav = np.frombuffer(wf.readframes(wf.getnframes()), dtype="int16")
			wav = wav.astype("float")
	elif ".bin" in wav_path:
		wav = np.fromfile(wav_path, dtype="int16")
		wav = wav.astype("float")
	else:
		print("Error: ", wav_path)
		raise ValueError("Error: "+wav_path)
		
	if normalize:
		return (wav - wav.mean()) / wav.std()
	else:
		return wav


def magphase(D):
    """Separate a complex-valued spectrogram D into its magnitude (S)
    and phase (P) components, so that `D = S * P`.
    Parameters
    ----------
    D       : np.ndarray [shape=(d, t), dtype=complex]
        complex-valued spectrogram
    Returns
    -------
    D_mag   : np.ndarray [shape=(d, t), dtype=real]
        magnitude of `D`
    D_phase : np.ndarray [shape=(d, t), dtype=complex]
        `exp(1.j * phi)` where `phi` is the phase of `D`
    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> magnitude, phase = librosa.magphase(D)
    >>> magnitude
    array([[  2.524e-03,   4.329e-02, ...,   3.217e-04,   3.520e-05],
           [  2.645e-03,   5.152e-02, ...,   3.283e-04,   3.432e-04],
           ...,
           [  1.966e-05,   9.828e-06, ...,   3.164e-07,   9.370e-06],
           [  1.966e-05,   9.830e-06, ...,   3.161e-07,   9.366e-06]], dtype=float32)
    >>> phase
    array([[  1.000e+00 +0.000e+00j,   1.000e+00 +0.000e+00j, ...,
             -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j],
           [  1.000e+00 +1.615e-16j,   9.950e-01 -1.001e-01j, ...,
              9.794e-01 +2.017e-01j,   1.492e-02 -9.999e-01j],
           ...,
           [  1.000e+00 -5.609e-15j,  -5.081e-04 +1.000e+00j, ...,
             -9.549e-01 -2.970e-01j,   2.938e-01 -9.559e-01j],
           [ -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j, ...,
             -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j]], dtype=complex64)
    Or get the phase angle (in radians)
    >>> np.angle(phase)
    array([[  0.000e+00,   0.000e+00, ...,   3.142e+00,   3.142e+00],
           [  1.615e-16,  -1.003e-01, ...,   2.031e-01,  -1.556e+00],
           ...,
           [ -5.609e-15,   1.571e+00, ...,  -2.840e+00,  -1.273e+00],
           [  3.142e+00,   3.142e+00, ...,   3.142e+00,   3.142e+00]], dtype=float32)
    """
 
    mag = np.abs(D)
    phase = np.exp(1.j * np.angle(D))
 
    return mag, phase

def spectrogram(wav, mel_spec=False ,normalize=True):
# librosa
	if not mel_spec:
		D = librosa.stft(
			wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
		)

		spec, phase = magphase(D)
		spec = np.log1p(spec)
	else:
		spec = librosa.feature.melspectrogram(wav,
											 sr=sample_rate,
											 n_mels=n_mels,
											 hop_length=hop_length,
											 win_length=win_length,
											 window=window,
											 n_fft=n_fft)
		spec = librosa.power_to_db(spec)  # 转化频谱系数单位
		# spec = spec.astype(np.float32)
	
# handled
	
	spec = torch.FloatTensor(spec)

	if normalize:
		spec = (spec - spec.mean()) / spec.std()

	return spec

# 获取信号的时频图
def compute_fbank(file, normalize=True):
	# _path = file.replace(".bin", ".npz")
	# try:
		# if os.path.exists(_path):
			# _data = np.load(_path)["data"]
			# return _data
	# except Exception as e:
		# print("Error:", e)
	x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
	w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
	# fs, wavsignal = wav.read(file)
	fs = 16000
	wavsignal = np.fromfile(file, dtype=np.int16)
	# wav波形 加时间窗以及时移10ms
	time_window = 25  # 单位ms
	wav_arr = wavsignal # np.array(wavsignal)
	range0_end = int(len(wavsignal) / fs * 1000 - time_window) // 10 + 1 # 计算循环终止的位置，也就是最终生成的窗数
	data_input = np.zeros((range0_end, 200), dtype=np.float)  # 用于存放最终的频率特征数据
	data_line = np.zeros((1, 400), dtype=np.float)
	for i in range(0, range0_end):
		p_start = i * 160
		p_end = p_start + 400
		data_line = wav_arr[p_start:p_end]
		data_line = data_line * w  # 加窗
		data_line = np.abs(fft(data_line))
		data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的
	data_input = np.log(data_input + 1)
	
	spec = torch.FloatTensor(data_input)

	if normalize:
		spec = (spec - spec.mean()) / spec.std()
	
	# data_input = data_input[::]
	return spec
	
def GetEditDistance(str1, str2):
	leven_cost = 0
	s = difflib.SequenceMatcher(None, str1, str2)
	for tag, i1, i2, j1, j2 in s.get_opcodes():
		if tag == 'replace':
			leven_cost += max(i2-i1, j2-j1)
		elif tag == 'insert':
			leven_cost += (j2-j1)
		elif tag == 'delete':
			leven_cost += (i2-i1)
	return leven_cost

class MASRDataset(Dataset):
	def __init__(self, index_path, labels_path, mode = "train", config = None, device_type = None):
	
		self.mode = mode
		self.config = config
		self.device_type = device_type
#		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		if ".json" not in index_path:
			with open(index_path) as f:
				idx = f.readlines()
			idx = [x.strip().split(",", 1) for x in idx]
		else:
			with open(index_path, "r") as f:
				idx = json.load(f)
		self.idx = idx
		with open(labels_path) as f:
			labels = json.load(f)
		self.labels = dict([(labels[i], i) for i in range(len(labels))])
		self.labels_str = labels

	def __getitem__(self, index):
		wav_path, transcript = self.idx[index]
		if self.mode == "dev" and self.device_type is not None:
			wav_path =  "/kaggle/input/magicdata/dev_device/dev_byte(%s)/"%self.device_type + os.path.basename(wav_path)#/root
		if self.mode == "test" and self.device_type is not None:
			wav_path =  "/kaggle/input/magicdata/test/test_byte(%s)/"%self.device_type + os.path.basename(wav_path)
		wav = load_audio(wav_path)
		
		if self.mode == "train":
			if self.config.speed and self.config.pitch:
				wav = speed_tune(wav, prob = 0.33) if random.random() <= 0.75 else pitch_tune(wav, prob = 0.33)
			elif self.config.speed:
				wav = speed_tune(wav)
			elif self.config.pitch:
				wav = pitch_tune(wav)
			
		spect = spectrogram(wav, self.config.mel_spec)
		if self.mode == "train":
			if self.config.specaug:
				spect = specaugment(spect)
		
		if self.mode in ["train", "dev"]:
			transcript = list(filter(None, [self.labels.get(x, 0) for x in transcript]))
		elif self.mode == "test":
			transcript = wav_path
		return spect, transcript

	def __len__(self):
		return len(self.idx)




class MASRDataLoader(DataLoader):
	def __init__(self, *args, **kwargs):
		super(MASRDataLoader, self).__init__(*args, **kwargs)
		self.collate_fn = self._collate_fn

	def _collate_fn(self, batch):
		def func(p):
			return p[0].size(1)

		batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
		longest_sample = max(batch, key=func)[0]
		freq_size = longest_sample.size(0)
		minibatch_size = len(batch)
		max_seqlength = longest_sample.size(1)
		inputs = torch.zeros(minibatch_size, freq_size, max_seqlength)
		input_lens = torch.IntTensor(minibatch_size)
		target_lens = torch.IntTensor(minibatch_size)
		targets = []
		for x in range(minibatch_size):
			sample = batch[x]
			tensor = sample[0]
			target = sample[1]
			seq_length = tensor.size(1)
			inputs[x].narrow(1, 0, seq_length).copy_(tensor)
			input_lens[x] = seq_length
			target_lens[x] = len(target)
			if self.dataset.mode == "test": 
				targets.append(target)
			else:
				targets.extend(target)
		if self.dataset.mode == "test":
			return inputs, targets, input_lens
		else:
			targets = torch.IntTensor(targets)
			return inputs, targets, input_lens, target_lens

