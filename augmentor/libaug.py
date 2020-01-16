# -*- coding: utf-8 -*-

import random
import numpy as np
import librosa
from augmentor.specaugment import time_warp, freq_mask, time_mask

def read_wav_data_from_librosa(filename, framerate=16000, speed=False):
	wave_data = librosa.load(filename, sr=framerate, mono=True)[0]
	if speed:
		if random.random() <= 0.5:
			speed_rate = random.random() * 0.2 + 0.9
			wave_data = speed_tune(wave_data, speed_rate)
	return [wave_data], framerate
	
def speed_tune(wav, speed_rate = None, prob = 0.5):
	# wav_speed_tune = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()
	if speed_rate is None:
		if random.random() <= prob:
			speed_rate = random.random() * 0.2 + 0.9
		else:
			return wav
	wav_speed_tune = librosa.effects.time_stretch(wav, speed_rate)
	return wav_speed_tune
	
def pitch_tune(wav, sr=16000, n_steps = None, prob = 0.5):
	if n_steps is None:
		if random.random() <= prob:
			n_steps = round(random.random() - 0.5, 2)
		else:
			return wav
	wav_pitch_tune = librosa.effects.pitch_shift(wav, sr, n_steps=n_steps)
	return wav_pitch_tune

def specaugment(spec, W=60, F=27, T=70, num_freq_masks=2, num_time_masks=2, p=0.2, replace_with_zero=False):
	'''SpecAugment
	Reference: SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
		(https://arxiv.org/pdf/1904.08779.pdf)
	This implementation modified from https://github.com/zcaceres/spec_augment
	:param torch.Tensor spec: input tensor with the shape (T, dim)
	:param int W: time warp parameter
	:param int F: maximum width of each freq mask
	:param int T: maximum width of each time mask
	:param int num_freq_masks: number of frequency masks
	:param int num_time_masks: number of time masks
	:param bool replace_with_zero: if True, masked parts will be filled with 0, if False, filled with mean
	'''
	if replace_with_zero:
		pad_value = 0
	else:
		pad_value = spec.mean()
	
	if random.random() <= 0.5 and spec.size(1) > 2*W:
		spec = time_warp(spec, W=W)
	if random.random() <= 0.65:
		spec = freq_mask(spec, F=F, num_masks=num_freq_masks, pad_value=pad_value)
	if random.random() <= 0.65:
		spec = time_mask(spec, T=T, num_masks=num_time_masks, p=p, pad_value=pad_value)
		
	return spec
	
# import librosa
# y,sr = librosa.load("/Users/birenjianmo/Desktop/learn/librosa/mp3/in.wav")
# # 通过移动音调变声 ，14是上移14个半步， 如果是 -14 下移14个半步
# b = librosa.effects.pitch_shift(y, sr, n_steps=14)
# y_fast = librosa.effects.time_stretch(y, 2.0)

# librosa.output.write_wav("pitch_shift.wav",b,sr)

