import librosa
import wave
import numpy as np
import torch

sample_rate = 16000
window_size = 0.02
window_stride = 0.01
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


def spectrogram(wav, normalize=True):
# librosa
	D = librosa.stft(
		wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
	)

	spec, phase = librosa.magphase(D)
	spec = np.log1p(spec)
	
# handled
	
	spec = torch.FloatTensor(spec)

	if normalize:
		spec = (spec - spec.mean()) / spec.std()

	return spec