# import _init_path
import torch
import feature
from models.conv import GatedConv
import torch.nn.functional as F
from ctcdecode import CTCBeamDecoder

from multiprocessing import cpu_count
cpu_num = cpu_count()
print("CPU的核数为：{}".format(cpu_count()))

def model_setup(pretrained_path = "pretrained/gated-conv.pth",
				alpha = 0.8,
				beta = 0.3,
				lm_path = "/kaggle/working/lm/zh_giga.no_cna_cmn.prune01244.klm",
				cutoff_top_n = 40,
				cutoff_prob = 1.0,
				beam_width = 32,
				num_processes = 4,
				blank_index = 0,
				vocabulary = None):
	num_processes = cpu_num
	model = GatedConv.load(pretrained_path)
	model.eval()
	
	if vocabulary is not None:
		model.vocabulary = vocabulary
	
	decoder = CTCBeamDecoder(
		model.vocabulary,
		lm_path,
		alpha,
		beta,
		cutoff_top_n,
		cutoff_prob,
		beam_width,
		num_processes,
		blank_index,
	)
	return model, decoder

def translate(vocab, out, out_len):
	return "".join([vocab[x] for x in out[0:out_len]])


def predict(f, model, decoder):
	wav = feature.load_audio(f)
	spec = feature.spectrogram(wav)
	spec.unsqueeze_(0)
	with torch.no_grad():
		y = model.cnn(spec)
		y = F.softmax(y, 1)
	y_len = torch.tensor([y.size(-1)])
	y = y.permute(0, 2, 1)  # B * T * V
	# print("decoding")
	out, score, offset, out_len = decoder.decode(y, y_len)
	return translate(model.vocabulary, out[0][0], out_len[0][0])
