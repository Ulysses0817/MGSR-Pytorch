import sys
import json
import data
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from models.conv import GatedConv
from decoder import GreedyDecoder

def model_setup(args = None):
	
	test_dataset = data.MASRDataset(args.test_index_path, args.labels_path, args.mode)
	dataloader = data.MASRDataLoader(
			test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
		)
		
	model = GatedConv.load(args.pretrained_path)
	
	return model, dataloader
	
def test(model, dataloader, device):
	model.eval()
	decoder = GreedyDecoder(dataloader.dataset.labels_str)
	results, targets = [], []
	probs = []
	print("testing")
	with torch.no_grad():
		for i, (x, y, x_lens) in tqdm(enumerate(dataloader)):
			x = x.to(device)
			outs, out_lens = model(x, x_lens)
			# probs.append([outs.cpu().numpy(), out_lens.cpu().numpy()])
			outs = F.softmax(outs, 1)
			outs = outs.transpose(1, 2)
			out_strings, out_offsets = decoder.decode(outs, out_lens)
			results.extend(out_strings)
			targets.extend(y)
	# try:
		# np.save("./probs.npy", probs)
	# except Exception as e:
		# print(e)
	return results, targets

def eval(model, dataloader,device):
	model.eval()
	decoder = GreedyDecoder(dataloader.dataset.labels_str)
	from warpctc_pytorch import CTCLoss
	ctcloss = CTCLoss(size_average=True)
	cer = 0
	epoch_loss = 0
	print("decoding")
	with torch.no_grad():
		for i, (x, y, x_lens, y_lens) in tqdm(enumerate(dataloader)):
			x = x.to(device)
			outs, out_lens = model(x, x_lens)
			loss = ctcloss(outs.transpose(0, 1).transpose(0, 2), y, out_lens, y_lens)
			epoch_loss += loss.item()
			outs = F.softmax(outs, 1)
			outs = outs.transpose(1, 2)
			ys = []
			offset = 0
			for y_len in y_lens:
				ys.append(y[offset : offset + y_len])
				offset += y_len
			out_strings, out_offsets = decoder.decode(outs, out_lens)
			y_strings = decoder.convert_to_strings(ys)
			for pred, truth in zip(out_strings, y_strings):
				trans, ref = pred[0], truth[0]
				cer += decoder.cer(trans, ref) / float(len(ref))
		cer /= len(dataloader.dataset)
		epoch_loss /= i+1
	print("cer:{}, epoch_loss:{}".format(cer, epoch_loss))
	return cer, epoch_loss

if __name__ == "__main__":
	import argparse, ast
	parser = argparse.ArgumentParser()
	parser.description='Hi guys, let\'s test!'
	parser.add_argument("-b","--batch_size", type=int, default=32)
	parser.add_argument("-ptp","--pretrained_path", default=None)
	parser.add_argument("-te","--test_index_path", default="./dataset/test.json")
	parser.add_argument("-l","--labels_path", default="./dataset/labels.json")
	parser.add_argument("-cpu","--num_workers", default=8, type=int)
	parser.add_argument("-m","--mode", default="test")
	
	args = parser.parse_args()
	# MultiGpu
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if device.type == "cpu":
		print("cpu is available...")
	else:
		print("cuda is available...")
		
	model, dataloader = model_setup(args)
	model.to(device)
	
	if args.mode == "test":
		results, targets = test(model, dataloader, device)
	else:
		results, targets = eval(model, dataloader, device)
	with open("./greedy_results.json", "w") as fw:
		json.dump([results, targets], fw)