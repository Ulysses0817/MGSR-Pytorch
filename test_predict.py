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
	
	test_dataset = data.MASRDataset(args.test_index_path, args.labels_path, mode="test")
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
			results.extend(out_strings[0])
			targets.extend(y)
	# try:
		# np.save("./probs.npy", probs)
	# except Exception as e:
		# print(e)
	return results, targets
	
if __name__ == "__main__":
	import argparse, ast
	parser = argparse.ArgumentParser()
	parser.description='Hi guys, let\'s test!'
	parser.add_argument("-b","--batch_size", type=int, default=32)
	parser.add_argument("-ptp","--pretrained_path", default=None)
	parser.add_argument("-te","--test_index_path", default="./dataset/test.json")
	parser.add_argument("-l","--labels_path", default="./dataset/labels.json")
	parser.add_argument("-cpu","--num_workers", default=8, type=int)
	
	args = parser.parse_args()
	# MultiGpu
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if device.type == "cpu":
		print("cpu is available...")
	else:
		print("cuda is available...")
		
	model, dataloader = model_setup(args)
	model.to(device)
	
	results, targets = test(model, dataloader, device)
	with open("./greedy_results.json", "w") as fw:
		json.dump([results, targets], fw)