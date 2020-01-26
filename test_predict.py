#-*- coding: UTF-8 -*- 

import sys, os
import json
import torch
import data

import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from models.conv import GatedConv
from decoder import GreedyDecoder, BeamCTCDecoder

from multiprocessing import cpu_count
cpu_num = cpu_count()
print("CPU的核数为：{}".format(cpu_count()))

def model_setup(args = None):
	
	test_dataset = data.MASRDataset(args.test_index_path, args.labels_path, args.mode, config=args)
	dataloader = data.MASRDataLoader(
			test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
		)
		
	model = GatedConv.load(args.pretrained_path)
	
	global decoder
	decoder = BeamCTCDecoder(
								dataloader.dataset.labels_str,
								alpha = 0.8,
								beta = 0.3,
								lm_path = "/root/lm/zh_giga.no_cna_cmn.prune01244.klm",
								cutoff_top_n = 40,
								cutoff_prob = 1.0,
								beam_width = 100,
								num_processes = args.num_workers,
								blank_index = 0,
							)
							 
	return model, dataloader

def test(model, dataloader, device, lm_alpha, lm_beta):
	model.eval()
	# decoder = GreedyDecoder(dataloader.dataset.labels_str)
	
	global decoder
	decoder._decoder.reset_params(lm_alpha, lm_beta)
	
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
			results.extend([_v[0] for _v in out_strings])
			targets.extend(y)
	# try:
		# np.save("./probs.npy", probs)
	# except Exception as e:
		# print(e)
	return results, targets

def eval(model, dataloader, device, save_output = None, lm_alpha=None, lm_beta=None):
	model.eval()
	ae_decoder = GreedyDecoder(dataloader.dataset.labels_str)
	
	global decoder
	if lm_alpha is not None or lm_beta is not None:
		decoder._decoder.reset_params(lm_alpha, lm_beta)
	
	# from warpctc_pytorch import CTCLoss
	
	# ctcloss = CTCLoss(size_average=True)
	cer = 0
	epoch_loss = 0
	output_data = []
	print("decoding")
	with torch.no_grad():
		for i, (x, y, x_lens, y_lens) in tqdm(enumerate(dataloader)):
			x = x.to(device)
			outs, out_lens = model(x, x_lens)
			# loss = ctcloss(outs.transpose(0, 1).transpose(0, 2), y, out_lens, y_lens)
			# epoch_loss += loss.item()
			outs = F.softmax(outs, 1)
			outs = outs.transpose(1, 2)
			ys = []
			offset = 0
			for y_len in y_lens:
				ys.append(y[offset : offset + y_len])
				offset += y_len
			out_strings, out_offsets = decoder.decode(outs, out_lens)
			y_strings = ae_decoder.convert_to_strings(ys)
			
			if save_output is not None:
				# add output to data array, and continue
				output_data.append((outs.cpu().numpy(), out_lens.numpy(), y_strings))
			
			for pred, truth in zip(out_strings, y_strings):
				trans, ref = pred[0], truth[0]
				cer += decoder.cer(trans, ref) / float(len(ref))
		cer /= len(dataloader.dataset)
		epoch_loss /= i+1
	print("cer:{}, epoch_loss:{}".format(cer, epoch_loss))
	
	if save_output is not None:
		np.save(save_output, output_data)
		
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
	parser.add_argument("-mel","--mel_spec", default=False, type=ast.literal_eval,)
	parser.add_argument("-m","--mode", default="test")
	parser.add_argument('--save-output', default=None, help="Saves output of model from test to this file_path")
	
	args = parser.parse_args()
	# MultiGpu
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if device.type == "cpu":
		print("cpu is available...")
	else:
		print("cuda is available...")
		
	model, dataloader = model_setup(args)
	model.to(device)
	
	
	if "tg" in args.test_index_path:
		tune_param = "./tune_output_model32_tg085.json"
	elif "ts" in args.test_index_path:
		tune_param = "./tune_output_model32_ts085.json"
	else:
		tune_param = "./tune_output_model32_mycer128.json"
		
	with open(tune_param) as f:
		results = json.load(f)
	results = sorted(results, key=lambda x: x[3])[:1]
	params_grid = [tuple(_v[:2]) for _v in results]
	
	if args.mode == "test":
		datas = []
		for a, b in tqdm(params_grid):
			if "tg" in args.test_index_path:
				# a = 0.341
				# b = 0.857
				print("ok-tg")
			elif "ts" in args.test_index_path:
				a = 0.11363636363636363
				b = 0
			print(a, b)
			results, targets = test(model, dataloader, device, a, b)
			datas.append([results, targets])
		with open("./tune-beam_results_%s_%s.json"%(os.path.basename(args.pretrained_path), os.path.basename(args.test_index_path)), "w") as fw:
			json.dump(datas, fw)
	else:
		if "tg" in args.test_index_path:
			a = 0.3409090909090909#0.341
			b = 0.8571428571428571#0.857
		else:
			a = 0.11363636363636363#0.3
			b = 0
		print(a, b)
		# results, targets = eval(model, dataloader, device, args.save_output, a, b)
		
		if "tg" in args.test_index_path:
			args.test_index_path = args.test_index_path.replace("tg", "ts")
			# args.save_output = args.save_output.replace("tg", "ts")
			a = 0.05#0.11363636363636363
			b = 0
			print(args.test_index_path, args.save_output)
		else:
			args.test_index_path = args.test_index_path.replace("ts", "tg")
			# args.save_output = args.save_output.replace("ts", "tg")
			a = 0.05#0.11363636363636363
			b = 0
			print(args.test_index_path, args.save_output)
		print(a, b)
		test_dataset = data.MASRDataset(args.test_index_path, args.labels_path, args.mode, config=args)
		dataloader = data.MASRDataLoader(
				test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
			)
		results, targets = eval(model, dataloader, device, args.save_output, a, b)
