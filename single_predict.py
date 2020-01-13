import sys,json
import pandas as pd
from beamdecode import model_setup, predict

def getres(wp, m, d, save_path):
	res = predict(wp, m, d)
#     preds.append(res)
	if res in ["", " "]:
		wp = wp.replace("IOS", "Android")
		res = predict(wp, m, d)
	if res in ["", " "]:
		wp = wp.replace("Android", "Recorder")
		res = predict(wp, m, d)
	with open(save_path, "a+") as fw:
		fw.write(",".join([wp, res+"\r\n"]))
	return res

if __name__ == "__main__":
	import argparse, ast
	parser = argparse.ArgumentParser()
	parser.description='Hi guys, let\'s test!'
	parser.add_argument("-b","--batch_size", type=int, default=32)
	parser.add_argument("-ptp","--pretrained_path", default=None)
	parser.add_argument("-wp","--wav_path", default="./dataset/test.json")
	parser.add_argument("-lp","--labels_path", default="./dataset/labels.json")
	parser.add_argument("-sp","--save_path", default="./dataset/labels.json")
	parser.add_argument("-m","--mode", default="test")
	
	args = parser.parse_args()

	with open(args.labels_path, "r") as f:
		vocab = json.load(f)

	gt1 = []
	for i in vocab:
		if len(i) >1:
			gt1.append(i)
	import codecs
	start,end = (0x4E00, 0x9FA5)
	hanzis = []
	for codepoint in range(int(start),int(end)):
		if chr(codepoint) not in vocab:
			hanzis.append(chr(codepoint))
	hanzis = hanzis[:len(gt1)]
	en2chr = dict(zip(gt1, hanzis))
	vocab = [en2chr[i] if i in gt1 else i for i in vocab]

	model, decoder = model_setup(args.pretrained_path, vocabulary = vocab)#"./pretrained/model_6_33.6234_0.6837.pth"
	# print(model, decoder)
	with open(args.wav_path, "r") as fr:
		paths = fr.readlines()
	try:
		existp = pd.read_csv(args.save_path, delimiter=",", header=None)[0].values
	except Exception as e:
		print(e)
		existp = []
	for p in paths:
		p = p.strip()
		if p in existp: continue
		res = getres(p, model, decoder, args.save_path)
		print(p, "\t", res)
