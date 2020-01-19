import torch, time
import torch.nn as nn
import data
from warpctc_pytorch import CTCLoss
from models.conv import GatedConv
from tqdm import tqdm
from decoder import GreedyDecoder
import numpy as np
import tensorboardX as tensorboard
import torch.nn.functional as F
import os, json, random
from lr_scheduler.Adamw import AdamW
from lr_scheduler.cyclic_scheduler import CyclicLRWithRestarts, ReduceMaxLROnRestart

#torch.multiprocessing.set_start_method('spawn', force=True)

# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
# warnings.filterwarnings('ignore')

# MultiGpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
	print("cpu is available...")
else:
	print("cuda is available...")
if torch.cuda.device_count() > 1:
	# config.batch_size = config.batch_size * torch.cuda.device_count()
	print("Let's use", torch.cuda.device_count(), "GPUs - ", config.batch_size)
	model = nn.DataParallel(model)#, device_ids=[0, 2]

def train(
	model,
	epochs=1000,
	batch_size=64,
	train_index_path="./dataset/train_index.json",
	dev_index_path="./dataset/dev_index.json",
	labels_path="./dataset/labels.json",
	learning_rate=0.6,
	momentum=0.8,
	max_grad_norm=0.2,
	weight_decay=0,
	config = None
):

	train_dataset = data.MASRDataset(train_index_path, labels_path, config = config)
	batchs = (len(train_dataset) + batch_size - 1) // batch_size
	train_dataloader = data.MASRDataLoader(
		train_dataset, batch_size=batch_size, num_workers=0
	)
	train_dataloader_shuffle = data.MASRDataLoader(
		train_dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
	)
	
	dev_datasets, dev_dataloaders = [], []
	for _item in ["IOS", "Android", "Recorder"]:
		dev_datasets.append(data.MASRDataset(dev_index_path, labels_path, mode = "dev", config = config, device_type = _item))
		dev_dataloaders.append(data.MASRDataLoader(dev_datasets[-1], batch_size=batch_size, num_workers=0, pin_memory=True))
	
	if config.optim == "sgd":
		print("choose sgd.")
		optimizer = torch.optim.SGD(
			model.parameters(),
			lr=learning_rate,
			momentum=momentum,
			nesterov=True,
			weight_decay=weight_decay,
		)
	else:
		print("choose adamwr.")
		optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
		if config.fp16:
			# Allow Amp to perform casts as required by the opt_level
			model, optimizer = amp.initialize(model, optimizer, opt_level=config.fp16, loss_scale=args.loss_scale)
			scheduler = CyclicLRWithRestarts(optimizer, config.batch_size, epoch_size=len(train_dataloader.dataset), restart_period=5, t_mult=1.2, 
										  eta_on_restart_cb=ReduceMaxLROnRestart(ratio=config.wr_ratio), policy="cosine")
		else:
			scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size=len(train_dataloader.dataset), restart_period=5, t_mult=1.2, 
										  eta_on_restart_cb=ReduceMaxLROnRestart(ratio=config.wr_ratio), policy="cosine")
	
	ctcloss = CTCLoss()#size_average=True
	decoder = GreedyDecoder(train_dataloader.dataset.labels_str)
	# lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.985)
	writer = tensorboard.SummaryWriter('./logs/')
	gstep = 0
	best_cer = 1
	for epoch in range(epochs):
		epoch_loss = 0
		cer_tr = 0
		if epoch > 0:
			train_dataloader = train_dataloader_shuffle
		# lr_sched.step()
		lr = get_lr(optimizer)
		writer.add_scalar("lr/epoch", lr, epoch)
		if config.optim == "adamwr": scheduler.step()
		for i, (x, y, x_lens, y_lens) in enumerate(train_dataloader):
			x = x.to(device)
			out, out_lens = model(x, x_lens)
			outs = out.transpose(0, 1).transpose(0, 2)
			loss = ctcloss(outs, y, out_lens, y_lens).to(device)
			loss = loss / x.size(0)
			optimizer.zero_grad()
			# 混合精度加速
			if config.fp16 is not None:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()

			nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) #if config.optim == "sgd": 
			optimizer.step()
			if config.optim == "adamwr": scheduler.batch_step()
			#loss_value = loss.item()
			
			# cer
			outs = F.softmax(out, 1)
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
				cer_tr += decoder.cer(trans, ref) / float(len(ref))
			# loss
			epoch_loss += loss.item()
			writer.add_scalar("loss/step", loss.item(), gstep)
			writer.add_scalar("cer_tr/step", cer_tr/(batch_size*(i+1)), gstep)
			gstep += 1
			# display
			if i % 5 == 0:
				print(
					"[{}/{}][{}/{}]\tLoss = {:.4f},\tCer = {:.4f}".format(
						epoch + 1, epochs, i, int(batchs), loss.item(), cer_tr/(batch_size*(i+1))
					),
					flush=True
				)
				
				if args.debug: break
				
		cer_tr /= len(train_dataloader.dataset)
		epoch_loss = epoch_loss / batchs
		
		cer_devs, loss_devs = [], []
		for dev_dataloader in dev_dataloaders:
			cer_dev, loss_dev = eval(model, dev_dataloader)
			cer_devs.append(cer_dev)
			loss_devs.append(loss_dev)
			
			if args.debug: 
				cer_devs.extend([0, 0])
				loss_devs.extend([0, 0])
				break
			
		cer_dev = sum(cer_devs)/3
		loss_dev  = sum(loss_devs)/3
		
		writer.add_scalar("loss/epoch", epoch_loss, epoch)
		# writer.add_scalar("loss_dev/epoch", loss_dev, epoch)
		writer.add_scalars("loss_dev/epoch", {"loss_dev":loss_dev, "loss_ios":loss_devs[0], "loss_recorder":loss_devs[2], "loss_android":loss_devs[1]}, epoch)
		writer.add_scalar("cer_tr/epoch", cer_tr, epoch)
		writer.add_scalars("cer_dev/epoch", {"cer_dev":cer_dev, "cer_ios":cer_devs[0], "cer_recorder":cer_devs[2], "cer_android":cer_devs[1]}, epoch)
		print("Epoch {}: Loss= {:.4f}, Loss_dev= {:.4f}, CER_tr = {:.4f}, CER_dev = {:.4f}".format(epoch, epoch_loss, loss_dev, cer_tr, cer_dev))
		if cer_dev <= best_cer:
			best_cer = cer_dev
			torch.save(model, "pretrained/model_bestCer_{}_{:.4f}_{}.pth".format(config.optim, epoch+cer_dev, time.strftime("%m%d%H%M", time.localtime())))
			with open("{}_{:.4f}_{:.4f}_{:.4f}.info".format(epoch, epoch_loss, cer_tr, cer_dev), "w") as _fw:
				_fw.write("")
				
		if args.debug: break

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group["lr"]


def eval(model, dataloader):
	model.eval()
	decoder = GreedyDecoder(dataloader.dataset.labels_str)
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
				#if len(ref) == 0 : print("ref:", ref, y_strings)
				cer += decoder.cer(trans, ref) / float(len(ref))
		cer /= len(dataloader.dataset)
		epoch_loss /= i+1
	model.train()
	return cer, epoch_loss
	
def load_pretrained(model, pretrained_path):
	package = torch.load(pretrained_path, map_location="cpu")
	try:
		pretrained_dict = package["state_dict"]
		package_output_units = len(package["config"]["vocabulary"])
	except:
		pretrained_dict = package.state_dict()
		package_output_units = len(package.vocabulary)
	if package_output_units != model.output_units:
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if ("cnn.10" not in k) and ("cnn.0" not in k)}
		print(pretrained_dict.keys())
	model_dict = model.state_dict()
	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)
	return model
	
if __name__ == "__main__":
	import argparse, ast
	parser = argparse.ArgumentParser()
	parser.description='Hi guys!'
	parser.add_argument("-e","--epochs", help="迭代次数",type=int, default=200)
	parser.add_argument("-b","--batch_size", type=int, default=32)
	parser.add_argument("-tr","--train_index_path", default="./dataset/train_index.json")
	parser.add_argument("-dev","--dev_index_path", default="./dataset/dev_index.json")
	parser.add_argument("-lp","--labels_path", default="./dataset/labels.json")
	parser.add_argument("-lr","--learning_rate", default=0.6, type=float)
	parser.add_argument("-mgn","--max_grad_norm", default=0.2, type=float)
	parser.add_argument("-mon","--momentum", default=0.8, type=float)
	parser.add_argument("-w","--weight_decay", default=0, type=float)	
	parser.add_argument("-wrr","--wr_ratio", default=0.66, type=float,)
	parser.add_argument("-opt","--optim", default="sgd")
	parser.add_argument("-fp16","--fp16", default=None, type=str, help="this parameter can be set as 'O1' or 'O2' ")
	parser.add_argument("-lscale","--loss_scale", default='dynamic', type=str)
	parser.add_argument("-speed","--speed", default=True, type=ast.literal_eval,)
	parser.add_argument("-pitch","--pitch", default=True, type=ast.literal_eval,)
	parser.add_argument("-specaug","--specaug", default=True, type=ast.literal_eval,)
	parser.add_argument("-mel","--mel_spec", default=True, type=ast.literal_eval,)
	parser.add_argument("-ptp","--pretrained_path", default=None)
	parser.add_argument("-debug","--debug", default=False, type=ast.literal_eval,)
	
	args = parser.parse_args()
	
	if args.fp16 is not None:
		from apex import amp
	
	
	
	with open(args.labels_path) as f:
		vocabulary = json.load(f)
		# vocabulary = "".join(vocabulary)
	model = GatedConv(vocabulary)
	if args.pretrained_path is not None:
		print("loading pretrained model: ", args.pretrained_path)
		model = load_pretrained(model, args.pretrained_path)
	
	model.to(device)
	
	train(model, 	
		epochs=args.epochs,
		batch_size=args.batch_size,
		train_index_path=args.train_index_path,
		dev_index_path=args.dev_index_path,
		labels_path=args.labels_path,
		learning_rate=float(args.learning_rate),
		momentum=args.momentum,
		max_grad_norm=args.max_grad_norm,
		weight_decay=args.weight_decay,
		config = args)
