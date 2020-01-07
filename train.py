import torch
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


# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 3"
torch.backends.cudnn.benchmark = True
# warnings.filterwarnings('ignore')

# MultiGpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
	# config.batch_size = config.batch_size * torch.cuda.device_count()
	print("Let's use", torch.cuda.device_count(), "GPUs - ", config.batch_size)
	model = nn.DataParallel(model)#, device_ids=[0, 2]

def train(
    model,
    epochs=1000,
    batch_size=64,
    train_index_path="dataset/train_index.json",
    dev_index_path="dataset/dev_index.json",
    labels_path="dataset/labels.json",
    learning_rate=0.6,
    momentum=0.8,
    max_grad_norm=0.2,
    weight_decay=0,
):
    train_dataset = data.MASRDataset(train_index_path, labels_path)
    batchs = (len(train_dataset) + batch_size - 1) // batch_size
    dev_dataset = data.MASRDataset(dev_index_path, labels_path)
    train_dataloader = data.MASRDataLoader(
        train_dataset, batch_size=batch_size, num_workers=8
    )
    train_dataloader_shuffle = data.MASRDataLoader(
        train_dataset, batch_size=batch_size, num_workers=8, shuffle=True
    )
    dev_dataloader = data.MASRDataLoader(
        dev_dataset, batch_size=batch_size, num_workers=8
    )
    parameters = model.parameters()
    optimizer = torch.optim.SGD(
        parameters,
        lr=learning_rate,
        momentum=momentum,
        nesterov=True,
        weight_decay=weight_decay,
    )
    ctcloss = CTCLoss(size_average=True)
    # lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.985)
    writer = tensorboard.SummaryWriter()
    gstep = 0
    for epoch in range(epochs):
        epoch_loss = 0
        if epoch > 0:
            train_dataloader = train_dataloader_shuffle
        # lr_sched.step()
        lr = get_lr(optimizer)
        writer.add_scalar("lr/epoch", lr, epoch)
        for i, (x, y, x_lens, y_lens) in enumerate(train_dataloader):
            x = x.to(device)
            out, out_lens = model(x, x_lens)
            out = out.transpose(0, 1).transpose(0, 2)
            loss = ctcloss(out, y, out_lens, y_lens)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
            writer.add_scalar("loss/step", loss.item(), gstep)
            gstep += 1
            print(
                "[{}/{}][{}/{}]\tLoss = {}".format(
                    epoch + 1, epochs, i, int(batchs), loss.item()
                )
            )
        epoch_loss = epoch_loss / batchs
        cer = eval(model, dev_dataloader)
        writer.add_scalar("loss/epoch", epoch_loss, epoch)
        writer.add_scalar("cer/epoch", cer, epoch)
        print("Epoch {}: Loss= {}, CER = {}".format(epoch, epoch_loss, cer))
        torch.save(model, "pretrained/model_{}_{}_{}.pth".format(epoch, epoch_loss, cer))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def eval(model, dataloader):
    model.eval()
    decoder = GreedyDecoder(dataloader.dataset.labels_str)
    cer = 0
    print("decoding")
    with torch.no_grad():
        for i, (x, y, x_lens, y_lens) in tqdm(enumerate(dataloader)):
            x = x.to(device)
            outs, out_lens = model(x, x_lens)
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
    model.train()
    return cer


if __name__ == "__main__":
    with open("dataset/labels.json") as f:
        vocabulary = json.load(f)
        vocabulary = "".join(vocabulary)
    model = GatedConv(vocabulary)
    model.to(device)
    train(model)
