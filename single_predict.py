import sys
from beamdecode import model_setup, predict

def getres(wp, m, d):
	res = predict(wp, m, d)
#     preds.append(res)
	with open("./Recoder_new.txt", "a+") as fw:
		fw.write(",".join([wp, res+"\n\r"]))
	return res
	
vocab = model.vocabulary

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

model, decoder = model_setup(sys.argv[2], vocabulary = vocab)#"./pretrained/model_6_33.6234_0.6837.pth"
print(model, decoder)
with open(sys.argv[1], "r") as fr:
	paths = fr.readlines()
for p in paths:
	getres(p.strip(), model, decoder)
