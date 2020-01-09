import sys
from beamdecode import model_setup, predict

def getres(wp, m, d):
	res = predict(wp, m, d)
#     preds.append(res)
	with open("./Recoder_new.txt", "a+") as fw:
		fw.write(",".join([wp, res+"\n\r"]))
	return res
	
model, decoder = model_setup(sys.argv[2])#"./pretrained/model_6_33.6234_0.6837.pth"
with open(sys.argv[1], "r") as fr:
	paths = fr.readlines()
for p in paths:
	getres(p.strip(), model, decoder)
