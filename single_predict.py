import sys
from beamdecode import model_setup, predict

def getres(wp, m, d):
	res = predict(wp, m, d)
#     preds.append(res)
	with open("./Recoder_new.txt", "a+") as fw:
		fw.write(",".join([wp, res+"\n"]))
	return res
	
model, decoder = model_setup("./pretrained/model_6_33.6234_0.6837.pth")#sys.argv[1]
getres(sys.argv[1], model, decoder)
