import sys
import swifter
import pandas as pd
from beamdecode import model_setup, predict
from tqdm import tqdm

# def getres(wp, m, d):
	# res = predict(wp, m, d)
# #     preds.append(res)
	# with open("./Recoder_new.txt", "a+") as fw:
		# fw.write(res+"\n")
	# return res
# model, decoder = model_setup(sys.argv[2])#"./pretrained/model_6_33.6234_0.6837.pth"

test_orig = pd.read_csv("./dataset/test.csv")
test = test_orig.loc[test_orig.words==" "]
test["wav_path"] = test["Unnamed: 0"].swifter.apply(lambda x: "/kaggle/input/magicdata/test/test_byte(%s)/"%sys.argv[1]+"wave_%s.bin"%x)
# test.to_csv("./test_orig.csv", index = False)
# test["pred"] = test.wav_path.swifter.apply(lambda x: predict(x))
# if os.path.exists("./Recoder_new.txt"):
	# os.remove("./Recoder_new.txt")
# ress = []
for i in tqdm(test["wav_path"].values):
	os.system("python single_predict.py %s"%i)
# test["pred"] = ress
# test.to_csv("./test_res.csv", index = False)
# test_orig.to_csv("./test_orig_res.csv", index = False)