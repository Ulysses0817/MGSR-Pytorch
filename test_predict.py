import sys
import swifter
import pandas as pd
from beamdecode import predict

test_orig = pd.read_csv("/kaggle/input/speechrecon/test.csv")
test = test_orig.loc[test_orig.words==" "]
test["wav_path"] = test["Unnamed: 0"].swifter.apply(lambda x: "/kaggle/input/magicdata/test/test_byte(%s)/"%sys.argv[1]+"wave_%s.bin"%x)
test_orig.to_csv("./test_orig.csv", index = False)
test["pred"] = test.wav_path.swifter.apply(lambda x: predict(x))
test.to_csv("./test_res.csv", index = False)
test_orig.to_csv("./test_orig_res.csv", index = False)
