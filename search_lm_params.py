import argparse
import json
import sys
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
cpu_num = cpu_count()
print("CPU的核数为：{}".format(cpu_count()))

import numpy as np
import torch
from tqdm import tqdm

from decoder import BeamCTCDecoder, GreedyDecoder
from models.conv import GatedConv

def add_decoder_args(parser):
	beam_args = parser.add_argument_group("Beam Decode Options",
										  "Configurations options for the CTC Beam Search decoder")
	beam_args.add_argument('--top-paths', default=1, type=int, help='number of beams to return')
	beam_args.add_argument('--beam-width', default=10, type=int, help='Beam width to use')
	beam_args.add_argument('--lm-path', default=None, type=str,
						   help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
	beam_args.add_argument('--alpha', default=0.8, type=float, help='Language model weight')
	beam_args.add_argument('--beta', default=1, type=float, help='Language model word bonus (all words)')
	beam_args.add_argument('--cutoff-top-n', default=40, type=int,
						   help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
								'vocabulary will be used in beam search, default 40.')
	beam_args.add_argument('--cutoff-prob', default=1.0, type=float,
						   help='Cutoff probability in pruning,default 1.0, no pruning.')
	beam_args.add_argument('--lm-workers', default=64, type=int, help='Number of LM processes to use')
	return parser

parser = argparse.ArgumentParser(description='Tune an ARPA LM based on a pre-trained acoustic model output')
parser.add_argument('--model-path', default='models/deepspeech_final.pth',
					help='Path to model file created by training')
parser.add_argument('--saved-output', default="", type=str, help='Path to output from test.py')
parser.add_argument('--num-workers', default=16, type=int, help='Number of parallel decodes to run')
parser.add_argument('--output-path', default="tune_results.json", help="Where to save tuning results")
parser.add_argument('--lm-alpha-from', default=0.0, type=float, help='Language model weight start tuning')
parser.add_argument('--lm-alpha-to', default=3.0, type=float, help='Language model weight end tuning')
parser.add_argument('--lm-beta-from', default=0.0, type=float,
					help='Language model word bonus (all words) start tuning')
parser.add_argument('--lm-beta-to', default=0.5, type=float,
					help='Language model word bonus (all words) end tuning')
parser.add_argument('--lm-num-alphas', default=45, type=float, help='Number of alpha candidates for tuning')
parser.add_argument('--lm-num-betas', default=5, type=float, help='Number of beta candidates for tuning')
parser = add_decoder_args(parser)
args = parser.parse_args()

if args.lm_path is None:
	print("error: LM must be provided for tuning")
	sys.exit(1)

model = GatedConv.load(args.model_path)

saved_output = np.load(args.saved_output, allow_pickle=True)


def init(beam_width, blank_index, lm_path):
	global decoder, ae_decoder
	decoder = BeamCTCDecoder(model.vocabulary, lm_path=lm_path, beam_width=beam_width, num_processes=args.lm_workers,
							 blank_index=blank_index)
	ae_decoder = GreedyDecoder(model.vocabulary)


def decode_dataset(params):
	lm_alpha, lm_beta = params
	global decoder
	decoder._decoder.reset_params(lm_alpha, lm_beta)
	
	total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
	cer = 0.0
	wer = 0.0
	for out, sizes, target_strings in saved_output:
		out = torch.Tensor(out).float()
		sizes = torch.Tensor(sizes).int()
		decoded_output, _, = decoder.decode(out, sizes)
		for x in range(len(target_strings)):
			transcript, reference = decoded_output[x][0], target_strings[x][0]
			cer += decoder.cer(transcript, reference) / float(len(reference))
			
	cer /= 1758
	# epoch_loss /= i+1
	# wer = float(total_wer) / num_tokens
	# cer = float(total_cer) / num_chars

	return [lm_alpha, lm_beta, wer * 100, cer * 100]


if __name__ == '__main__':
	p = Pool(args.num_workers, init, [args.beam_width, model.vocabulary.index('_'), args.lm_path])

	cand_alphas = np.linspace(args.lm_alpha_from, args.lm_alpha_to, args.lm_num_alphas)
	cand_betas = np.linspace(args.lm_beta_from, args.lm_beta_to, args.lm_num_betas)
	params_grid = [(float(alpha), float(beta)) for alpha in cand_alphas
				   for beta in cand_betas]

	# with open("./tune_output_model32.json") as f:
		# results = json.load(f)
	# results = sorted(results, key=lambda x: x[3])[:100]
	# params_grid = [tuple(_v[:2]) for _v in results]

	scores = []
	for params in tqdm(p.imap(decode_dataset, params_grid), total=len(params_grid)):
		scores.append(list(params))
	print("Saving tuning results to: {}".format(args.output_path))
	with open(args.output_path, "w") as fh:
		json.dump(scores, fh)
