import os
import json
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
from show_mel import plot_data
from SpeechSplit.inference import SpeechSplitInferencer
from Waveglow.inference import WaveglowInferencer

def files_from_path(path):
	if os.path.isdir(path):
		wav_fpaths = list(Path(path).glob('**/*.wav'))
		speakers = list(map(lambda wav_fpath: wav_fpath.parent.stem, wav_fpaths))	
	else:
		wav_fpaths = [Path(path)]
		speakers = [Path(path).stem]
	return wav_fpaths, speakers

def run_conversion(args):

	wav_fpaths, speakers = files_from_path(args.source)
	target = args.target

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	

	with open(args.waveglow_conf) as f:
		print(f"Obtaining waveglow config from {args.waveglow_conf}")
		waveglow_conf = json.load(f)['data_config']

	speechsplit_inf = SpeechSplitInferencer(args, waveglow_conf)
	waveglow_inf = WaveglowInferencer(args)

	print(f"Reading audio from:\n{args.source}")

	for i, speaker in enumerate(np.unique(speakers)):
		utts = np.where(np.array(speakers) == speaker)

		if not os.path.exists(f"{args.output_dir}/{speaker}"):
			os.makedirs(f"{args.output_dir}/{speaker}")

		for source in np.array(wav_fpaths)[utts]:
			src_mel, src_f0_norm = speechsplit_inf.read_audio(source, target)
			print(f"\nPreparing {source.name} to convert to {target}")
			src_utt_pad, src_f0_onehot = speechsplit_inf.prep_data(src_mel, src_f0_norm)
			print(f"Running SpeechSplit")
			utt_pred = speechsplit_inf.forward(src_utt_pad, src_f0_onehot)
			print("Running Waveglow")
			out_path = Path(args.output_dir,speaker,source.stem+"_"+target+".wav")
			waveglow_inf.inference(utt_pred.T, out_path)


if __name__ == '__main__':
	parser = ArgumentParser()

	parser.add_argument('-source', '-s', help='source wav path')
	parser.add_argument('-target', '-t', help='target wav path')
	parser.add_argument('-output_dir', '-o', help='output dir path')

	# SpeechSplit
	parser.add_argument('-speech_split_conf', help='SpeechSplit config file path')
	parser.add_argument('-ss_g', help='SpeechSplit G model path')
	parser.add_argument('-sample_rate', '-sr', help='sample rate', default=22050, type=int)

	# Waveglow
	parser.add_argument('-waveglow_model','-w',  help='Path to waveglow decoder model')
	parser.add_argument('-waveglow_conf',  help='Path to waveglow config')
	parser.add_argument("-sigma", default=1.0, type=float)
	parser.add_argument("--is_fp16", action="store_true")
	parser.add_argument("-denoiser_strength","-d", default=0.0, type=float, 
				help='Removes model bias. Start with 0.1 and adjust')

	args = parser.parse_args()

	run_conversion(args)
