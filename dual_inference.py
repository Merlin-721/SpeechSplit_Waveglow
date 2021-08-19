import os
import json
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
from show_mel import plot_data
from SpeechSplit.inference import SpeechSplitInferencer
from Waveglow.inference import WaveglowInferencer

if __name__ == '__main__':
	parser = ArgumentParser()

	parser.add_argument('-source', '-s', help='source wav path')
	parser.add_argument('-target', '-t', help='target wav path')
	parser.add_argument('-output_dir', '-o', help='output dir path')
	parser.add_argument('-output_name', help='name of output file')

	# SpeechSplit
	parser.add_argument('-speech_split_conf', help='SpeechSplit config file path')
	parser.add_argument('-ss_g', help='SpeechSplit G model path')
	parser.add_argument('-ss_p', help='SpeechSplit P model path')
	parser.add_argument('-sample_rate', '-sr', help='sample rate', default=22050, type=int)

	# Waveglow
	parser.add_argument('-waveglow_model','-w',  help='Path to waveglow decoder model')
	parser.add_argument('-waveglow_conf',  help='Path to waveglow config')
	parser.add_argument("-sigma", default=1.0, type=float)
	parser.add_argument("--is_fp16", action="store_true")
	parser.add_argument("-denoiser_strength","-d", default=0.0, type=float, 
				help='Removes model bias. Start with 0.1 and adjust')

	args = parser.parse_args()

	if os.path.isdir(args.target):
		raise Exception("Target is a directory. Please provide a single wav file")

	if os.path.isdir(args.source):
		wav_fpaths = list(Path(args.source).glob('**/*.wav'))
		speakers = list(map(lambda wav_fpath: wav_fpath.parent.stem, wav_fpaths))	
	else:
		wav_fpaths = [Path(args.source)]
		speakers = [Path(args.source).stem]
	target = Path(args.target)

	if not os.path.exists(target):
		raise Exception(f"Target file {target} does not exist")

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	

	with open(args.waveglow_conf) as f:
		print(f"Obtaining waveglow config from {args.waveglow_conf}")
		waveglow_conf = json.load(f)['data_config']

	speechsplit_inf = SpeechSplitInferencer(args, waveglow_conf)
	waveglow_inf = WaveglowInferencer(args)

	print(f"Reading audio from:\n{args.source} and \n{args.target}")

	for i, speaker in enumerate(np.unique(speakers)):
		utts = np.where(np.array(speakers) == speaker)

		if not os.path.exists(f"{args.output_dir}/{speaker}"):
			os.makedirs(f"{args.output_dir}/{speaker}")

		for source in np.array(wav_fpaths)[utts]:

			src_mel, trg_mel, trg_f0_norm = speechsplit_inf.read_audio(source, target)
			# plot_data(src_mel,"source")
			# plot_data(trg_mel,"target")
			print(f"\nPreparing {source.name} to convert to {target.name}")
			src_utt_pad, trg_utt_pad, trg_f0_onehot = speechsplit_inf.prep_data(src_mel, trg_mel, trg_f0_norm)
			# plot_data(src_utt_pad.cpu().numpy(),"src padded")
			# plot_data(trg_utt_pad.cpu().numpy(), "trg padded")
			print(f"Running SpeechSplit")
			utt_pred = speechsplit_inf.forward(src_utt_pad, trg_utt_pad, trg_f0_onehot)
			# plot_data(utt_pred,"prediction")
			# plot_data(mel)
			print("Running Waveglow")
			# name = f"{args.oneshot_model.split('/')[-1][-9:-5]}_sig_{args.sigma}_den_{args.denoiser_strength}_{args.output_name}"
			# g = args.ss_g.split('/')[-1].split('.')[0]
			# p = args.ss_p.split('/')[-1].split('.')[0]
			# name = f"{g}_{p}_"
			# name += f"{args.source.split('/')[-1][:4]}_to_{args.target.split('/')[-1][:4]}_{args.output_name}"
			name = f"{speaker}/{source.stem}_{target.stem}"
			waveglow_inf.inference(utt_pred.T, name)