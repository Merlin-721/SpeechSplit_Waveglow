import os
import json
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
	parser.add_argument('-ss_g', help='SpeechSplit G model path')
	parser.add_argument('-ss_p', help='SpeechSplit P model path')
	parser.add_argument('-sample_rate', '-sr', help='sample rate', default=16000, type=int)

	# Waveglow
	parser.add_argument('-waveglow_model','-w',  help='Path to waveglow decoder model')
	parser.add_argument('-waveglow_conf',  help='Path to waveglow config')
	parser.add_argument("-sigma", default=0.6, type=float)
	parser.add_argument("--is_fp16", action="store_true")
	parser.add_argument("-denoiser_strength","-d", default=0.0, type=float, 
				help='Removes model bias. Start with 0.1 and adjust')

	args = parser.parse_args()


	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	
	with open(args.waveglow_conf) as f:
		print(f"Obtaining waveglow config from {args.waveglow_conf}")
		waveglow_conf = json.load(f)['data_config']

	speechsplit_inf = SpeechSplitInferencer(args)
	waveglow_inf = WaveglowInferencer(args)

	print("\nRunning SpeechSplit")
	print(f"Reading audio from:\n{args.source} and \n{args.target}")
	src_mel, trg_mel, trg_f0_norm = speechsplit_inf.read_audio(args.source, args.target)
	# plot_data(src_mel,"source")
	# plot_data(trg_mel,"target")

	print(f"Preprocessing data")
	src_utt_pad, trg_utt_pad, trg_f0_onehot = speechsplit_inf.prep_data(src_mel, trg_mel, trg_f0_norm)
	# plot_data(src_utt_pad.cpu().numpy(),"src padded")
	# plot_data(trg_utt_pad.cpu().numpy(), "trg padded")

	print(f"Running inference")
	utt_pred = speechsplit_inf.forward(src_utt_pad, trg_utt_pad, trg_f0_onehot)
	# plot_data(utt_pred,"prediction")

	print("\nRunning Waveglow")
	g = args.ss_g.split('/')[-1].split('.')[0]
	p = args.ss_p.split('/')[-1].split('.')[0]
	name = f"{g}_{p}_"
	name += f"{args.source.split('/')[-1][:4]}_to_{args.target.split('/')[-1][:4]}_{args.output_name}"
	waveglow_inf.inference(utt_pred.T, name, plot=True)