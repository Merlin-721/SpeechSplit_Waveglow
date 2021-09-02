import os
import json
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
from SpeechSplit.inference import SpeechSplitInferencer
from Waveglow.inference import WaveglowInferencer
import pickle
from distutils.dir_util import copy_tree

def files_from_path(path):
	wav_fpaths = list(Path(path).glob('**/*.wav'))
	speakers = list(map(lambda wav_fpath: wav_fpath.parent.stem, wav_fpaths))	
	return wav_fpaths, speakers

paths = [
	["man","/home/merlin/OneDrive/modules/individualProject/voiceChanger/Evaluation/Eval_Dataset_Reduced/Men"],
	["woman","/home/merlin/OneDrive/modules/individualProject/voiceChanger/Evaluation/Eval_Dataset_Reduced/Women"]
	]
target_parent_path = Path("/home/merlin/OneDrive/modules/individualProject/voiceChanger/Evaluation/multi_eval_dataset")
targets = [
	"p233", "p248", "p297", "p225", "p269", # F
	"p254", "p232", "p227", "p270", "p285"] # M

save_dir = "/home/merlin/OneDrive/modules/individualProject/voiceChanger/Evaluation/Resemblyzer/audio_data/SpeechSplit_Waveglow"

spk2gen = pickle.load(open('SpeechSplit/assets/spk2gen.pkl', "rb"))


def make_eval_set(args):
	with open(args.waveglow_conf) as f:
		print(f"Obtaining waveglow config from {args.waveglow_conf}")
		waveglow_conf = json.load(f)['data_config']

	speechsplit_inf = SpeechSplitInferencer(args, waveglow_conf)
	waveglow_inf = WaveglowInferencer(args)

	for path in paths:
		gender, data_dir = path

		wav_fpaths, speakers = files_from_path(data_dir)

		for target in targets:
			if spk2gen[target] == 'M':
				tar_gen = '_to_man'
			elif spk2gen[target] == 'F':
				tar_gen = '_to_woman'
			else:
				raise Exception("Speaker not in dataset")
			conv_type = gender + tar_gen

			# copy target data to evaluation set
			targ_out = str(Path(save_dir, conv_type, f"target_{target}", f"Target ({target})"))
			if not os.path.exists(targ_out):
				os.makedirs(targ_out)
			print(f"Copying target {target} to {targ_out}")
			copy_tree(str(Path(target_parent_path,target)), targ_out)


			for i, speaker in enumerate(np.unique(speakers)):
				utts = np.where(np.array(speakers) == speaker)

				out_dir = Path(save_dir, conv_type, f"target_{target}", speaker)

				if not os.path.exists(out_dir):
					os.makedirs(out_dir)

				for source in np.array(wav_fpaths)[utts]:
					src_mel, src_f0_norm = speechsplit_inf.read_audio(source, target)

					print(f"\nPreparing {source.name} to convert to {target}")
					src_utt_pad, src_f0_onehot = speechsplit_inf.prep_data(src_mel, src_f0_norm)

					utt_pred = speechsplit_inf.forward(src_utt_pad, src_f0_onehot)

					name = f"{source.stem}_{target}.wav"
					filepath = Path(out_dir, name)
					waveglow_inf.inference(utt_pred.T, filepath)


if __name__ == '__main__':
	parser = ArgumentParser()

	parser.add_argument('-speech_split_conf', help='SpeechSplit config file path')
	parser.add_argument('-ss_p', help='SpeechSplit P model path')
	parser.add_argument('-ss_g', help='SpeechSplit G model path')
	parser.add_argument('-sample_rate', '-sr', help='sample rate', default=22050, type=int)

	parser.add_argument('-waveglow_model','-w',  help='Path to waveglow decoder model')
	parser.add_argument('-waveglow_conf',  help='Path to waveglow config')
	parser.add_argument("-sigma", default=1.0, type=float)
	parser.add_argument("--is_fp16", action="store_true")
	parser.add_argument("-denoiser_strength","-d", default=0.0, type=float, 
				help='Removes model bias. Start with 0.1 and adjust')

	args = parser.parse_args()

	make_eval_set(args)