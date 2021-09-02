python dual_inference.py \
	-source=sources/ \
	-target=p270 \
	-output_dir=Converted_wavs \
	-speech_split_conf=SpeechSplit/run/models/Full_C_8_16_R_8_4_P_8_32/hparams.json \
	-ss_g=SpeechSplit/run/models/Full_C_8_16_R_8_4_P_8_32/50000-G.ckpt \
	-waveglow_model=Waveglow/models/waveglow_256channels_universal_v5.pt \
	-waveglow_conf=Waveglow/config.json \
	-sigma=1 \