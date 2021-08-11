python inference.py \
-f <(ls input_specs/*.pt) \
-w waveglow_swara_model_544000_16kHz.pt \
-o output_wavs/ \
--is_fp16 \
-s 0.6 \
--sampling_rate 22050