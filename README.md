## To Run Conversion
In terminal type:

	chmod +x run_dual_inference.sh 
	run_dual_inference.sh

Alternatively:

python dual_inference.py \
	-source=input_wavs/
	-target=p270 \
	-output_dir=Converted_wavs \
	-speech_split_conf=SpeechSplit/run/models/Full_C_8_16_R_8_4_P_8_32/hparams.json \
	-ss_g=SpeechSplit/run/models/Full_C_8_16_R_8_4_P_8_32/50000-G.ckpt \
	-waveglow_model=Waveglow/models/waveglow_256channels_universal_v5.pt \
	-waveglow_conf=Waveglow/config.json \
	-sigma=1 \


<!-- WaveGlow -->
### Dependencies 
* numpy
* torch==1.0
* tensorflow
* numpy==1.13.3
* inflect==0.2.5
* librosa==0.6.0
* scipy==1.0.0
* tensorflow==2.5.0 
* tensorboardX==1.1
* Unidecode==1.0.22
* tqdm
* pillow

# Global

| File | Description |
|----------------|----------------|
| dual_inference.py | Main script to perform audio-to-audio conversion. <li>Obtains paths to convert and creates target dir.</li><li>Loads configuration files.</li><li>Instantiates SpeechSplit and WaveGlow inference models.</li><li>Performs conversion on each source wav, using the style of the target.</li><li>Saves converted wavs to output directory. </li> |
| run_dual_inference.sh | Shell script to call dual_inference.py. |



# SpeechSplit


| File | Description |
|----------------|----------------|
| inference.py | Contains inferencer object which runs the forward pass through SpeechSplit. <li> Loads the saved models</li><li> Generates spectrograms with WaveGlow's Mel2Samp object</li><li> Creates embedding of the target voice</li><li> Performs conversion, returning converted mel-spectrogram</li> |
| audio.py | Utilities for processing audio. |
| audioRead.py | Contains function for generating speaker IDs. |
| data_loader.py | Creates batches of training data for training SpeechSplit. |
| hparams.json | Hyperparameters output with training run. |
| hparams.py | Hyperparameters to train next training run with. |
| LICENSE | MIT License provided by creators of SpeechSplit. |
| main.py | Generates data loader and training objects to run SpeechSplit training loop. |
| make_metadata.py | Separates training data into train and validation .pkl files. |
| make_spect_f0.py | Generates mel-specs using WaveGlow's Mel2Samp object, extracts F0 features from audio, and saves each as .npy format for each training instance. |
| model.py | Contains SpeechSplit models. |
| solver.py | The training loop for SpeechSplit. |
| TFhparams.py | Creates hyperparameter object. |
| utils.py | Utilities for processing mel-specs and F0 features. |


# WaveGlow

### Pre-trained WaveGlow vocoder:

| File | Description |
|----------------|----------------|
| inference.py | WaveGlow inferencer object, receiving mels from SpeechSplit. <li>Denoises spectrogram</li><li>Runs inference</li><li>Converts to audio</li><li>Saves .wav</li> |
| mel2samp.py | Creates mel-spectrograms using Tacotron-2's STFT. This object is used to create training data, as well as extracting mel-specs during inference.|
| glow.py | Contains WaveGlow model, with loss function, and WaveNet affine coupling layers. |
| denoiser.py | Denoiser model for removing bias of WaveGlow from generated spectrogram |
| config.json | Configuration file of WaveGlow, specifying training, data and hyperparameter configs. |
| make_train_files.py | Creates two text files containing paths to wav files, split into training and test sets. |
| train.py | Training loop for WaveGlow. |
| distributed.py | Training loop for distributed training of WaveGlow across multiple GPUs.| 
| models/ | Directory containing pre-trained WaveGlow model. Pre-trained vocoder can be found at: https://drive.google.com/file/d/1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF/view|
| tacotron2/ | Directory containing functions and objects from Tacotron-2 for mel-spectrogram generation. |
| LICENSE | BSD 3-Clause License. Copyright statement from Nvidia. |