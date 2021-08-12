from .TFhparams import HParams

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:
hparams = HParams(
    # Content Encoder (Encoder_7)
    freq = 8, # downsample factor (tune)
    dim_neck = 8, # BLSTM dim
    dim_enc = 512, # Conv dim (tune)

    # Rhythm encoder (Encoder_t)
    freq_2 = 8,
    dim_neck_2 = 1, 
    dim_enc_2 = 128, 

    # Pitch encoder (Encoder_6)
    freq_3 = 8,
    dim_neck_3 = 32,
    dim_enc_3 = 256, 

    out_channels = 10 * 3,
    layers = 24,
    stacks = 4,
    residual_channels = 512,
    gate_channels = 512,  # split into 2 groups internally for gated activation
    skip_out_channels = 256,
    cin_channels = 80,
    gin_channels = -1,  # i.e., speaker embedding dim
    weight_normalization = True,
    n_speakers = -1,
    dropout = 1 - 0.95,
    kernel_size = 3,
    upsample_conditional_features = True,
    upsample_scales = [4, 4, 4, 4],
    freq_axis_kernel_size = 3,
    legacy = True,
    
    
    dim_freq = 80,
    dim_spk_emb = 82,
    dim_f0 = 257,
    dim_dec = 512,
    len_raw = 128,
    chs_grp = 16,
    
    # interp
    min_len_seg = 19,
    max_len_seg = 32,
    min_len_seq = 64,
    max_len_seq = 128,
    max_len_pad = 192,
    
    # data loader
    root_dir = 'assets/spmel',
    feat_dir = 'assets/raptf0',
    batch_size = 128,
    mode = 'train',
    shuffle = True,
    num_workers = 0,
    samplier = 8,

    # Convenient model builder
    builder = "wavenet",

    hop_size = 256,
    window_length = 1024,
    sample_rate = 16000,
    log_scale_min = float(-32.23619130191664),
    
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in values]
    return 'Hyperparameters:\n' + '\n'.join(hp)
