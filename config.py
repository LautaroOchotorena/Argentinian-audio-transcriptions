folder_path_audio = './audio_16k'
######## Spectrogram ########
sr = 22050
frame_length = 256
frame_step = 160
fft_length = 384
mfcc=False
stft=True
max_time_len = 1 + (7*sr-frame_length)//frame_step # n_seconds_audio = 7,
                                            # almost 90% of the audio files have
                                            # at most 7 seconds
spectrogram_path = './spectrogram/'

######## Data augmentation times applied #######
times_augmentations = 1
# > 1 amplifies the volume
amplitude_change_factor_lower_bound = 0.5 # > 1 Amplifies the volume
amplitude_change_factor_upper_bound = 1.4

# Masking
frequency_mask_lower_bound = 30
frequency_mask_upper_bound = 40
time_mask_lower_bound = 10
time_mask_upper_bound = 20
# Times applying masks
num_masks_lower_bound = 1
num_masks_upper_bound = 3

# Time stretch
rate_lower_bound = 0.9
rate_upper_bound = 1.2

# Pitch change
semitones_lower_bound = -1
semitones_upper_bound = 2

######## Model ####### 
initial_epoch = 0 # checkpoints/epoch_{initial_epoch}.h5
batch_size = 7
rnn_layers=5
learning_rate=1e-4
dropout = 0.5
rnn_units=512
