original_folder_path_audio = './data'
# I transform the 48khz wav files to 16khz
folder_path_audio = './data/audio_16k'

######## Spectrogram ########
sr = 16000
frame_length = int((11.6/1000) * sr)   # 11.6 ms per sample
frame_step = int((7.25/1000) * sr)     # 7.25 ms per sample
fft_length = int((1/57.4) * sr)        # 57.4 Hz per bin
mfcc=False
stft=True
max_time_len = int(1 + (8.7*sr-frame_length)//frame_step) # n_seconds_audio = 8.7
spectrogram_path = './spectrogram/'

######## Transcription ########
max_target_len = 110

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

# Add noise
noise_level_lower_bound = 0.01
noise_level_upper_bound = 0.03

####### Vocabulary #########
# The set of characters accepted in the transcription
characters = [x for x in "abcdefghijklmnopqrstuvwxyz?! ¿áéíúóñ¡"]

######## Model ####### 
run_id = None
default_initial_epoch = 0 # loads weights from checkpoints/ckpt-{initial_epoch}.data-00000-of-00001
                           # when initial_epoch > 0
batch_size = 16
num_hid=200
num_head=2
num_feed_forward=300
num_layers_enc=6
num_layers_dec=2
default_learning_rate=0.0009773430262285175
dropout_rate=0.3
l2_reg = 6e-4