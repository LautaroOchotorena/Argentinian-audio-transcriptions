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

# Add noise
noise_level_lower_bound = 0.01
noise_level_upper_bound = 0.03

####### Vocabulary #########
# The set of characters accepted in the transcription
characters = [x for x in "abcdefghijklmnopqrstuvwxyz?! ¿áéíúóñ¡"]

######## Model ####### 
run_id = None
default_initial_epoch = 0 # loads weights from checkpoints/epoch_{initial_epoch}.h5 
                           # when initial_epoch > 0
batch_size = 7
rnn_layers=5
default_learning_rate=0.000196  
dropout = 0.5
rnn_units=448

######## Rescoring ####### 
# If using WSL2 on Windows it is better to have the kenlm folder on linux,
# not on windows
rescoring_path = "~/arg_audio_transcriptions/rescoring/kenlm"
rescoring = False