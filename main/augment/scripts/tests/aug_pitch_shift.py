# %% This script is used to augment the audio samples in the dataset by pitch shifting them.
# Import packages
import librosa as lr
import soundfile as sf
import os

# %% 
# read sample directories
root = '/Users/duyx/Code/Classify/inverted_samples/'
inverted_directory = os.listdir('/Users/duyx/Code/Classify/inverted_samples/')

# %% 
# create directory
output_folder = '/Users/duyx/Code/Classify/pitch_shift_samples/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# %% 
# pitch shift the samples
semitones = [-6, -5, -4, -3, -2, -1, 1, 2, 3]
for sample in inverted_directory:
    signal, sr = lr.load(os.path.join(root, sample), sr=16000)

    for semitone in semitones:
        augmented_audio = lr.effects.pitch_shift(y=signal, sr=sr, n_steps=semitone)
        output_path = os.path.join(output_folder, 'pitch_shift_' + str(semitone) + '_' + sample)
        sf.write(output_path, augmented_audio, sr)

# %%
