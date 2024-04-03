# %%
# import libraries
import librosa as lr
import soundfile as sf
import os

# %%
# choose an audio category
root = '/Users/duyx/Code/Classify/pitch_shift_samples/'
samples = os.listdir(root)


# %%
# create directory
output_folder = '/Users/duyx/Code/Classify/time_stretch_samples/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# %%
# time stretch the samples
rates = [0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1,15, 1.2]
for sample in samples:
    signal, sr = lr.load(os.path.join(root, sample), sr=16000) 
    
    for rate in rates:
        augmented_audio = lr.effects.time_stretch(y=signal, rate=rate)
        output_path = os.path.join(output_folder, 'time_stretch_' + str(rate) + '_' + sample)
        sf.write(output_path, augmented_audio, sr)
# %%
