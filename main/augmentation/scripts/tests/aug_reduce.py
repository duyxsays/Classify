# %% Reduce the samples to one second
# import libraries
import librosa as lr
import os


# %% 
# create directory
output_folder = '/Users/duyx/Code/Classify/reduced_samples/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)


# %% 
# set the root directory and select a directory
sample_root = '/Users/duyx/Code/Classify/samples/'
samples_dir = os.listdir(sample_root)

# Read wav files, reduce to one second samples, and save them in a folder
for dir in samples_dir:
    items = os.listdir(os.path.join(sample_root, dir))
    for item in items:
        audio, sr = lr.load(os.path.join(sample_root, dir, item), sr=16000)
        duration = len(audio) / sr
        if duration > 1:
            audio = audio[:sr]  # Take the first second of audio
        output_path = os.path.join(output_folder, item)
        lr.output.write_wav(output_path, audio, sr)