# %% 
# import libraries
import librosa as lr
import plot_package as pp
import soundfile as sf
import os

# %% 
# set the root directory and select a directory
sample_root = '/Users/duyx/Code/Classify/samples/'
samples_dir = os.listdir(sample_root)

dir_selector = samples_dir[3] # select a directory/category
directory = os.path.join(sample_root, dir_selector)

samples = os.listdir(directory)
samples.sort()

# %%
output_folder = '/Users/duyx/Code/Classify/one_second_samples/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

audio_categories = []
for dir in samples_dir:
    items = os.listdir(os.path.join(sample_root, dir))
    audio, sr = lr.load(os.path.join(sample_root, dir, items[0]), sr=16000)

    duration = len(audio) / sr
    if duration > 1:
        audio = audio[:sr]  # Take the first second of audio
    
    audio_categories.append(audio)
    output_path = os.path.join(output_folder, items[0])
    sf.write(output_path, audio, sr)
    


# %%
pp.plot_category(directory, samples)

pp.plot_category2(samples_dir, sample_root)