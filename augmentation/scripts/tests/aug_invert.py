# %%
# import libraries
import librosa as lr
import os
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

# %% 
# Choose a audio category
samples_dir = os.listdir('/Users/duyx/Code/Classify/samples/')
dir_selector = samples_dir[4] # select a directory/category
print('Chosen category: ' + dir_selector)
directory = os.path.join('/Users/duyx/Code/Classify/samples/', dir_selector)

# %% 
# Load the samples in the directory
samples = os.listdir(directory)

# %% 
# create directory
output_folder = '/Users/duyx/Code/Classify/inverted_samples/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# %% 
# augment the samples
for sample in samples:
    signal, sr = lr.load(os.path.join(directory, sample), sr=16000)

    augmented_audio = signal * -1
    output_path = os.path.join(output_folder, 'inverted_' + sample)
    sf.write(output_path, augmented_audio, sr)

# %%
# test the inversion

original, sr = lr.load(os.path.join(directory, samples[0]), sr=16000)
inverted, sr = lr.load(os.path.join(output_folder, 'inverted_' + samples[0]), sr=16000)

# plot the original and augmented audio
plt.figure()
plt.plot(np.arange(0, len(original))/sr, original)
plt.title('Original audio')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.figure()
plt.plot(np.arange(0, len(inverted))/sr, inverted)
plt.title('Inverted audio')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# %%
