# %%
import librosa as lr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf

# %%
# Load the audio file
file_path = '/Users/duyx/Code/Classify/training_samples/slap/slap_1.wav'
# file_path = '/Users/duyx/Code/Classify/augmentation/augmented_test/slap/time_stretch_0.8_pitch_shift_-1_inverted_slap_6.wav'
#file_path = '/Users/duyx/Code/Classify/augmentation/augmented_test/slap/time_stretch_0.8_pitch_shift_-1_inverted_slap_23.wav'
y, sr = lr.load(file_path, sr=16000)
hardcoded_threshold = 0.01
frame_length = 2048

# %% 
plt.figure()
plt.plot(y)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Waveform')
plt.show()
# %%

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean

# %%
mask, env = envelope(y, sr, hardcoded_threshold)
# %%
trimmed = y[mask]

# %%
# plot the reshaped audio
plt.figure()
plt.plot(trimmed)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Reshaped Audio')
plt.show()
# %%
# Play the trimmed audio

mask2, env = envelope(trimmed, sr, hardcoded_threshold)

double_trimmed = trimmed[mask2]

# %%
plt.figure()
plt.plot(double_trimmed)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Reshaped Audio')
plt.show()
# %%
