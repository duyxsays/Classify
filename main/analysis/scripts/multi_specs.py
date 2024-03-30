# %%
# import libraries
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import numpy as np
from tensorflow import keras
from kapre.composed import get_melspectrogram_layer, get_stft_magnitude_layer

plt.style.use('ggplot')

# %%
# create STFT and Mel-spectrogram models from the kapre framework
stft_model = keras.Sequential()
stft_mag = get_stft_magnitude_layer(input_shape=(1, 16000), n_fft=512, hop_length=160, return_decibel=True,
                input_data_format='channels_first', output_data_format='channels_last')
stft_model.add(stft_mag)

mel_model = keras.Sequential()
melgram = get_melspectrogram_layer(input_shape=(1, 16000), n_fft=512, hop_length=160, return_decibel=True,
                n_mels=128, input_data_format='channels_first', output_data_format='channels_last')
mel_model.add(melgram)


# %% 
def plot_signals_time(titles, signals):
    
    nrows, ncols = 2, 5
    fig, ax = plt.subplots(nrows, ncols, figsize=(16, 6))
    
    z = 0
    for i in range(nrows):
        for y in range(ncols):
            ax[i,y].set_title(titles[z])
            ax[i,y].plot(signals[z])
            ax[i,y].set_xticks([])
            ax[i,y].set_yticks([])
            ax[i,y].grid(False)
            z += 1
    
    plt.show()

# %% 
# functions for plotting spectrograms used for both the stft and mel-spectrogram
def plot_spectrogram(titles, signals, title, shape=(16,8)):
    nrows, ncols = 2, 5
    fig, ax = plt.subplots(nrows, ncols, figsize=shape)
    fig.suptitle(title, size=20)
    plt.set_cmap('viridis')
    
    z = 0
    for i in range(nrows):
        for y in range(ncols):
            ax[i,y].set_title(titles[z])
            # squeeze to having 2 channels & transpose to have time on x-axis
            signal = signals[z].squeeze().transpose()
            # flip the signal to have low frequencies at the bottom
            signal = signal[::-1]

            ax[i,y].imshow(signal)
            ax[i,y].set_xticks([])
            ax[i,y].set_yticks([])
            ax[i,y].grid(False)
            z += 1
    
    plt.show()

# %%
# load audio files, retrieve their data and plot their waveforms
src_root = 'my_clean'

classes = os.listdir(src_root)
signals = []
titles = []
stft_specs = []
mel_specs = []

for _cls in sorted(classes):
    for fn in sorted(os.listdir(os.path.join(src_root, _cls))):
        rate, wav = wavfile.read(os.path.join(src_root, _cls, fn))
        signals.append(wav)
        titles.append(_cls)
        
        wav = wav.reshape(1, 1, -1)
        spec = stft_model.predict(wav)
        mel = mel_model.predict(wav)
        stft_specs.append(spec)
        mel_specs.append(mel)
        break

plot_signals_time(titles, signals)


# %%
# plot spectrograms
plot_spectrogram(titles, stft_specs, title= 'Decibel Spectrograms (257 x 100)',shape=(12,12))
# %%
# plot mel-spectrograms
plot_spectrogram(titles, mel_specs, title='Mel Spectrograms (128 x 100)', shape=(16,8))

# %%
# plot specified signal in time domain
rate, wav = wavfile.read(os.path.join(src_root, 'Bassgrowl', 'growl_mono_0.wav'))

plt.plot(wav)
plt.title('Bass Growl (time domain)', size=20)
plt.grid(False)
locs, labels = plt.xticks()
plt.xticks(locs, ['0.0', '0.1', '0.2', '0,3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
plt.ylabel('Amplitude (16 bits)', size=18)
plt.xlabel('Time (seconds)', size=18)
plt.show()

# %%
# plot specified signal in frequency domain
fft = np.fft.rfft(wav)
plt.plot(np.abs(fft)/np.sum(np.abs(fft)))
plt.title('Bass Growl (frequency domain)', size=20)
plt.grid(False)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel('Magnitude (norm)', size=18)
plt.xlabel('Frequency (hertz)', size=18)
plt.show()
# %%
