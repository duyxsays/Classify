# %%
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# %%
sample_808 = '/Users/duyx/Code/Classify/data/train/version2.0/808/808_15.wav'
sample_growl = '/Users/duyx/Code/Classify/data/train/version2.0/growl/growl_9.wav'

# %%
# Load the audio file using torchaudio
audio_808, sr = torchaudio.load(sample_808)
audio_growl, sr = torchaudio.load(sample_growl)


# %%
# Plot the waveform side by side
fig, axs = plt.subplots(1, 2, figsize=(15, 4))

axs[0].plot(audio_808.t().numpy(), color='gray')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('808')

axs[1].plot(audio_growl.t().numpy(), color='gray')
axs[1].set_xlabel('Time')
axs[1].set_title('Growl')

plt.show()

# %%
def stereo_to_mono(data):
    # If the input is already mono, return it unchanged
    if len(data.shape) == 1:
        return data
    # If the input is stereo, convert it to mono by averaging the channels
    return np.mean(data, axis=1)

# Load the first audio file
sample_rate1, data1 = wavfile.read(sample_808)

# Load the second audio file
sample_rate2, data2 = wavfile.read(sample_growl)

# Convert stereo signals to mono
data1_mono = stereo_to_mono(data1)
data2_mono = stereo_to_mono(data2)

# Compute the Fast Fourier Transform (FFT) of the first audio data
fft_result1 = np.fft.fft(data1_mono)

# Compute the Fast Fourier Transform (FFT) of the second audio data
fft_result2 = np.fft.fft(data2_mono)

# Calculate the frequencies corresponding to the FFT result
frequencies = np.fft.fftfreq(len(data1_mono), 1/sample_rate1)

# Plot the frequency spectra side by side
plt.figure(figsize=(14, 6))

# Plot for Audio 1
plt.subplot(1, 2, 1)
plt.plot(frequencies[:len(data1_mono)//2], np.abs(fft_result1[:len(data1_mono)//2]), color='gray')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 2000)
plt.ylim(0, 10000000000000)
plt.title('808')

# Plot for Audio 2
plt.subplot(1, 2, 2)
plt.plot(frequencies[:len(data2_mono)//2], np.abs(fft_result2[:len(data2_mono)//2]), color='gray')
plt.xlabel('Frequency (Hz)')
plt.ylim(0, 1000000000000)
plt.title('Growl')

plt.tight_layout()
plt.show()

# %%

def plot_spectrogram(ax, data, sample_rate, title):
    ax.specgram(data, Fs=sample_rate, cmap='binary', aspect='auto')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot the spectrogram of the first audio sample
plot_spectrogram(ax1, data1_mono, sample_rate1, '808')

# Plot the spectrogram of the second audio sample
plot_spectrogram(ax2, data2_mono, sample_rate2, 'Growl')

plt.tight_layout()
plt.show()
# %%
