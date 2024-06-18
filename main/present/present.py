# %%
# imports
import torchaudio
import librosa
import matplotlib.pyplot as plt
import numpy as np

# %%
# Load the .wav file
sample_808 = '/Users/duyx/code/Classify/data/train/version2.0/808/808_8.wav'
sample_growl = '/Users/duyx/code/Classify/data/train/version2.0/growl/growl_10.wav'
samples = [sample_808, sample_growl]

for i, file_path in enumerate(samples):
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Remove the last channel
    waveform = waveform[:-1, :]

    # Plot the waveform
    plt.figure(figsize=(20, 4))
    plt.plot(waveform.t().numpy(), color='grey')
    plt.title(f'Waveform of {file_path}')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()

    y = waveform.numpy()[0]  # Convert to NumPy array and select the first channel (assuming mono audio)
    sr = sample_rate
    
    # Apply FFT
    frequency_spectrum = np.fft.fft(y)
    
    # Get the magnitude of the frequency spectrum
    magnitude = np.abs(frequency_spectrum)
    
    # Compute the frequency axis
    frequencies = np.fft.fftfreq(len(magnitude), 1/sr)
    
    # Plot the frequency spectrum
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, magnitude, color='grey')
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    # plt.xlim(0, sr/2)  # Plot only the positive frequencies
    plt.xlim(0, 1000)
    plt.show()


# %%
