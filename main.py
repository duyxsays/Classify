# %% 
# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# %% 
import matplotlib.pyplot as plt
# Read the .wav file
filepath = 'samples/wave.wav'
sample_rate, data = wavfile.read(filepath)

# %%
# Compute the spectrogram
data = data[:, 0]
frequencies, times, spectrogram_data = spectrogram(data, sample_rate)

# %%
# Plot the spectrogram
plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data))
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power Spectral Density (dB)')
plt.title('Spectrogram')
plt.show()


# %%
