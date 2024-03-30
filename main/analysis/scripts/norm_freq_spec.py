# %%
# import libraries
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from kapre.composed import get_melspectrogram_layer, get_stft_magnitude_layer
import librosa

# %%
def visualise_model(X):
    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_title('Normalized Frequency Spectrogram', size=20)
    ax.imshow(X)
    ax.set_ylabel('Mel bins', size=18)
    ax.set_xlabel('Time (10 ms)', size=18)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.show()

# %%
src, sr = librosa.load('/Users/duyx/Code/Classify/my_clean/Bassgrowl/growl_mono_0.wav', sr=None, mono=True)
print('Audio length: %d samples, %04.2f seconds. \n' % (len(src), len(src) / sr) +
      'Audio sample rate: %d Hz' % sr)
dt = 1.0
_src = src[:int(sr*dt)]
src = np.expand_dims(_src, axis=1)
input_shape = src.shape
print(input_shape)

# %%
melgram = get_melspectrogram_layer(input_shape=input_shape,
                                   n_mels=128,
                                   mel_norm='slaney',
                                   pad_end=True,
                                   n_fft=512,
                                   win_length=400,
                                   hop_length=160,
                                   sample_rate=sr,
                                   db_ref_value=1.0,
                                   return_decibel=True,
                                   input_data_format='channels_last',
                                   output_data_format='channels_last')

norm = keras.layers.LayerNormalization(axis=2)
melgram.shape = (16000, 1)
model = keras.Sequential()
model.add(melgram)
model.add(norm)
model.summary()
# %%
batch = np.expand_dims(src, axis=0)
X = model.predict(batch).squeeze().T
visualise_model(X)

plt.title('Normalized Frequency Histogram')
plt.hist(X.flatten(), bins='auto')
plt.show()

# %%
mel_model = keras.Sequential()
melgram = get_melspectrogram_layer(input_shape=(1, 16000), n_fft=512, hop_length=160, return_decibel=True,
                n_mels=128, input_data_format='channels_first', output_data_format='channels_last')
mel_model.add(melgram)

stft_model = keras.Sequential()
stft_mag = get_stft_magnitude_layer(input_shape=(1, 16000), n_fft=512, hop_length=160, return_decibel=True,
                input_data_format='channels_first', output_data_format='channels_last')
stft_model.add(stft_mag)