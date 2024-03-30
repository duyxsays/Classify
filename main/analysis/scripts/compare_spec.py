# %%
# import libraries
import matplotlib.pyplot as plt
from tensorflow import keras
from matplotlib import cm
from kapre.composed import get_melspectrogram_layer, get_stft_magnitude_layer
import librosa

# %%
# set the style of the plots
viridis = cm.get_cmap('viridis', 12)
plt.set_cmap('viridis')

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
# load the audio file
wav, _ = librosa.load('/Users/duyx/Code/Classify/my_clean/Bassgrowl/growl_mono_0.wav', sr=16000, mono=True)
print(wav.shape)
# %%
# plot the audio signal
viridis = cm.get_cmap('inferno', 20)

fig, ax = plt.subplots(figsize=(11,4))
plt.plot(wav, color=viridis.colors[6,:3])
fig.patch.set_visible(False)
ax.axis('off')
plt.xticks([])
plt.yticks([])
plt.show()

# %%
wav_batch = wav.reshape(1,1,-1)

mel_model = keras.Sequential()
#mel_model.add(Melspectrogram(sr=16000, n_mels=128, 
#          n_dft=512, n_hop=160, input_shape=(1,wav.shape[0]),
#          return_decibel_melgram=True,
#          trainable_kernel=False, name='melgram'))
mel_model = get_melspectrogram_layer(input_shape=(1, 16000), n_fft=512, hop_length=160, return_decibel=True,
                n_mels=128, input_data_format='channels_first', output_data_format='channels_last')

spec_model = keras.Sequential()
#spec_model.add(Spectrogram(n_dft=512,
#                           return_decibel_spectrogram=True))
spec_model.add(get_stft_magnitude_layer(input_shape=(1, 16000), n_fft=512, hop_length=160, return_decibel=True,
                input_data_format='channels_first', output_data_format='channels_last'))

Y = mel_model.predict(x=wav_batch).squeeze()
Y_spec = spec_model.predict(x=wav_batch).squeeze()

# %%
# transpose to have time on x-axis
Y = Y.transpose()
# flip the signal to have low frequencies at the bottom
Y = Y[::-1]

fig, ax = plt.subplots(figsize=(10,4))
plt.imshow(Y)
plt.set_cmap('inferno')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

# %%
Y_spec = Y_spec.transpose()
Y_spec = Y_spec[::-1]

fig, ax = plt.subplots(figsize=(14,4.80))
plt.set_cmap('inferno')
ax.set_xticks([])
ax.set_yticks([])
plt.imshow(Y_spec)

# %%
