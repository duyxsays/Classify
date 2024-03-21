# %% 
# libaries
from transformers import AutoModelForAudioClassification, pipeline
import torchaudio
import numpy as np
# %%
model = AutoModelForAudioClassification.from_pretrained("TheDuyx/distilhubert-finetuned-bass-test")
# %%
model
# %%
pipe = pipeline("audio-classification", model="TheDuyx/distilhubert-finetuned-bass-test")

# %%
pipe

# %%

# Load the audio sample
waveform, sample_rate = torchaudio.load("/Users/duyx/Code/Classify/moved/808/808_1.wav")

# Resample the audio to 16000 Hz if necessary
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)

# %%
waveform
# %%
waveform_np = np.array(waveform[0])
# %%
pipe(waveform_np)
# %%
