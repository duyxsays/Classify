# %% 
# libaries
from transformers import AutoModelForAudioClassification, pipeline
import torchaudio
import numpy as np
import os
# %%
# model = AutoModelForAudioClassification.from_pretrained("TheDuyx/distilhubert-finetuned-gtzan")
model = AutoModelForAudioClassification.from_pretrained("TheDuyx/distilhubert-finetuned-bass-audio")
# %%
model
# %%
# pipe = pipeline("audio-classification", model="TheDuyx/distilhubert-finetuned-gtzan")
pipe = pipeline("audio-classification", model="TheDuyx/distilhubert-finetuned-bass-audio")

# %%
pipe

# %%

# Load the audio sample
#waveform, sample_rate = torchaudio.load("/Users/duyx/Code/Classify/moved/808/808_1.wav")
waveform, sample_rate = torchaudio.load("/Users/duyx/Downloads/STCR2_MTA_Synth_Lead_One_Shot_Acid_A.wav")
waveform, sample_rate = torchaudio.load("/Users/duyx/Code/Classify/evaluate/")

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
def closest_to_one(data_list):
    closest_item = None
    min_difference = float('inf')  # Initialize with a large value
    
    for item in data_list:
        difference = abs(item['score'] - 1)
        if difference < min_difference:
            min_difference = difference
            closest_item = item
    
    return closest_item
# %%
correct_guesses = 0
for dir in os.listdir("/Users/duyx/Code/Classify/evaluate"):
    samples = os.listdir(f"/Users/duyx/Code/Classify/evaluate/{dir}")
    samples.sort()
    amount = len(samples)

    for sample in samples:
        waveform, sample_rate = torchaudio.load(f"/Users/duyx/Code/Classify/evaluate/{dir}/{sample}")
        waveform_np = np.array(waveform[0])
        classification = pipe(waveform_np)
        result = closest_to_one(classification)
        if result['label'] == f"{dir}":
            correct_guesses += 1

        print(f"{sample}: {result['label']}")

print(f"Correct guesses: {correct_guesses} out of {amount}")
# %%