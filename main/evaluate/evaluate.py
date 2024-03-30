# %% 
# libaries
from transformers import pipeline
from functions import service
import torchaudio
import numpy as np
import os

# %%
# load pipeline
# model = AutoModelForAudioClassification.from_pretrained("TheDuyx/distilhubert-finetuned-gtzan")
# pipe = pipeline("audio-classification", model="TheDuyx/distilhubert-finetuned-gtzan")
# pipe = pipeline("audio-classification", model="TheDuyx/distilhubert-finetuned-bass-audio")
pipe = pipeline("audio-classification", model="TheDuyx/distilhubert-bass-classifier")

# %%                
correct_guesses = 0
eval_dir = "/Users/duyx/Code/Classify/data/evaluate/"

service.delete_ds_store(eval_dir)

file_path = "/Users/duyx/Code/Classify/main/evaluate/log/results.txt"
file = open(file_path, "w")

for dir in os.listdir(eval_dir):

    samples = os.listdir(f"{eval_dir}{dir}")
    samples.sort()
    amount = service.total_samples(eval_dir)
    
    file.write(f"\n### {dir}\n")

    for sample in samples:
        waveform, sample_rate = torchaudio.load(f"{eval_dir}{dir}/{sample}")
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        waveform_np = np.array(waveform[0])

        mask, sr = service.envelope(waveform_np, 16000, 0.1)
        waveform_np = waveform_np[mask]

        classification = pipe(waveform_np)

        result = service.closest_to_one(classification)
        eval = "❌"
        if result['label'] == f"{dir}":
            eval = "✅"
            correct_guesses += 1

        file.write(f"- {eval}: {dir} / {result['label']} - {sample}\n")

print(f"Correct guesses: {correct_guesses} out of {amount}")
print(f"Accuracy: {correct_guesses / amount * 100}%")
file.close()
# %%
