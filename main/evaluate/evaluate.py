# %% 
# libaries
from transformers import pipeline
from functions import service
import torchaudio
import numpy as np
import os

# %%
# load pipeline
pipe = pipeline("audio-classification", model="TheDuyx/distilhubert-bass-classifier5")

# %%                
correct_guesses = 0
eval_dir = "/Users/duyx/Code/Classify/data/evaluate/version2.0/"

service.delete_ds_store(eval_dir)

file_path = "/Users/duyx/Code/Classify/main/evaluate/log/results.txt"
amount = service.total_samples(eval_dir)
file = open(file_path, "w")
result_list = []
true_labels = []
predicted_labels = []

for dir in os.listdir(eval_dir):
    
    samples = os.listdir(f"{eval_dir}{dir}")
    samples.sort()
    sub_amount = len(samples)
    sub_guesses = 0
    
    file.write(f"\n### {dir}\n")
    
    for sample in samples:
        waveform, sample_rate = torchaudio.load(f"{eval_dir}{dir}/{sample}")
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        waveform_np = np.array(waveform[0]) # convert to numpy array and cut channel

        mask, sr = service.envelope(waveform_np, 16000, 0.1) # retrieve envelope masking
        waveform_np = waveform_np[mask] # apply envelope masking

        classification = pipe(waveform_np) # classify sample

        result = service.closest_to_one(classification) # format result to the closest to one
        eval = "❌"
        if result['label'] == f"{dir}":
            eval = "✅"
            correct_guesses += 1 # count amount of correct guesses
            sub_guesses += 1

        true_labels.append(dir)
        predicted_labels.append(result['label'])
        file.write(f"- {eval}: {dir} / {result['label']} - {sample}\n")

    result_list.append(f"{dir} - {sub_guesses}/{sub_amount} ~ {round(sub_guesses / sub_amount * 100, 2)}%")

file.close()
print(f"Correct guesses: {correct_guesses} out of {amount}")
print(f"Accuracy: {round(correct_guesses / amount * 100, 2)}%\n")
for i in result_list:
    print(i)


service.create_confusion_matrix(true_labels, predicted_labels)
 # %%
