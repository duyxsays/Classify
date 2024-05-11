# %%
# load libraries
from transformers import pipeline
from functions import service
import torchaudio
import numpy as np
import os

# %% 
# define the pipelines
# "TheDuyx/distilhubert-bass-classifier5"       87.68%
# "TheDuyx/distilhubert-bass-classifier7"       86.23%
# "TheDuyx/distilhubert-bass-classifier8"       86.96%
# "TheDuyx/distilhubert-bass"                   86.68%
# "TheDuyx/distilhubert-bass-classifier9",    # 89.13%
# "TheDuyx/distilhubert-bass3",               # 89.13%
# "TheDuyx/distilhubert-bass4",               # 89.13%
# "TheDuyx/distilhubert-bass6"                # 86.96%
# "TheDuyx/distilhubert-bass7"                # 87.68%

pipelines = [
    "TheDuyx/distilhubert-bass5",               # 89.86%
    "TheDuyx/distilhubert-bass9"                # 90.58%
    ]

# %%
# evaluate the pipelines
log_path = "/Users/duyx/Code/classify/main/evaluate/log/all_results.txt"
log = open(log_path, "w")
eval_dir = "/Users/duyx/Code/classify/data/evaluate/version3.0/"
service.delete_ds_store(eval_dir)


amount = service.total_samples(eval_dir)
service.delete_ds_store(eval_dir)

for _pipeline in pipelines:
    results_list = []
    correct_guesses = 0
    pipe = pipeline("audio-classification", model=_pipeline)

    for dir in os.listdir(eval_dir):
        samples = os.listdir(f"{eval_dir}{dir}")
        samples.sort()
        sub_amount = len(samples)
        sub_guesses = 0

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
            
            if result['label'] == f"{dir}":
                eval = "✅"
                correct_guesses += 1 # count amount of correct guesses
                sub_guesses += 1

        results_list.append(f"{_pipeline} - {dir} - {sub_guesses}/{sub_amount} ~ {round(sub_guesses / sub_amount * 100, 2)}%")


    print(f"### {_pipeline}")
    print(f"Correct guesses: {correct_guesses} out of {amount}\n")
    print(f"Accuracy: {round(correct_guesses / amount * 100, 2)}%\n\n")
    
    log.write(f"### {_pipeline}\n")
    log.write(f"Correct guesses: {correct_guesses} out of {amount}\n")
    log.write(f"Accuracy: {round(correct_guesses / amount * 100, 2)}%\n\n")
    

log.close()

# %%