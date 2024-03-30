import numpy as np
import pandas as pd
import os

# function to find the closest item to 1
def closest_to_one(data_list):
    closest_item = None
    min_difference = float('inf')  # Initialize with a large value
    
    for item in data_list:
        difference = abs(item['score'] - 1)
        if difference < min_difference:
            min_difference = difference
            closest_item = item
    
    return closest_item

# function to delete .DS_Store files
def delete_ds_store(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == ".DS_Store":
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

def total_samples(directory):
    return sum([len(files) for r, d, files in os.walk(directory)])

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean

# Other models or pipelines
# model = AutoModelForAudioClassification.from_pretrained("TheDuyx/distilhubert-finetuned-gtzan")
# pipe = pipeline("audio-classification", model="TheDuyx/distilhubert-finetuned-gtzan")
# pipe = pipeline("audio-classification", model="TheDuyx/distilhubert-finetuned-bass-audio")