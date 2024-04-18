import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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


def create_confusion_matrix(true_labels, predicted_labels):
    
    labels = ["slap", "acid", "brass", "sub", "reese", "growl", "808"]
    
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, cmap='Blues', cbar=False, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('./confusion_matrix.png')
    plt.show()
    

    #sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))    
    