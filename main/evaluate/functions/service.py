import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
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
    sns.heatmap(cm, 
                cmap='Blues', 
                cbar=False, 
                annot=True, fmt='d', 
                xticklabels=np.unique(true_labels), 
                yticklabels=np.unique(true_labels)
                )
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('./confusion_matrix.png')
    plt.show()

def create_report(true_labels, predicted_labels):
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    final_report = pd.DataFrame(report).transpose()
    print(final_report)


def create_roc_curve(binary_labels, predicted_probabilities):
    labels = np.array(binary_labels)
    probs = np.array(predicted_probabilities)

    # calculate FPR and TPR
    fpr, tpr, thresholds = roc_curve(labels, probs)

    # calculate AUC
    roc_auc = auc(fpr, tpr)

    print("FPR:", fpr)
    print("TPR:", tpr)
    print("AUC-ROC:", roc_auc)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('./roc_curve.png')
    plt.show()
