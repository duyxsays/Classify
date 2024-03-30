# %%
# import libraries - class env
import numpy as np
from transformers import AutoFeatureExtractor
from datasets import load_dataset

# %%
# create my dataset using the audiofolder builder from datasets library
dataset = load_dataset("audiofolder", data_dir="/Users/duyx/Code/Classify/augmentation/data/augmented_bass_data")
# %%
# print the dictionary
dataset
# %%
# print the last element of the dictionary
dataset["train"][-1]
# %%
# make a train-test split of the dataset
dataset = dataset["train"].train_test_split(seed=42, shuffle=True, test_size=0.1)
# %%
# print the train dataset
dataset["train"][0]
# %%
# extract features from the pretrained model
model_id = "ntu-spml/distilhubert"
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True, return_attention_mask=True
)
# %%
# determine what the required sampling rate is for the training data
sampling_rate = feature_extractor.sampling_rate
sampling_rate
# %%
sample = dataset["train"][0]["audio"]
inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
print(f"Mean: {np.mean(sample['array']):.3}, Variance: {np.var(sample['array']):.3}")


print(f"inputs keys: {list(inputs.keys())}")

print(f"Mean: {np.mean(inputs['input_values']):.3}, Variance: {np.var(inputs['input_values']):.3}")
# %%
max_length = 1
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_length),
        truncation=True,
        return_attention_mask=True,
        )
    
    return inputs


# %%

dataset_encoded = dataset.map(
    preprocess_function,
    remove_columns=["audio"],
    batched=True,
    batch_size=100,
    num_proc=1,
    )

dataset_encoded
# %%
id2label_fn = dataset["train"].features["label"].int2str

id2label = {
    str(i): id2label_fn(i)
    for i in range(len(dataset_encoded["train"].features["label"].names))
}

label2id = {v: k for k, v in id2label.items()}

id2label["0"]


# %%
dataset_encoded.push_to_hub("TheDuyx/augmented_bass_data")


# %%
