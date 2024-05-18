# %%
# load libraries
from datasets import load_dataset
from transformers import Wav2Vec2Processor

# %%
# Load dataset
dataset = load_dataset("TheDuyx/augmented_bass_sounds")

# %%
# Load processor
processor = Wav2Vec2Processor.from_pretrained("TheDuyx/distilhubert-bass9")

# %%
# Preprocessing function
def preprocess_function(examples):
    audio = examples["audio"]
    examples["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    examples["labels"] = processor.tokenizer(examples["label"]).input_ids
    return examples

# Apply preprocessing
train_dataset = dataset["train"].map(preprocess_function, remove_columns=dataset["train"].column_names)
test_dataset = dataset["test"].map(preprocess_function, remove_columns=dataset["test"].column_names)
