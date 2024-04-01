import numpy as np
import evaluate
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoModelForAudioClassification, AutoFeatureExtractor

dataset = load_dataset("TheDuyx/augmented_bass_data")

model_id = "ntu-spml/distilhubert"
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True, return_attention_mask=True
)

id2label_fn = dataset["train"].features["label"].int2str

id2label = {
    str(i): id2label_fn(i)
    for i in range(len(dataset["train"].features["label"].names))
}

label2id = {v: k for k, v in id2label.items()}

id2label["0"]

num_labels = len(id2label)

model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)

pre_name="bass-classifier3"

model_name = model_id.split("/")[-1]
batch_size = 8
gradient_accumulation_steps = 1
num_train_epochs = 10 # usually sat to 10

training_args = TrainingArguments(
    f"{model_name}-{pre_name}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=False,
    push_to_hub=True,
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

trainer = Trainer(
    model,
    training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

kwargs = {
    "dataset_tags": "TheDuyx/augmented_bass_data",
    "dataset": "bass_design_encoded",
    "model_name": f"{model_name}-{pre_name}",
    "finetuned_from": model_id,
    "tasks": "audio-classification",
}

trainer.push_to_hub(**kwargs)
