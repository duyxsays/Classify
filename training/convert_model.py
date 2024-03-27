import coreml
import torch
from transformers import AutoModelForAudioClassification

torch_model = AutoModelForAudioClassification.from_pretrained("TheDuyx/distilhubert-finetuned-bass-test")