import numpy as np
import gradio as gr
from transformers import pipeline

pipe = pipeline("audio-classification", model="TheDuyx/distilhubert-bass-classifier")

def classify_audio(audio):
    sr, data = audio

    modified_array = np.delete(data, 1, axis=1) # deleting the second column
    
    waveform_np = modified_array.flatten() # flatten the array for mono signal

    waveform_np = np.float32(waveform_np) # convert to float32
    
    classification = pipe(waveform_np) # predicting the class

    result = closest_to_one(classification) # finding the closest class to 1

    formatted_result = f"Prediction: {result['label']} with a score of {result['score']}"

    return formatted_result

def closest_to_one(data_list):
    closest_item = None
    min_difference = float('inf')  # Initialize with a large value
    
    for item in data_list:
        difference = abs(item['score'] - 1)
        if difference < min_difference:
            min_difference = difference
            closest_item = item
    
    return closest_item

input_audio = gr.Audio(
    sources=["upload"],
    waveform_options=gr.WaveformOptions(
        waveform_color="#01C6FF",
        waveform_progress_color="#0066B4",
        skip_length=2,
        show_controls=False,
    ),
)

demo = gr.Interface(
    fn=classify_audio,
    inputs=input_audio,
    outputs="textbox",
    examples=[["brass.wav"], ["growl.wav"]],
    cache_examples=True,
)

demo.launch()