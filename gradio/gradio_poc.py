import numpy as np
import gradio as gr
from transformers import pipeline

pipe = pipeline("audio-classification", model="TheDuyx/distilhubert-finetuned-bass-test")

def classify_audio(audio):
    sr, data = audio

    modified_array = np.delete(data, 1, axis=1) # deleting the second column
    
    waveform_np = float_array.flatten() # flatten the array for mono signal

    float_array = np.float32(modified_array) # convert to float32
    
    result = pipe(waveform_np) # predicting the class

    return result

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
    cache_examples=True,
)

demo.launch()