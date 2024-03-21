import numpy as np
import gradio as gr
import torchaudio
import librosa
from transformers import pipeline

pipe = pipeline("audio-classification", model="TheDuyx/distilhubert-finetuned-bass-test")

def classify_audio(audio):
    
    sr, data = audio
    print(data)
    print(data.shape)
    modified_array = np.delete(data, 1, axis=1)
    
    float_array = np.float32(modified_array)
    print(float_array)
    
    #waveform_np = np.array(data)

    # Resample the audio to 16000 Hz if necessary
    # if sr != 16000:
    #    data = librosa.resample(data, sr, 16000)

    waveform_np = float_array.flatten()
    print(waveform_np)
    result = pipe(waveform_np)

    print(result)

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

#examples=[
#        "https://samplelib.com/lib/preview/mp3/sample-3s.mp3",
#        os.path.join(os.path.dirname(__file__), "audio/recording1.wav"),
#    ],