# %%
from transformers import pipeline
import torchaudio
import numpy as np
import os
import time

# %%
latencies = []
model = pipeline("audio-classification", model="TheDuyx/distilhubert-bass9")
eval_dir = "/Users/duyx/Code/Classify/data/evaluate/version3.0/808/"

# %% 
for sample in os.listdir(eval_dir):
        waveform, sample_rate = torchaudio.load(f"{eval_dir}/{sample}")
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        waveform_np = np.array(waveform[0]) # convert to numpy array and cut channel
        

        start_time = time.time()
        _ = model(waveform_np) # classify sample
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        latencies.append(latency)

# %%
# Calculate desired percentiles
p50 = np.percentile(latencies, 50)
p90 = np.percentile(latencies, 90)
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)

print(f"Median (p50) latency: {p50} ms")
print(f"90th percentile (p90) latency: {p90} ms")
print(f"95th percentile (p95) latency: {p95} ms")
print(f"99th percentile (p99) latency: {p99} ms")

# %%

latencyMeasures = np.array([38.27, 77.67, 80.42, 124.20])  # Example latencies in ms

# Function to simulate one request of 10 predictions
def simulate_request(latency_distribution, num_predictions=50):
    simulated_latencies = np.random.choice(latency_distribution, num_predictions)
    total_latency = np.sum(simulated_latencies)
    return total_latency

# Simulate a single request with 10 predictions
total_latency = simulate_request(latencies)
print(f"Total latency for 10 predictions: {total_latency} ms")
# %%

def simulate_multiple_requests(latency_distribution, num_requests=1000, num_predictions=100):
    total_latencies = []
    for _ in range(num_requests):
        simulated_latencies = np.random.choice(latency_distribution, num_predictions)
        total_latency = np.sum(simulated_latencies)
        total_latencies.append(total_latency)
    
    total_latencies = np.array(total_latencies)
    
    p50 = np.percentile(total_latencies, 50)
    p90 = np.percentile(total_latencies, 90)
    p95 = np.percentile(total_latencies, 95)
    p99 = np.percentile(total_latencies, 99)
    
    return p50, p90, p95, p99

# Simulate 1000 requests, each with 10 predictions
p50, p90, p95, p99 = simulate_multiple_requests(latencyMeasures)
print(f"Median (p50) latency for 10 predictions: {p50} ms")
print(f"90th percentile (p90) latency for 10 predictions: {p90} ms")
print(f"95th percentile (p95) latency for 10 predictions: {p95} ms")
print(f"99th percentile (p99) latency for 10 predictions: {p99} ms")
# %%
