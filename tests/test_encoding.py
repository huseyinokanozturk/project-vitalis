import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from src.encoding import Encoder

def test_sensor_eye():
    # Initialize Encoder: Maximum firing rate set to 200 Hz
    encoder = Encoder(min_val=0.0, max_val=1.0, max_freq=200.0, dt=1.0) # dt=1ms
    
    steps = 100
    
    # Simulated Sensor Data (Static)
    # Sensor 1: 0.2 (Dark / Far) - Low Intensity
    # Sensor 2: 0.8 (Bright / Near) - High Intensity
    sensor_data = np.array([0.2, 0.8]) 
    
    spike_train_1 = []
    spike_train_2 = []
    
    print("👀 Sensor Test Initiated...")
    for _ in range(steps):
        # Pass through Encoder
        # Returns binary spikes (1/0) instead of current
        spikes = encoder.encode(sensor_data)
        
        spike_train_1.append(spikes[0])
        spike_train_2.append(spikes[1])
        
    # Visualization
    plt.figure(figsize=(10, 4))
    
    # Sensor 1 (Weak Signal)
    plt.subplot(2, 1, 1)
    # Draw vertical lines only where spikes occur
    spike_times_1 = [i for i, x in enumerate(spike_train_1) if x > 0]
    plt.vlines(spike_times_1, 0, 1, color='blue')
    plt.title(f"Weak Signal (0.2) -> Sparse Firing ({len(spike_times_1)} Spikes)")
    plt.xlim(0, steps)
    plt.yticks([])
    
    # Sensor 2 (Strong Signal)
    plt.subplot(2, 1, 2)
    spike_times_2 = [i for i, x in enumerate(spike_train_2) if x > 0]
    plt.vlines(spike_times_2, 0, 1, color='red')
    plt.title(f"Strong Signal (0.8) -> Frequent Firing ({len(spike_times_2)} Spikes)")
    plt.xlim(0, steps)
    plt.yticks([])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_sensor_eye()