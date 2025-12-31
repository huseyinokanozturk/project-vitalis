import numpy as np

class Encoder:
    """
    Encoder class for encoding continuous values into spikes.
    Using: Poisson Coding
    """

    def __init__(self, min_val=0.0, max_val=1.0, max_freq = 100.0, dt = 0.1):
        """
        Initialize the encoder.
        
        Args:
            min_val (float): Minimum value of the input range.
            max_val (float): Maximum value of the input range.
            max_freq (float): Maximum frequency of the neuron's spiking rate.
            dt (float): Time step.
        """
        self.min_val = min_val
        self.max_val = max_val
        self.max_freq = max_freq
        self.dt = dt

    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode the input data into spikes.
        
        Args:
            data (np.ndarray): Input data to be encoded. Example: [0.1, 0.8, 0.5, ...]
        
        Returns:
            Encoded spikes: A boolean array of the same shape. Example: [0, 1, 0, ...]
        """

        # Normalize the data to the range [0,1]
        normalized_data = np.clip((data - self.min_val) / (self.max_val - self.min_val), 0.0, 1.0)

        # Calculate the rate of spiking
        firing_probs = (normalized_data * self.max_freq) * (self.dt / 1000.0)
        
        # Roll a dice to determine if a spike occurs
        spikes = np.random.rand(*data.shape) < firing_probs

        return spikes.astype(float)

    def step_current(self, data: np.ndarray, gain: float = 10.0) -> np.ndarray:
        """
        Alternative Method: For those who want to generate Current directly instead of Spikes.
        In our network architecture, 'external_inputs' are added as current.

        This function generates Poisson Spikes and converts them into current by multiplying them with a Gain factor.
        In this way, we obtain a 'Noisy' and 'Realistic' current.

        """   
        spikes = self.encode(data)
        currents = spikes * gain
        return currents 
        
