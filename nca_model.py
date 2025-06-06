# nca_model.py
"""Neural Cellular Automata Model definition."""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np
from nca_globals import CHANNEL_N, DEFAULT_ENTROPY_ENABLED, DEFAULT_ENTROPY_STRENGTH
from nca_utils import get_living_mask

class CAModel(tf.keras.Model):
    """The core Cellular Automata Model."""
    def __init__(self, channel_n=CHANNEL_N, fire_rate=0.5,
                 enable_entropy=DEFAULT_ENTROPY_ENABLED,
                 entropy_strength=DEFAULT_ENTROPY_STRENGTH):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.enable_entropy = enable_entropy
        self.entropy_strength = entropy_strength

        self.dmodel = tf.keras.Sequential([
            Conv2D(128, 1, activation=tf.nn.relu),
            Conv2D(self.channel_n, 1, activation=None,
                   kernel_initializer=tf.zeros_initializer),
        ])
        # Build the model with a dummy call using expected CHANNEL_N
        self(tf.zeros([1, 3, 3, self.channel_n])) 

    @tf.function
    def perceive(self, x, angle=0.0):
        identify = np.float32([0, 1, 0]) 
        identify = np.outer(identify, identify)
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  
        dy = dx.T  
        
        c, s = tf.cos(angle), tf.sin(angle)
        kernel_dx_rot = c * dx - s * dy
        kernel_dy_rot = s * dx + c * dy
        
        kernel_stacked = tf.stack([identify, kernel_dx_rot, kernel_dy_rot], axis=-1) 
        kernel_reshaped = kernel_stacked[:, :, None, :] 
        # The input 'x' to perceive will have self.channel_n channels.
        # So, final_kernel needs to be compatible.
        final_kernel = tf.repeat(kernel_reshaped, self.channel_n, axis=2) 
        
        y = tf.nn.depthwise_conv2d(x, final_kernel, strides=[1, 1, 1, 1], padding='SAME')
        return y

    @tf.function
    def call(self, x, fire_rate=None, angle=0.0, step_size=1.0,
             enable_entropy=None, entropy_strength=None):
        """Forward pass of the CA. Expects x to have self.channel_n channels."""
        # Input x should be [B, H, W, self.channel_n]

        current_enable_entropy = enable_entropy if enable_entropy is not None else self.enable_entropy
        current_entropy_strength = entropy_strength if entropy_strength is not None else self.entropy_strength

        if current_enable_entropy and current_entropy_strength > 0:
            # Add Gaussian noise to the input state
            noise = tf.random.normal(shape=tf.shape(x), stddev=current_entropy_strength)
            x += noise

        pre_life_mask = get_living_mask(x)

        y = self.perceive(x, angle) # perceive is designed for x with self.channel_n
        dx_update = self.dmodel(y) * step_size # dmodel outputs self.channel_n channels
        
        current_fire_rate = fire_rate if fire_rate is not None else self.fire_rate
        
        # Robust way to generate update_probabilities of shape [B, H, W, 1]
        # This uses the shape of the first channel slice of x.
        # tf.shape(x[..., 0:1]) gives [B, H, W, 1]
        update_probabilities = tf.random.uniform(shape=tf.shape(x[..., 0:1]))
                                               
        update_mask_bool = update_probabilities <= current_fire_rate # Shape [B, H, W, 1]
        
        # dx_update is [B,H,W,CHANNEL_N], tf.cast(update_mask_bool,...) is [B,H,W,1]
        # Broadcasting works here, scaling each of the CHANNEL_N channels by the same mask value.
        x += dx_update * tf.cast(update_mask_bool, tf.float32)

        post_life_mask = get_living_mask(x)
        life_mask_combined = pre_life_mask & post_life_mask 
        
        return x * tf.cast(life_mask_combined, tf.float32)