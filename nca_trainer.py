# nca_trainer.py
"""NCA Training Logic."""

import tensorflow as tf
import numpy as np
import time
from nca_model import CAModel
from nca_utils import SamplePool, make_circle_masks 
from nca_globals import CHANNEL_N

class NCATrainer:
    def __init__(self, target_img_rgba, config):
        self.pad_target = tf.pad(target_img_rgba, 
                                 [(config['target_padding'], config['target_padding'])]*2 + [(0,0)])
        self.config = config
        self.ca = CAModel(channel_n=CHANNEL_N, fire_rate=config['fire_rate'])
        
        self.pool = self._initialize_pool() 
        self.loss_log = []
        self.current_step = 0
        self.training_start_time = None
        self.last_preview_state = None # Will hold a copy of a state from the batch
        self.last_loss = None

        lr = self.config.get('learning_rate', 2e-3)
        lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [2000], [lr, lr * 0.1] 
        )
        self.optimizer = tf.keras.optimizers.Adam(lr_sched)

    def _initialize_pool(self):
        h, w = self.pad_target.shape[:2]
        seed = np.zeros([h, w, CHANNEL_N], np.float32)
        seed[h // 2, w // 2, 3:] = 1.0  
        
        initial_states = np.repeat(seed[None, ...], self.config['pool_size'], axis=0)
        return SamplePool(x=initial_states) 

    def _loss_fn(self, x_state):
        return tf.reduce_mean(tf.square(x_state[..., :4] - self.pad_target), axis=[1, 2, 3])

    @tf.function
    def _train_step_tf(self, x_batch_tensor): 
        iter_n = tf.random.uniform([], 64, 96, dtype=tf.int32) 
        with tf.GradientTape() as tape:
            x_intermediate = x_batch_tensor
            for _ in tf.range(iter_n):
                x_intermediate = self.ca(x_intermediate, fire_rate=self.ca.fire_rate)
            loss_val = tf.reduce_mean(self._loss_fn(x_intermediate))
        
        trainable_vars = self.ca.trainable_weights
        grads = tape.gradient(loss_val, trainable_vars)
        
        valid_grads_and_vars = []
        for grad, var_w in zip(grads, trainable_vars):
            if grad is not None:
                norm_grad = grad / (tf.norm(grad) + 1e-8)
                valid_grads_and_vars.append((norm_grad, var_w))
        
        if valid_grads_and_vars:
            self.optimizer.apply_gradients(valid_grads_and_vars)
        return x_intermediate, loss_val


    def run_training_step(self): 
        if not self.pool or len(self.pool) == 0:
            # tf.print("Trainer: Pool is empty.")
            return None, None # Should not happen if initialized

        if self.training_start_time is None:
            self.training_start_time = time.time()

        num_to_sample = self.config['batch_size']
        sampled_indices = self.pool.get_indices(num_to_sample)

        if sampled_indices.size == 0 :
             # tf.print(f"Trainer: Could not sample {num_to_sample} items from pool.")
             return None, None 

        x0_from_pool_batch = self.pool.x[sampled_indices].copy() 
        
        damage_n_config = self.config.get('damage_n', 0)
        if damage_n_config > 0 and x0_from_pool_batch.shape[0] > 0: 
            h_target, w_target = self.pad_target.shape[:2]
            num_to_damage_actual = min(damage_n_config, x0_from_pool_batch.shape[0]) 
            
            if num_to_damage_actual > 0:
                # make_circle_masks returns [N, H, W], need [N, H, W, 1] for broadcasting
                damage_masks_np = make_circle_masks(num_to_damage_actual, h_target, w_target).numpy()[..., None]
                x0_from_pool_batch[-num_to_damage_actual:] *= (1.0 - damage_masks_np)

        if x0_from_pool_batch.shape[0] == 0: 
            # tf.print("Trainer: Batch is empty after damage.")
            return None, None

        x_updated_batch_tf, loss_tf = self._train_step_tf(tf.convert_to_tensor(x0_from_pool_batch))
        
        self.pool.x[sampled_indices] = x_updated_batch_tf.numpy()

        self.current_step += 1
        self.last_loss = loss_tf.numpy()
        self.loss_log.append(self.last_loss)
        # Store a copy for the preview, not a reference
        self.last_preview_state = x_updated_batch_tf[0].numpy().copy() 
        
        return self.last_preview_state, self.last_loss

    def get_status(self):
        elapsed_time_val = 0
        if self.training_start_time:
            elapsed_time_val = time.time() - self.training_start_time
        
        current_loss_val = self.last_loss if self.last_loss is not None else 0.0
        log_loss_val_str = "N/A"
        if current_loss_val > 1e-9: # Avoid log(0) or log(negative)
            try:
                log_loss_val_str = f"{np.log10(current_loss_val):.3f}"
            except Exception: 
                pass # Keep as N/A if math error

        return {
            "step": self.current_step,
            "loss": f"{current_loss_val:.4f}",
            "log_loss": log_loss_val_str,
            "training_time_seconds": elapsed_time_val,
            # "preview_state" is not sent here; client fetches /get_trainer_preview
        }

    def get_model(self):
        return self.ca

    def load_weights(self, path_to_weights):
        try:
            self.ca.load_weights(path_to_weights)
            # tf.print(f"Trainer: Weights loaded from {path_to_weights}")
        except Exception as e_detail:
            tf.print(f"Trainer: Error loading weights: {e_detail}")


    def save_weights(self, path_to_save):
        try:
            self.ca.save_weights(path_to_save)
            # tf.print(f"Trainer: Weights saved to {path_to_save}")
        except Exception as e_detail:
            tf.print(f"Trainer: Error saving weights: {e_detail}")