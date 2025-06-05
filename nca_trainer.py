# nca_trainer.py
"""NCA Training Logic."""

import os # Added for path operations
import tensorflow as tf
import numpy as np
import time
from nca_model import CAModel
from nca_utils import SamplePool, make_circle_masks
from nca_globals import CHANNEL_N, TARGET_PADDING # TARGET_PADDING needed for file-based padding

class NCATrainer:
    def __init__(self, target_img_rgba_processed, config): # target_img_rgba_processed is already TARGET_SIZE for files, or (TARGET_SIZE + 2*PAD) for drawn
        self.config = config
        self.ca = CAModel(channel_n=CHANNEL_N, fire_rate=config['fire_rate'])
        
        target_source_kind = config.get('target_source_kind', 'file') # Default to file if not specified

        self.best_loss = float('inf') # Initialize best loss tracking
        self.best_model_save_path = os.path.join(config['model_folder_path'], "best_trainer_model.h5")
        tf.print(f"NCATrainer: Best model will be saved to {self.best_model_save_path}")

        if target_source_kind == "drawn_defines_padded_grid":
            # For drawn patterns, target_img_rgba_processed is already the final padded size
            self.pad_target = tf.convert_to_tensor(target_img_rgba_processed, dtype=tf.float32)
            tf.print(f"NCATrainer: Using pre-sized drawn target. Shape: {self.pad_target.shape}")
        else: # Default for "file" source or unspecified
            # For file-loaded patterns, target_img_rgba_processed is TARGET_SIZE x TARGET_SIZE content
            pad_amount = config.get('target_padding', TARGET_PADDING) # Use configured padding
            self.pad_target = tf.pad(target_img_rgba_processed, 
                                     [(pad_amount, pad_amount)]*2 + [(0,0)])
            tf.print(f"NCATrainer: Padded file target. Original content shape: {target_img_rgba_processed.shape}, Padded shape: {self.pad_target.shape}")
        
        self.pool = self._initialize_pool() 
        self.loss_log = []
        self.current_step = 0
        self.training_start_time = None
        self.last_preview_state = None
        self.last_loss = None
        self.total_training_time_paused = 0.0 # New: Accumulates time spent paused
        self.last_pause_time = None # New: Timestamp when training was last paused
        self.total_training_time_paused = 0.0 # New: Accumulates time spent paused
        self.last_pause_time = None # New: Timestamp when training was last paused

        lr = self.config.get('learning_rate', 2e-3)
        lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [2000], [lr, lr * 0.1] 
        )
        self.optimizer = tf.keras.optimizers.Adam(lr_sched)

    def _initialize_pool(self):
        # Pool initialization should use the shape of self.pad_target
        # as this is the actual grid size the CA operates on.
        h_grid, w_grid = self.pad_target.shape[:2]
        seed = np.zeros([h_grid, w_grid, CHANNEL_N], np.float32)
        # Seed in the center of the *actual operational grid*
        if h_grid > 0 and w_grid > 0:
            seed[h_grid // 2, w_grid // 2, 3:] = 1.0  
        
        pool_s = self.config.get('pool_size', 1024) # Get pool_size from config
        initial_states = np.repeat(seed[None, ...], pool_s, axis=0)
        return SamplePool(x=initial_states) 

    def _loss_fn(self, x_state):
        # Loss is always calculated against self.pad_target
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
            return None, None 

        if self.training_start_time is None:
            self.training_start_time = time.time()

        num_to_sample = self.config['batch_size']
        sampled_indices = self.pool.get_indices(num_to_sample)

        if sampled_indices.size == 0 :
             return None, None 

        x0_from_pool_batch = self.pool.x[sampled_indices].copy() 
        
        damage_n_config = self.config.get('damage_n', 0)
        if damage_n_config > 0 and x0_from_pool_batch.shape[0] > 0: 
            # Damage should be applied to the pad_target dimensions
            h_grid_for_damage, w_grid_for_damage = self.pad_target.shape[:2]
            num_to_damage_actual = min(damage_n_config, x0_from_pool_batch.shape[0]) 
            
            if num_to_damage_actual > 0:
                damage_masks_np = make_circle_masks(num_to_damage_actual, h_grid_for_damage, w_grid_for_damage).numpy()[..., None]
                x0_from_pool_batch[-num_to_damage_actual:] *= (1.0 - damage_masks_np)

        if x0_from_pool_batch.shape[0] == 0: 
            return None, None

        x_updated_batch_tf, loss_tf = self._train_step_tf(tf.convert_to_tensor(x0_from_pool_batch))
        
        self.pool.x[sampled_indices] = x_updated_batch_tf.numpy()

        self.current_step += 1
        self.last_loss = loss_tf.numpy()
        self.loss_log.append(self.last_loss)
        self.last_preview_state = x_updated_batch_tf[0].numpy().copy()
        
        self._save_best_model_if_improved() # Call the new method here

        return self.last_preview_state, self.last_loss

    def _save_best_model_if_improved(self):
        if self.last_loss is None or self.current_step == 0:
            return

        # Check every 100 steps
        if self.current_step % 100 == 0:
            if self.last_loss < self.best_loss:
                tf.print(f"NCATrainer: New best loss found at step {self.current_step}: {self.last_loss:.4f} (previous best: {self.best_loss:.4f}). Saving model.")
                self.best_loss = self.last_loss
                try:
                    self.ca.save_weights(self.best_model_save_path)
                    tf.print(f"NCATrainer: Best model weights saved to {self.best_model_save_path}")
                except Exception as e:
                    tf.print(f"NCATrainer: Error saving best model weights: {e}")
            # else:
                # tf.print(f"NCATrainer: Current loss {self.last_loss:.4f} not better than best {self.best_loss:.4f} at step {self.current_step}.")

    def pause_training_timer(self):
        """Pauses the training timer, accumulating elapsed time."""
        if self.training_start_time and self.last_pause_time is None:
            self.last_pause_time = time.time()
            tf.print(f"NCATrainer: Training timer paused at step {self.current_step}.")

    def resume_training_timer(self):
        """Resumes the training timer, accounting for paused time."""
        if self.training_start_time and self.last_pause_time is not None:
            self.total_training_time_paused += (time.time() - self.last_pause_time)
            self.last_pause_time = None
            tf.print(f"NCATrainer: Training timer resumed. Total paused time: {self.total_training_time_paused:.2f}s")
        elif self.training_start_time is None: # First start
            self.training_start_time = time.time()
            tf.print("NCATrainer: Training timer started for the first time.")

    def get_status(self):
        elapsed_time_val = 0
        if self.training_start_time:
            current_active_time = time.time()
            if self.last_pause_time is not None: # If currently paused, use the pause time as the "current" time
                current_active_time = self.last_pause_time
            elapsed_time_val = (current_active_time - self.training_start_time) - self.total_training_time_paused
            if elapsed_time_val < 0: # Prevent negative time if clock skews or very short intervals
                elapsed_time_val = 0

        current_loss_val = self.last_loss if self.last_loss is not None else 0.0
        log_loss_val_str = "N/A"
        if current_loss_val > 1e-9:
            try:
                log_loss_val_str = f"{np.log10(current_loss_val):.3f}"
            except Exception:
                pass

        return {
            "step": self.current_step,
            "loss": f"{current_loss_val:.4f}",
            "log_loss": log_loss_val_str,
            "training_time_seconds": elapsed_time_val, # Raw seconds
        }

    def get_model(self):
        return self.ca

    def load_weights(self, path_to_weights):
        try:
            self.ca.load_weights(path_to_weights)
        except Exception as e_detail:
            tf.print(f"Trainer: Error loading weights: {e_detail}")

    def save_weights(self, path_to_save):
        try:
            self.ca.save_weights(path_to_save)
        except Exception as e_detail:
            tf.print(f"Trainer: Error saving weights: {e_detail}")