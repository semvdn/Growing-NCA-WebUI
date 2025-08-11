# nca_trainer.py
"""NCA Training Logic."""

import os
import tensorflow as tf
import numpy as np
import time
from nca_model import CAModel
from nca_utils import SamplePool, make_circle_masks
from nca_globals import CHANNEL_N, TARGET_PADDING, DEFAULT_ENTROPY_ENABLED, DEFAULT_ENTROPY_STRENGTH

class NCATrainer:
    def __init__(self, target_img_rgba_processed, config):
        self.config = config
        
        enable_entropy = config.get('enable_entropy', DEFAULT_ENTROPY_ENABLED)
        entropy_strength = config.get('entropy_strength', DEFAULT_ENTROPY_STRENGTH)

        self.ca = CAModel(channel_n=CHANNEL_N, fire_rate=config['fire_rate'],
                          enable_entropy=enable_entropy, entropy_strength=entropy_strength)
        
        self.run_dir = config.get('run_dir')
        target_source_kind = config.get('target_source_kind', 'file')

        self.best_loss = float('inf')
        if self.run_dir:
            self.best_model_save_path = os.path.join(self.run_dir, "best_model.weights.h5")
            tf.print(f"NCATrainer: Best model will be saved to {self.best_model_save_path}")
        else:
            self.best_model_save_path = None
            tf.print("NCATrainer: No run_dir provided, 'best model' checkpointing is disabled.")

        self.pad_target = tf.convert_to_tensor(target_img_rgba_processed, dtype=tf.float32)
        
        h_grid, w_grid = self.pad_target.shape[:2]
        self.seed_state = np.zeros([h_grid, w_grid, CHANNEL_N], np.float32)
        if h_grid > 0 and w_grid > 0:
            self.seed_state[h_grid // 2, w_grid // 2, 3:] = 1.0

        self.pool = self._initialize_pool()
        self.loss_log = []
        self.current_step = 0
        self.training_start_time = None
        self.last_preview_state = None
        self.last_loss = None
        self.total_training_time_paused = 0.0
        self.last_pause_time = None

        lr = self.config.get('learning_rate', 2e-3)
        lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([2000], [lr, lr * 0.1])
        self.optimizer = tf.keras.optimizers.Adam(lr_sched)

    def _initialize_pool(self):
        pool_s = self.config.get('pool_size', 1024)
        initial_states = np.repeat(self.seed_state[None, ...], pool_s, axis=0)
        return SamplePool(x=initial_states)

    def _loss_fn(self, x_state):
        return tf.reduce_mean(tf.square(x_state[..., :4] - self.pad_target), axis=[1, 2, 3])

    @tf.function
    def _train_step_tf(self, x_batch_tensor):
        iter_n = tf.random.uniform([], 64, 96, dtype=tf.int32)
        with tf.GradientTape() as tape:
            x_intermediate = x_batch_tensor
            for _ in tf.range(iter_n):
                x_intermediate = self.ca(x_intermediate, fire_rate=self.ca.fire_rate,
                                         enable_entropy=self.ca.enable_entropy,
                                         entropy_strength=self.ca.entropy_strength)
            loss_val = tf.reduce_mean(self._loss_fn(x_intermediate))
        
        trainable_vars = self.ca.trainable_weights
        grads = tape.gradient(loss_val, trainable_vars)
        
        # --- MODIFICATION START: Implement robust gradient clipping ---
        # Instead of normalizing each gradient individually, we clip all of them
        # together based on their combined (global) norm. This is a more stable
        # and standard way to prevent exploding gradients.
        
        # Filter out None gradients (for layers that might not be training)
        grads, trainable_vars_with_grads = zip(*[(g, v) for g, v in zip(grads, trainable_vars) if g is not None])
        
        # Clip the gradients
        clipped_grads, _ = tf.clip_by_global_norm(grads, 1.0)
        
        # Apply the clipped gradients
        self.optimizer.apply_gradients(zip(clipped_grads, trainable_vars_with_grads))
        # --- MODIFICATION END ---
        
        return x_intermediate, loss_val

    def run_training_step(self):
        if self.training_start_time is None: self.training_start_time = time.time()
        batch_size = self.config['batch_size']
        experiment_type = self.config.get('experiment_type', 'Growing')

        if experiment_type == "Growing":
            x0_batch = np.repeat(self.seed_state[None, ...], batch_size, axis=0)
            sampled_indices = None
        else:
            if not self.pool or len(self.pool) == 0: return None, None
            sampled_indices = self.pool.get_indices(batch_size)
            if sampled_indices.size == 0: return None, None
            
            x0_batch = self.pool.x[sampled_indices].copy()
            loss_ranks = self._loss_fn(tf.convert_to_tensor(x0_batch)).numpy().argsort()[::-1]
            x0_batch, sampled_indices = x0_batch[loss_ranks], sampled_indices[loss_ranks]
            x0_batch[0] = self.seed_state

            damage_n_config = self.config.get('damage_n', 0)
            if damage_n_config > 0 and x0_batch.shape[0] > 0:
                h_grid, w_grid = self.pad_target.shape[:2]
                num_to_damage = min(damage_n_config, x0_batch.shape[0])
                if num_to_damage > 0:
                    damage_masks = make_circle_masks(num_to_damage, h_grid, w_grid).numpy()[..., None]
                    x0_batch[-num_to_damage:] *= (1.0 - damage_masks)

        if x0_batch.shape[0] == 0: return None, None

        x_updated_batch, loss = self._train_step_tf(tf.convert_to_tensor(x0_batch))
        
        if sampled_indices is not None and experiment_type != "Growing":
            self.pool.x[sampled_indices] = x_updated_batch.numpy()

        self.current_step += 1
        self.last_loss = loss.numpy()
        self.loss_log.append(self.last_loss)
        self.last_preview_state = x_updated_batch[0].numpy().copy()
        
        self._save_best_model_if_improved()
        return self.last_preview_state, self.last_loss

    # The rest of the file remains the same.
    # ... (get_status, get_model, etc.) ...
    def _save_best_model_if_improved(self):
        if self.last_loss is None or self.current_step == 0 or self.best_model_save_path is None:
            return
        if self.current_step % 100 == 0 and self.last_loss < self.best_loss:
            tf.print(f"NCATrainer: New best loss at step {self.current_step}: {self.last_loss:.4f}. Saving model.")
            self.best_loss = self.last_loss
            try:
                self.ca.save_weights(self.best_model_save_path)
            except Exception as e:
                tf.print(f"NCATrainer: Error saving best model: {e}")

    def pause_training_timer(self):
        if self.training_start_time and self.last_pause_time is None:
            self.last_pause_time = time.time()

    def resume_training_timer(self):
        if self.training_start_time and self.last_pause_time is not None:
            self.total_training_time_paused += (time.time() - self.last_pause_time)
            self.last_pause_time = None
        elif self.training_start_time is None:
            self.training_start_time = time.time()

    def get_status(self):
        elapsed_time_val = 0
        if self.training_start_time:
            current_active_time = self.last_pause_time or time.time()
            elapsed_time_val = max(0, (current_active_time - self.training_start_time) - self.total_training_time_paused)
        current_loss_val = self.last_loss if self.last_loss is not None else 0.0
        log_loss_val_str = f"{np.log10(current_loss_val):.3f}" if current_loss_val > 1e-9 else "N/A"
        return {"step": self.current_step, "loss": f"{current_loss_val:.4f}", "log_loss": log_loss_val_str, "training_time_seconds": elapsed_time_val}

    def get_model(self):
        return self.ca

    def load_weights(self, path_to_weights):
        try:
            self.ca.load_weights(path_to_weights)
        except Exception as e:
            tf.print(f"Trainer: Error loading weights: {e}")

    def save_weights(self, path_to_save):
        try:
            self.ca.save_weights(path_to_save)
        except Exception as e:
            tf.print(f"Trainer: Error saving weights: {e}")