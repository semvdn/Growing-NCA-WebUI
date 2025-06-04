# nca_runner.py
"""NCA Runner Logic."""

import numpy as np
import tensorflow as tf
from nca_globals import CHANNEL_N, HISTORY_MAX_SIZE

class NCARunner:
    def __init__(self, ca_model_instance, initial_state_shape_tuple):
        self.ca = ca_model_instance # Expecting an initialized CAModel instance
        
        if len(initial_state_shape_tuple) == 3 and all(isinstance(dim, int) and dim > 0 for dim in initial_state_shape_tuple):
            self.h, self.w, self.ch = initial_state_shape_tuple
        else: 
            self.h, self.w, self.ch = 72, 72, CHANNEL_N 
            tf.print(f"NCARunner: Invalid initial_state_shape {initial_state_shape_tuple}. Using default {(self.h,self.w,self.ch)}.")

        self.current_state = self._initialize_run_state() # np.ndarray
        self.history = [self.current_state.copy()] 
        self.history_index = 0
        tf.print(f"NCARunner initialized with shape: {(self.h, self.w, self.ch)}")


    def _initialize_run_state(self):
        seed = np.zeros([self.h, self.w, self.ch], np.float32)
        # Ensure h, w are large enough for center seed. Min 1x1 grid.
        center_h = max(0, self.h // 2)
        center_w = max(0, self.w // 2)
        if self.h > 0 and self.w > 0: # Only seed if grid is valid
             seed[center_h, center_w, 3:] = 1.0 
        return seed 

    def reset_state(self):
        tf.print("NCARunner: Resetting state.")
        self.current_state = self._initialize_run_state()
        self.history = [self.current_state.copy()] 
        self.history_index = 0
        return self.current_state.copy() # Return a copy

    def step(self):
        if not self.ca or self.current_state is None:
            # tf.print("NCARunner: Cannot step, model or state not available.")
            return self.current_state # Return current state (which might be None)
        
        try:
            # CAModel.call expects a batch, so add batch dimension
            x_tensor_batch = tf.convert_to_tensor(self.current_state[None, ...], dtype=tf.float32)
            # Use the fire_rate defined within the loaded CAModel instance
            x_new_batch = self.ca(x_tensor_batch, fire_rate=self.ca.fire_rate) 
            self.current_state = x_new_batch[0].numpy() # Extract from batch

            # Manage history
            if self.history_index < len(self.history) - 1: # If navigated back, truncate future
                self.history = self.history[:self.history_index + 1]
            
            self.history.append(self.current_state.copy()) 
            self.history_index += 1

            if len(self.history) > HISTORY_MAX_SIZE:
                num_to_remove = len(self.history) - HISTORY_MAX_SIZE
                self.history = self.history[num_to_remove:]
                self.history_index -= num_to_remove # Adjust index relative to new history start
            
            return self.current_state # Current state, not a copy for internal use
        except Exception as e_detail:
            tf.print(f"NCARunner: Error during step: {e_detail}")
            # Potentially stop the runner or revert state if error is critical
            return self.current_state # Return previous state on error

    def erase(self, norm_x, norm_y, norm_eraser_size_factor, canvas_render_width, canvas_render_height):
        if self.current_state is None: return self.current_state

        state_h, state_w = self.current_state.shape[:2]

        if canvas_render_width <= 0 or canvas_render_height <= 0: 
            tf.print("NCARunner.erase: Invalid canvas render dimensions.")
            return self.current_state

        # Map normalized click on rendered canvas to state array coordinates
        click_x_on_render = norm_x * canvas_render_width
        click_y_on_render = norm_y * canvas_render_height

        scale_render_to_state_x = state_w / canvas_render_width
        scale_render_to_state_y = state_h / canvas_render_height
        
        erase_center_x_state = int(click_x_on_render * scale_render_to_state_x)
        erase_center_y_state = int(click_y_on_render * scale_render_to_state_y)

        # Eraser radius in state space pixels
        min_state_dim_for_radius = min(state_h, state_w)
        eraser_radius_pixels = max(1, int(norm_eraser_size_factor * min_state_dim_for_radius)) 
        
        # Create circular mask
        y_grid_range, x_grid_range = np.ogrid[-eraser_radius_pixels:eraser_radius_pixels+1, \
                                              -eraser_radius_pixels:eraser_radius_pixels+1]
        circle_mask_boolean = x_grid_range**2 + y_grid_range**2 <= eraser_radius_pixels**2

        # Determine bounds for applying mask in state array, clamping to state dimensions
        x_min_bound = max(0, erase_center_x_state - eraser_radius_pixels)
        x_max_bound = min(state_w, erase_center_x_state + eraser_radius_pixels + 1)
        y_min_bound = max(0, erase_center_y_state - eraser_radius_pixels)
        y_max_bound = min(state_h, erase_center_y_state + eraser_radius_pixels + 1)

        # Determine the slice of the circle_mask_boolean to use
        mask_slice_x_start = eraser_radius_pixels - (erase_center_x_state - x_min_bound)
        mask_slice_x_end = mask_slice_x_start + (x_max_bound - x_min_bound)
        mask_slice_y_start = eraser_radius_pixels - (erase_center_y_state - y_min_bound)
        mask_slice_y_end = mask_slice_y_start + (y_max_bound - y_min_bound)
        
        if x_min_bound < x_max_bound and y_min_bound < y_max_bound: # Valid slice
            active_mask_for_state = circle_mask_boolean[
                mask_slice_y_start:mask_slice_y_end, 
                mask_slice_x_start:mask_slice_x_end
            ]
            
            # Ensure shapes match for broadcasting
            if active_mask_for_state.shape == (y_max_bound - y_min_bound, x_max_bound - x_min_bound):
                self.current_state[y_min_bound:y_max_bound, x_min_bound:x_max_bound, :] *= \
                    ~active_mask_for_state[..., np.newaxis] # Apply inverse mask

                # Update history after modification
                if self.history_index < len(self.history) - 1:
                    self.history = self.history[:self.history_index + 1]
                self.history.append(self.current_state.copy())
                self.history_index += 1
                if len(self.history) > HISTORY_MAX_SIZE:
                    num_to_remove = len(self.history) - HISTORY_MAX_SIZE
                    self.history = self.history[num_to_remove:]
                    self.history_index -= num_to_remove
            # else:
                # tf.print("NCARunner.erase: Mask slice shape mismatch.")
        
        return self.current_state.copy() # Return a copy

    def rewind(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.current_state = self.history[self.history_index].copy() 
        return self.current_state.copy(), self.history_index

    def fast_forward(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.current_state = self.history[self.history_index].copy() 
        return self.current_state.copy(), self.history_index
        
    def get_current_state_for_display(self):
        # Return a copy to prevent external modification of the internal state
        return self.current_state.copy() if self.current_state is not None else None