# nca_runner.py
"""NCA Runner Logic."""

import numpy as np
import tensorflow as tf
import threading # For internal lock
from nca_globals import CHANNEL_N, HISTORY_MAX_SIZE
from nca_utils import to_rgb # For potential debug, not strictly needed here

class NCARunner:
    def __init__(self, ca_model_instance, initial_state_shape_tuple):
        self.ca = ca_model_instance 
        self._state_lock = threading.Lock() # Lock for self.current_state and self.history

        if len(initial_state_shape_tuple) == 3 and all(isinstance(dim, int) and dim > 0 for dim in initial_state_shape_tuple):
            self.h, self.w, self.ch = initial_state_shape_tuple
        else: 
            self.h, self.w, self.ch = 72, 72, CHANNEL_N 
            tf.print(f"NCARunner: Invalid initial_state_shape {initial_state_shape_tuple}. Using default {(self.h,self.w,self.ch)}.")

        # Initialize state under lock
        with self._state_lock:
            self.current_state = self._initialize_run_state_unsafe() # Unsafe for direct call
            self.history = [self.current_state.copy()] 
            self.history_index = 0
        tf.print(f"NCARunner initialized with shape: {(self.h, self.w, self.ch)}")

    def _initialize_run_state_unsafe(self): # Assumes lock is held
        seed = np.zeros([self.h, self.w, self.ch], np.float32)
        center_h = max(0, self.h // 2)
        center_w = max(0, self.w // 2)
        if self.h > 0 and self.w > 0:
             seed[center_h, center_w, 3:] = 1.0 
        return seed 

    def reset_state(self):
        tf.print("NCARunner: Resetting state.")
        with self._state_lock:
            self.current_state = self._initialize_run_state_unsafe()
            self.history = [self.current_state.copy()] 
            self.history_index = 0
            return self.current_state.copy()

    def step(self): # This will be called by app.py's run_thread_lock, so internal lock here is for safety
        with self._state_lock:
            if not self.ca or self.current_state is None:
                return self.current_state 
            
            try:
                x_tensor_batch = tf.convert_to_tensor(self.current_state[None, ...], dtype=tf.float32)
                x_new_batch = self.ca(x_tensor_batch, fire_rate=self.ca.fire_rate) 
                self.current_state = x_new_batch[0].numpy()

                if self.history_index < len(self.history) - 1:
                    self.history = self.history[:self.history_index + 1]
                
                self.history.append(self.current_state.copy()) 
                self.history_index += 1

                if len(self.history) > HISTORY_MAX_SIZE:
                    num_to_remove = len(self.history) - HISTORY_MAX_SIZE
                    self.history = self.history[num_to_remove:]
                    self.history_index -= num_to_remove
                
                return self.current_state # Return direct reference for internal use
            except Exception as e_detail:
                tf.print(f"NCARunner: Error during step: {e_detail}")
                return self.current_state 

    def modify_area(self, tool_mode, norm_x, norm_y, norm_brush_size_factor, 
                    canvas_render_width, canvas_render_height, draw_color_hex=None):
        with self._state_lock:
            if self.current_state is None: return self.current_state

            state_h, state_w = self.current_state.shape[:2]

            if canvas_render_width <= 0 or canvas_render_height <= 0: 
                tf.print("NCARunner.modify_area: Invalid canvas render dimensions.")
                return self.current_state.copy() # Return copy

            click_x_on_render = norm_x * canvas_render_width
            click_y_on_render = norm_y * canvas_render_height

            scale_render_to_state_x = state_w / canvas_render_width
            scale_render_to_state_y = state_h / canvas_render_height
            
            center_x_state = int(click_x_on_render * scale_render_to_state_x)
            center_y_state = int(click_y_on_render * scale_render_to_state_y)

            min_state_dim_for_radius = min(state_h, state_w)
            brush_radius_pixels = max(1, int(norm_brush_size_factor * min_state_dim_for_radius)) 
            
            y_grid_range, x_grid_range = np.ogrid[-brush_radius_pixels:brush_radius_pixels+1, \
                                                  -brush_radius_pixels:brush_radius_pixels+1]
            circle_mask_boolean = x_grid_range**2 + x_grid_range**2 <= brush_radius_pixels**2

            x_min_bound = max(0, center_x_state - brush_radius_pixels)
            x_max_bound = min(state_w, center_x_state + brush_radius_pixels + 1)
            y_min_bound = max(0, center_y_state - brush_radius_pixels)
            y_max_bound = min(state_h, center_y_state + brush_radius_pixels + 1)

            mask_slice_x_start = brush_radius_pixels - (center_x_state - x_min_bound)
            mask_slice_x_end = mask_slice_x_start + (x_max_bound - x_min_bound)
            mask_slice_y_start = brush_radius_pixels - (center_y_state - y_min_bound)
            mask_slice_y_end = mask_slice_y_start + (y_max_bound - y_min_bound)
            
            if x_min_bound < x_max_bound and y_min_bound < y_max_bound: 
                active_mask_for_state = circle_mask_boolean[
                    mask_slice_y_start:mask_slice_y_end, 
                    mask_slice_x_start:mask_slice_x_end
                ]
                
                if active_mask_for_state.shape == (y_max_bound - y_min_bound, x_max_bound - x_min_bound):
                    # Prepare mask for broadcasting (H, W, 1)
                    broadcast_mask = active_mask_for_state[..., np.newaxis]

                    if tool_mode == 'erase':
                        # Zero out all channels where mask is True
                        self.current_state[y_min_bound:y_max_bound, x_min_bound:x_max_bound, :] *= ~broadcast_mask
                        tf.print("NCARunner: Area erased.")
                    elif tool_mode == 'draw' and draw_color_hex:
                        try:
                            h = draw_color_hex.lstrip('#')
                            rgb_norm = [int(h[i:i+2], 16)/255.0 for i in (0, 2, 4)] # R, G, B normalized

                            # Apply to the slice of current_state
                            target_slice = self.current_state[y_min_bound:y_max_bound, x_min_bound:x_max_bound, :]
                            
                            # Where mask is true, set color and liveness
                            target_slice[..., 0] = np.where(active_mask_for_state, rgb_norm[0], target_slice[..., 0])
                            target_slice[..., 1] = np.where(active_mask_for_state, rgb_norm[1], target_slice[..., 1])
                            target_slice[..., 2] = np.where(active_mask_for_state, rgb_norm[2], target_slice[..., 2])
                            target_slice[..., 3] = np.where(active_mask_for_state, 1.0, target_slice[..., 3]) # Alpha
                            # Set hidden channels to 1.0 to encourage "life"
                            target_slice[..., 4:] = np.where(broadcast_mask, 1.0, target_slice[..., 4:])
                            
                            self.current_state[y_min_bound:y_max_bound, x_min_bound:x_max_bound, :] = target_slice
                            tf.print(f"NCARunner: Area drawn with color {rgb_norm}")
                        except Exception as e_draw:
                            tf.print(f"NCARunner: Error processing draw color: {e_draw}")
                    
                    # Update history after modification
                    if self.history_index < len(self.history) - 1:
                        self.history = self.history[:self.history_index + 1]
                    self.history.append(self.current_state.copy())
                    self.history_index += 1
                    if len(self.history) > HISTORY_MAX_SIZE:
                        num_to_remove = len(self.history) - HISTORY_MAX_SIZE
                        self.history = self.history[num_to_remove:]
                        self.history_index -= num_to_remove
            
            return self.current_state.copy() # Return a copy of the modified state

    def rewind(self):
        with self._state_lock:
            if self.history_index > 0:
                self.history_index -= 1
                self.current_state = self.history[self.history_index].copy() 
            return self.current_state.copy(), self.history_index

    def fast_forward(self):
        with self._state_lock:
            if self.history_index < len(self.history) - 1:
                self.history_index += 1
                self.current_state = self.history[self.history_index].copy() 
            return self.current_state.copy(), self.history_index
        
    def get_current_state_for_display(self):
        with self._state_lock:
            return self.current_state.copy() if self.current_state is not None else None