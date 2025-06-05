# nca_runner.py
"""NCA Runner Logic."""

import numpy as np
import tensorflow as tf
import threading
import time
from nca_globals import CHANNEL_N, HISTORY_MAX_SIZE

class NCARunner:
    def __init__(self, ca_model_instance, initial_state_shape_tuple):
        self.ca = ca_model_instance 
        self._state_lock = threading.Lock() 

        if len(initial_state_shape_tuple) == 3 and all(isinstance(dim, int) and dim > 0 for dim in initial_state_shape_tuple):
            self.h, self.w, self.ch = initial_state_shape_tuple
        else: 
            self.h, self.w, self.ch = 72, 72, CHANNEL_N 
            tf.print(f"NCARunner: Invalid initial_state_shape {initial_state_shape_tuple}. Using default {(self.h,self.w,self.ch)}.")

        with self._state_lock:
            self.current_state = self._initialize_run_state_unsafe()
            self.history = [self.current_state.copy()]
            self.history_index = 0
        
        # New attributes for FPS calculation
        self.actual_fps = 0.0
        self.last_fps_calc_time = time.perf_counter()
        self.steps_since_last_fps_calc = 0
        
        tf.print(f"NCARunner initialized with shape: {(self.h, self.w, self.ch)}")

    def _initialize_run_state_unsafe(self): 
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

    def step(self): 
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
                
                # Update FPS calculation counters
                self.steps_since_last_fps_calc += 1
                now = time.perf_counter()
                time_delta = now - self.last_fps_calc_time
                if time_delta >= 1.0: # Calculate FPS roughly every second
                    self.actual_fps = self.steps_since_last_fps_calc / time_delta
                    self.last_fps_calc_time = now
                    self.steps_since_last_fps_calc = 0
                
                return self.current_state
            except Exception as e_detail:
                tf.print(f"NCARunner: Error during step: {e_detail}")
                return self.current_state

    def modify_area(self, tool_mode, norm_x, norm_y, norm_brush_size_factor, 
                    canvas_render_width, canvas_render_height, draw_color_hex=None):
        # tf.print(f"NCARunner.modify_area called. Tool: {tool_mode}, NormX: {norm_x:.2f}, NormY: {norm_y:.2f}, BrushFactor: {norm_brush_size_factor:.2f}, Color: {draw_color_hex}")
        with self._state_lock:
            if self.current_state is None: 
                # tf.print("NCARunner.modify_area: current_state is None. Aborting.")
                return None 

            state_h, state_w = self.current_state.shape[:2]
            # tf.print(f"  State HxW: {state_h}x{state_w}. Canvas Render HxW: {canvas_render_height}x{canvas_render_width}")

            if canvas_render_width <= 0 or canvas_render_height <= 0: 
                # tf.print("NCARunner.modify_area: Invalid canvas render dimensions.")
                return self.current_state.copy() 

            click_x_on_render = norm_x * canvas_render_width
            click_y_on_render = norm_y * canvas_render_height

            scale_render_to_state_x = state_w / canvas_render_width
            scale_render_to_state_y = state_h / canvas_render_height
            
            center_x_state = int(click_x_on_render * scale_render_to_state_x)
            center_y_state = int(click_y_on_render * scale_render_to_state_y)

            min_state_dim_for_radius = min(state_h, state_w)
            brush_radius_pixels = max(1, int(norm_brush_size_factor * min_state_dim_for_radius)) 
            # tf.print(f"  Calculated state center: ({center_y_state},{center_x_state}), radius_pixels: {brush_radius_pixels}")
            
            y_grid_range, x_grid_range = np.ogrid[-brush_radius_pixels:brush_radius_pixels+1, \
                                                  -brush_radius_pixels:brush_radius_pixels+1]
            circle_mask_boolean = y_grid_range**2 + x_grid_range**2 <= brush_radius_pixels**2
            # tf.print(f"  Generated circle_mask_boolean shape: {circle_mask_boolean.shape}, sum: {np.sum(circle_mask_boolean)}")

            x_min_bound = max(0, center_x_state - brush_radius_pixels)
            x_max_bound = min(state_w, center_x_state + brush_radius_pixels + 1)
            y_min_bound = max(0, center_y_state - brush_radius_pixels)
            y_max_bound = min(state_h, center_y_state + brush_radius_pixels + 1)
            # tf.print(f"  State bounds for modification: Y({y_min_bound}:{y_max_bound}), X({x_min_bound}:{x_max_bound})")

            mask_slice_x_start = brush_radius_pixels - (center_x_state - x_min_bound)
            mask_slice_x_end = mask_slice_x_start + (x_max_bound - x_min_bound)
            mask_slice_y_start = brush_radius_pixels - (center_y_state - y_min_bound)
            mask_slice_y_end = mask_slice_y_start + (y_max_bound - y_min_bound)
            
            if x_min_bound < x_max_bound and y_min_bound < y_max_bound: 
                active_mask_for_state = circle_mask_boolean[
                    mask_slice_y_start:mask_slice_y_end, 
                    mask_slice_x_start:mask_slice_x_end
                ]
                # tf.print(f"  active_mask_for_state shape: {active_mask_for_state.shape}, sum: {np.sum(active_mask_for_state)}")
                
                if active_mask_for_state.shape == (y_max_bound - y_min_bound, x_max_bound - x_min_bound):
                    broadcast_mask = active_mask_for_state[..., np.newaxis]
                    if tool_mode == 'erase':
                        self.current_state[y_min_bound:y_max_bound, x_min_bound:x_max_bound, :] *= (1 - broadcast_mask.astype(self.current_state.dtype))
                        # tf.print(f"  NCARunner: Area erased.")
                    elif tool_mode == 'draw' and draw_color_hex:
                        try:
                            h = draw_color_hex.lstrip('#')
                            rgb_norm = [int(h[i:i+2], 16)/255.0 for i in (0, 2, 4)] 
                            target_slice = self.current_state[y_min_bound:y_max_bound, x_min_bound:x_max_bound, :]
                            for c_idx in range(3): 
                                target_slice[..., c_idx] = np.where(active_mask_for_state, rgb_norm[c_idx], target_slice[..., c_idx])
                            target_slice[..., 3] = np.where(active_mask_for_state, 1.0, target_slice[..., 3]) 
                            target_slice[..., 4:] = np.where(broadcast_mask, 1.0, target_slice[..., 4:])
                            self.current_state[y_min_bound:y_max_bound, x_min_bound:x_max_bound, :] = target_slice
                            # tf.print(f"  NCARunner: Area drawn with color {rgb_norm}.")
                        except Exception as e_draw:
                            tf.print(f"  NCARunner: Error processing draw color: {e_draw}")
                    
                    if self.history_index < len(self.history) - 1:
                        self.history = self.history[:self.history_index + 1]
                    self.history.append(self.current_state.copy())
                    self.history_index += 1
                    if len(self.history) > HISTORY_MAX_SIZE:
                        num_to_remove = len(self.history) - HISTORY_MAX_SIZE
                        self.history = self.history[num_to_remove:]
                        self.history_index -= num_to_remove
                # else:
                    # tf.print(f"  NCARunner.modify_area: Mask slice shape mismatch.")
            # else:
                # tf.print("  NCARunner.modify_area: Calculated modification bounds are invalid or zero-sized.")
            return self.current_state.copy() 

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