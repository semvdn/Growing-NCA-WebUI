# model_converter.py
"""
Converts trained Keras CAModels into JavaScript-runnable formats.

This module provides functions to export a model into two formats:
1. A standard TensorFlow.js GraphModel JSON.
2. A custom, quantized format suitable for a high-performance WebGL demo.

The logic is adapted from the "Growing Neural Cellular Automata" Colab notebook.
"""

import base64
import json
import numpy as np
import tensorflow as tf
from google.protobuf.json_format import MessageToDict
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from nca_globals import CHANNEL_N
from nca_model import CAModel


def export_to_tfjs_format(ca_model: CAModel) -> dict:
    """
    Exports a CA model to a TensorFlow.js compatible JSON format (GraphModel).

    Args:
        ca_model: The trained CAModel instance.

    Returns:
        A dictionary representing the TF.js GraphModel JSON.
    """
    concrete_func = ca_model.call.get_concrete_function(
        x=tf.TensorSpec([None, None, None, CHANNEL_N]),
        fire_rate=tf.constant(0.5),
        angle=tf.constant(0.0),
        step_size=tf.constant(1.0)
    )
    constant_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = constant_func.graph.as_graph_def()
    graph_json = MessageToDict(graph_def)
    graph_json['versions'] = dict(producer='1.14', minConsumer='1.14')
    
    model_json = {
        'format': 'graph-model',
        'modelTopology': graph_json,
        'weightsManifest': [],
    }
    return model_json


def _pack_layer(weight: np.ndarray, bias: np.ndarray, outputType=np.uint8) -> dict:
    """
    Packs and quantizes a single Conv2D layer's weights and biases for WebGL.

    Args:
        weight: The kernel weights of the layer.
        bias: The bias weights of the layer.
        outputType: The numpy dtype for quantization (typically np.uint8).

    Returns:
        A dictionary containing the packed, base64-encoded data and scaling factors.
    """
    in_ch, out_ch = weight.shape
    assert (in_ch % 4 == 0) and (out_ch % 4 == 0) and (bias.shape == (out_ch,)), \
        "Layer dimensions must be divisible by 4 for WebGL packing"
    
    weight_scale, bias_scale = 1.0, 1.0
    if outputType == np.uint8:
        weight_scale = 2.0 * np.abs(weight).max()
        bias_scale = 2.0 * np.abs(bias).max()
        # Clamp to avoid precision errors with very small scales
        if weight_scale < 1e-6: weight_scale = 1.0
        if bias_scale < 1e-6: bias_scale = 1.0
        
        weight = np.round((weight / weight_scale + 0.5) * 255)
        bias = np.round((bias / bias_scale + 0.5) * 255)

    packed = np.vstack([weight, bias[None, ...]])
    packed = packed.reshape(in_ch + 1, out_ch // 4, 4)
    packed = outputType(packed)
    packed_b64 = base64.b64encode(packed.tobytes()).decode('ascii')
    
    return {
        'data_b64': packed_b64,
        'in_ch': in_ch,
        'out_ch': out_ch,
        'weight_scale': float(weight_scale),  # Cast numpy.float32 to python float
        'bias_scale': float(bias_scale),      # Cast numpy.float32 to python float
        'type': outputType.__name__
    }

def export_to_webgl_format(ca_model: CAModel, outputType=np.uint8) -> list:
    """
    Exports a CA model to a custom quantized JSON format for the WebGL demo.

    Args:
        ca_model: The trained CAModel instance.
        outputType: The numpy dtype for quantization (typically np.uint8).
    
    Returns:
        A list of dictionaries, where each dictionary represents a packed layer.
    """
    chn = ca_model.channel_n
    w1 = ca_model.weights[0][0, 0].numpy()
    w1_reordered = w1.reshape(chn, 3, -1).transpose(1, 0, 2).reshape(3 * chn, -1)

    layers_data = [
        _pack_layer(w1_reordered, ca_model.weights[1].numpy(), outputType),
        _pack_layer(ca_model.weights[2][0, 0].numpy(), ca_model.weights[3].numpy(), outputType)
    ]
    
    return layers_data