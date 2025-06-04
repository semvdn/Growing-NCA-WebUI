import tensorflow as tf
print(f"TensorFlow Version: {tf.__version__}")
print(f"Is TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}") # Should be True
gpus = tf.config.list_physical_devices('GPU')
print(f"Num GPUs Available: {len(gpus)}")
if gpus:
    print("GPUs found:", gpus)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        details = tf.config.experimental.get_device_details(gpus[0])
        print("GPU Details:", details.get('device_name', 'Unknown GPU'))
    except RuntimeError as e:
        print("Error during GPU setup:", e)
else:
    print("No GPUs found by TensorFlow.")