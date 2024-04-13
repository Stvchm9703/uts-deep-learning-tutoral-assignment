import tensorflow as tf
from pprint import pprint
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

pprint(
  {
    'gpu_available' : tf.test.is_gpu_available(),
    'is_cuda_gpu_available' : tf.test.is_gpu_available(cuda_only=True),
    'num_of_device': len(tf.config.list_logical_devices( device_type='GPU' ))
  }
)

# tf.test.