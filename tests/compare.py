import numpy as np


def compare_equal(true_output, test_output):
    test_array = np.array(test_output, dtype=np.float32)
    true_array = np.array(true_output, dtype=np.float32)
    print("test_array:", test_array)
    print("true_array:", true_array)
    print("test_array_shape:", test_array.shape)
    print("true_array_shape:", true_array.shape)
    return np.allclose(true_array, test_array)
