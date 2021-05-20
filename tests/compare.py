import numpy as np


def compare_equal(test_output, true_output):
    test_array = np.array(test_output, dtype=np.float32)
    true_array = np.array(true_output, dtype=np.float32)
    return np.allclose(test_array, true_array)
