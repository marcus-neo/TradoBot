import numpy as np


def compare_equal(true_output, test_output):
    test_array = np.array(test_output, dtype=np.float32)
    true_array = np.array(true_output, dtype=np.float32)
    return np.allclose(true_array, test_array, atol=1e-03)
