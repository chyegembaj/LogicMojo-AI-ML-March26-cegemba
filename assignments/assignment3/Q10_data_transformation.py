# Q10. Data Transformation for Model Input (NumPy)

# Context: ML models require structured numeric arrays.

# Tasks:

import numpy as np

# 1. Create two arrays of shape (3,2)
arr1 = np.array([[10, 20], [15, 25], [30, 35]])


arr2 = np.array([[2, 5], [7, 8], [10, 11]])


# 2. Perform:
# Vertical stacking
arr1_vstack = np.vstack((arr1, arr2)) 


# Horizontal stacking
arr2_hstack = np.hstack((arr1, arr2))


# 3. Reshape into (2,6)
arr1_reshaped = arr1_vstack.reshape(2, 6)

arr2_reshaped = arr2_hstack.reshape(2, 6)

# 4. Explain:
# Why reshaping is required in ML pipelines