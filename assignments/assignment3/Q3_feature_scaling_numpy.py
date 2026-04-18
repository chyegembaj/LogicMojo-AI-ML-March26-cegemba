#%% md
# Q3. Feature Scaling using NumPy (Very Important)

# Context: Most ML models require normalized data.

# Dataset: Iris (NumPy array)

# Tasks:



# 1. Apply standardization:

# $$
# X' = \frac{X - \mu}{\sigma}
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()

print(iris.keys())


X = iris.data

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_scaled = (X - mean) / std

# 	​2. Verify: Mean ≈ 0 Std ≈ one
print(np.round(np.mean(X_scaled, axis=0), 10))
print(np.std(X_scaled, axis=0))
