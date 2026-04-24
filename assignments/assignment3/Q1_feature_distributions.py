# Q1. Understanding Feature Distributions (Iris Dataset)
#
# Context: Before training any ML model, we analyze feature distributions.
#
# Dataset: Load Iris dataset from sklearn
#
# Tasks:
import numpy as np
from sklearn.datasets import load_iris

# Dataset: Load Iris dataset from sklearn
iris = load_iris()


# 1. Extract feature matrix as NumPy array
x = iris.data
#print(data)

# 2. Compute:
# Mean
mean = np.mean(x, axis=0)
#print(mean)

# Median
median = np.median(x, axis=0)

# Standard deviation
standard_deviation = np.std(x, axis=0)

# Variance (for each feature)
variance = np.var(x, axis=0)
#print(variance)



# 3. Identify:
# Which feature has highest variability and why it matters in ML
max_var_index = np.argmax(variance)
feature_names = iris.feature_names
features_with_highest_var = feature_names[max_var_index]
print(features_with_highest_var)

#petal length has the  highest variability. Its matters because it indicates greater spread in data which helps
# machine models distinguish between samples.

# 4. Convert any one feature into shape (n,1) and explain why ML models expect this format

print(x[: , 0].reshape(-1, 1))




