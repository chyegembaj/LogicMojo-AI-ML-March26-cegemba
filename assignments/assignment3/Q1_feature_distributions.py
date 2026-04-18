# Q1. Understanding Feature Distributions (Iris Dataset)
#
# Context: Before training any ML model, we analyze feature distributions.
#
# Dataset: Load Iris dataset from sklearn
#
# Tasks:
#
# 1. Extract feature matrix as NumPy array
# 2. Compute:
# Mean
# Median
# Standard deviation
# Variance (for each feature)
# 3. Identify:
# Which feature has highest variability and why it matters in ML
# 4. Convert any one feature into shape (n,1) and explain why ML models expect this format


import numpy as np
from sklearn.datasets import load_iris

# Dataset: Load Iris dataset from sklearn
iris = load_iris()


# 1. Extract feature matrix as NumPy array
data = iris.data
#print(data)

# 2. Compute:
# Mean
mean = np.mean(data, axis=0)
#print(mean)

# Median
median = np.median(data, axis=0)

# Standard deviation
standard_deviation = np.std(data, axis=0)

# Variance (for each feature)
variance = np.var(data, axis=0)
#print(variance)



# 3. Identify:
# Which feature has highest variability and why it matters in ML
#petal length has the  highest variability. Its matters because it indicates greater spread in data which helps
# machine models distinguish between samples.

# 4. Convert any one feature into shape (n,1) and explain why ML models expect this format

data[: , 0].reshape()




