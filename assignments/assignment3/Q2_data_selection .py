
# Q2. Data Selection for Model Input

# Context: Models are trained on selective features, not full raw data.

# Dataset: Iris

# Tasks:

import seaborn as sns
import pandas as pd


# Extract: First 100 samples Only last 2 features
df = sns.load_dataset("iris")

print(df.iloc[ : 100, 2 : 4])

# Use boolean masking: Select samples where petal length is greater than dataset mean
print(df[df['petal_length'] > df['petal_length'].mean()])

# Count selected samples
print(df.iloc[ : 100, 2 : 4].count())

# Explain how this relates to feature-based filtering in ML
