# Q9. Advanced Filtering + Subsetting

# Context: Training data is often filtered based on business rules.

# Dataset: Tips
import seaborn as sns
df = sns.load_dataset("tips")
print(df.head())
# Tasks:

# 1. Filter:
# total_bill > 20 AND tip < 3
segment = df[(df['total_bill'] > 20) & (df['tip'] < 3)] 

# 2. Select only relevant columns
segment[['total_bill', 'tip', 'sex', 'day']]

# 3. Analyze:
# Is this segment under-tipping?
print(segment['tip'].mean())
print(df['tip'].mean())

# 4. Explain:
# How such filtering helps anomaly detection