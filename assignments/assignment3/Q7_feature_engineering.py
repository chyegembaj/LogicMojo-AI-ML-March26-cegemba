# Q7. Feature Engineering for Model Improvement

# Context: Raw data is rarely useful—features must be created.

# Dataset: Titanic
import seaborn as sns
df = sns.load_dataset("titanic")

# Tasks:

# 1. Create:
# family_size = sibsp + parch + 1
df['family_size'] = df['sibsp'] + df['parch'] + 1


# 2. Create:
# is_alone (binary feature)
df['is_alone'] = (df['sibsp'] + df['parch']) == 0


# 3. Compute survival rate by is_alone
print(df.groupby('is_alone')['survived'].mean())
#print(df.head())

# 4. Explain:
# Why engineered features improve ML models
#Engineered features improve ML models because they turn raw data into clearer, 
#more meaningful signals that models can learn from more easily.