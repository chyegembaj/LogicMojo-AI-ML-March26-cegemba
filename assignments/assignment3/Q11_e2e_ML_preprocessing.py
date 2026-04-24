# Q11. End-to-End ML Preprocessing Pipeline (Titanic)

# Context: Prepare dataset for ML model training

# Tasks:

# 1. Load dataset
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
df = sns.load_dataset("titanic")

# 2. Handle missing values
#Fill in missing age with median age
df['age'] = df['age'].fillna(df['age'].median()) 

#Remove deck column since it has large number of missing data
df = df.drop(columns='deck')

df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])


# 3. Encode categorical variables (sex, embarked)
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)



# 4. Normalize numerical features
x = df.drop(columns='survived')
y = df['survived']

scaler = StandardScaler()
num_cols = ['pclass', 'age', 'sibsp', 'parch', 'fare']
x[num_cols] = scaler.fit_transform(df[num_cols])

# 5. Convert to NumPy array
x_numpy = x.to_numpy()
y_numpy = y.to_numpy()


# Output:

# Final feature matrix ready for model training
x_numpy  # Feature matrix
