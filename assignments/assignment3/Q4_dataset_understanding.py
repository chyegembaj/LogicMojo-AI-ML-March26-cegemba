# Q4. Dataset Understanding (Titanic Dataset)

# Context: First step in ML pipeline is dataset inspection.

# Dataset: Titanic (Seaborn)

# Tasks:
import seaborn as sns
df = sns.load_dataset("titanic")

# Display: head(), tail() info(), describe()
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())




# Identify: Missing values per column Numerical vs categorical features
print(df[df['age'].isnull()])

print(df[df['deck'].isna()])

# Explain: Why identifying feature types is important before modeling
## This helps in undersatnding the type of data for each feature so that the right preprocessing is applied.
#
