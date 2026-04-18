# Q6. Handling Missing Data (Critical ML Step)

# Context: Models cannot handle null values.

# Dataset: Titanic
import seaborn as sns
df = sns.load_dataset("titanic")


# Tasks:

# 1. Fill missing age using median

df['age'] = df['age'].fillna(df['age'].median())

#print(df['age'].isna().sum())


# 2. Fill missing embarked using mode
#print(df.info())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
#print(df['embarked'].isna().sum())


# 3. Drop deck column
print(df.head())
df_deck = df.drop(columns=['deck'])
print(df_deck)

# 4. Explain: Why different strategies are used for different columns