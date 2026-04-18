# Q5. Filtering Data for Business Logic

# Context: Real ML problems require filtering relevant populations.

# Dataset: Titanic
import seaborn as sns

df = sns.load_dataset("titanic")
# Tasks:

# Filter: Female passengers in 1st class

female_first_class = df[(df['sex'] == 'female') & (df['class'] == 'First')]

# Compute: Survival rate for this group
survival_rate = female_first_class['survived'].mean()
print(survival_rate)


# Compare with overall survival rate
overall_survival_rate = df['survived'].mean()
print(overall_survival_rate)

# Interpret: What insight can be used in ML feature engineering

