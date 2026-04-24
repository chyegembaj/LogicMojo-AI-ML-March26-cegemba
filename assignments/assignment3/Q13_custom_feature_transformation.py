# Q13. Custom Feature Transformation using apply()

# Context: Real ML requires custom logic transformation.

# Dataset: Titanic
import seaborn as sns
df = sns.load_dataset('titanic')
print(df['who'])


# Tasks:

# Create age_group: Child / Adult / Senior
def age_group(age):
    if age < 18:
        return 'Child'
    elif age < 60:
        return 'Adult'
    else:
        return 'Senior'
    

# Use .apply()
df['age_group'] = df['age'].apply(age_group)

# Compute survival rate per group
survival = df.groupby('age_group')['survived'].mean()
print(survival)

# Interpret: Which segment has highest survival likelihood
#The Child segment has highest survival likelihood
