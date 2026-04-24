# Q12. Feature Selection using Statistical Understanding

# Context: Not all features improve model performance.

# Dataset: Iris
import seaborn as sns
df = sns.load_dataset('iris')

# Tasks:
# Compute correlation matrix
df = df.drop(columns=['species'])

corr_matrix = df.corr()
print(corr_matrix)



# Identify highly correlated features
#petal_length and petal_width are highly correlated

# Drop redundant features
df = df.drop(columns='petal_width')
print(df)
# Explain: Impact on overfitting and model performance
#Highly correlated features increases overfitting risk. makes models unstable.


