# Q8. Group-Based Insights (Customer Segmentation Logic)

# Context: ML models often depend on aggregated patterns.

# Dataset: Tips
import seaborn as sns
df = sns.load_dataset("tips")



# Tasks:

# 1. Compute:
# Average tip by day
df.groupby("day")['tip'].mean()

# Total bill by gender
total_bill_by_gender = df.groupby('sex')['total_bill'].sum()
print(total_bill_by_gender )

# 2. Create new feature:
# tip_percentage
df['tip_percentage'] = (df['tip'] / df['total_bill']) * 100
print(df.head())

# 3. Identify:
# Which day has highest tipping behavior
highest_tipping_day = df.groupby('day')['tip_percentage'].mean()
print(highest_tipping_day )

# Friday has highest tipping behaviour

# 4. Explain:
# How this helps in segmentation models
#Group-based insights help segmentation models by turning raw data into stable,
#behavior focused patterns that make it easier to identify and act on meaningfull customer groups