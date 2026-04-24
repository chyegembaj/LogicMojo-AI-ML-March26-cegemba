# Q15. Business + ML Insight Problem (Tips Dataset)

# Context: Convert data analysis into decision-making

import seaborn as sns
df = sns.load_dataset("tips")

# Tasks:

# Create: tip_percentage

df['tip_percentage'] = (df['tip']/ df['total_bill']) * 100
#print(df.head())

# Group by: day and time
group_day_time = df.groupby(['day', 'time'])

# Identify: Highest revenue segment
revenue = group_day_time['total_bill'].sum()
highest_revenue = revenue.idxmax()
print(highest_revenue)

# Provide: Recommendation for business strategy How ML model can use this insight
# This insight helps the business focus resources on high-revenue time segments while 
# ML models use it to predict demand, segment customers, and optimize revenue strategies.