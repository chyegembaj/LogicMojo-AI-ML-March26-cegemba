# Q14. Error Analysis Simulation (Model Evaluation Thinking)

# Context: Understanding prediction errors is key in ML.

# Tasks:
import numpy as np

# Generate 1000 random error values using NumPy
random_errors = np.random.randn(1000)
print(type(random_errors))

# Compute: Mean Standard deviation
mean = np.mean(random_errors)
std = np.std(random_errors)
print("mean: ", mean)
print("Standard deviation: ", std)


# Identify outliers: Using mean ± 2*std
lower_bound = mean - (2 * std)
upper_bound = mean + (2 * std)
print("lower_bound", lower_bound)
print("upper_bound ", upper_bound )

outliers = random_errors[(random_errors < lower_bound) | (random_errors > upper_bound)] 
print(outliers)


# Explain: What high variance means in model performance
# High variance means in model performance means the model's prediction are inconsistent and overly 
# sensitive to the training data.
