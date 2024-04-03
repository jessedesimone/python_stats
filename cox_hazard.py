#!/usr/local/bin/python3.9
import pandas as pd
from lifelines import CoxPHFitter
import numpy as np

# Assuming you have your dataset in a DataFrame called 'data'
# 'Time' represents the time to event or censoring
# 'Event' represents whether the event occurred (1) or not (0)
# 'Treatment' represents the treatment group (0 for Medication A, 1 for Medication B)
# 'Covariates' represent other variables that may influence the hazard

# Generating sample data
np.random.seed(0)

# Sample data for Medication A (Treatment = 0)
data_med_a = pd.DataFrame({
    'Time': np.random.randint(10, 100, size=100),
    'Event': np.random.choice([0, 1], size=100, p=[0.8, 0.2]),  # Higher event probability for Medication A
    'Treatment': np.zeros(100),
    'Age': np.random.randint(50, 80, size=100),
    'Smoking': np.random.choice([0, 1], size=100),
    'BloodPressure': np.random.randint(120, 180, size=100),
    'Cholesterol': np.random.randint(180, 250, size=100)
})

# Sample data for Medication B (Treatment = 1)
data_med_b = pd.DataFrame({
    'Time': np.random.randint(10, 100, size=100),
    'Event': np.random.choice([0, 1], size=100),
    'Treatment': np.ones(100),
    'Age': np.random.randint(50, 80, size=100),
    'Smoking': np.random.choice([0, 1], size=100),
    'BloodPressure': np.random.randint(120, 180, size=100),
    'Cholesterol': np.random.randint(180, 250, size=100)
})

# Concatenate the two datasets
data = pd.concat([data_med_a, data_med_b], ignore_index=True)

# Display the DataFrame
print(data)

# Fit Cox proportional hazards model
cph = CoxPHFitter()
cph.fit(data, duration_col='Time', event_col='Event', formula='Treatment + Age + Smoking + BloodPressure + Cholesterol')

# Print summary of the model
print(cph.summary)
hr=round(np.exp(cph.summary.loc['Treatment', 'coef']), 4)
hr_p = (hr - 1) * 100

if cph.summary.loc['Treatment', 'coef'] > 0 and cph.summary.loc['Treatment', 'p'] < 0.05:
    print("The coefficient is positive")
    print("The hazard ratio for treatment (Medication A vs. Medication B) is {:.2f}, which is significant (p < 0.05).".format(np.exp(cph.summary.loc['Treatment', 'coef'])))
    print("This indicates that Medication A has a significantly increased hazard risk of experiencing the event compared to Medication B")
    print("The event is {:.1f}% more likely to occur in Medication A.".format(hr_p)) 
elif cph.summary.loc['Treatment', 'coef'] < 0 and cph.summary.loc['Treatment', 'p'] < 0.05:
    print("The coefficient is negative")
    print("The hazard ratio for treatment (Medication A vs. Medication B) is {:.2f}, which is significant (p < 0.05).".format(np.exp(cph.summary.loc['Treatment', 'coef'])))
    print("This indicates that Medication B has a significantly increased hazard risk of experiencing the event compared to Medication A")
    print("The event is {:.1f}% more likely to occur in Medication B.".format(hr_p))
elif cph.summary.loc['Treatment', 'p'] >= 0.05:
    print("The hazard ratio for treatment (Medication A vs. Medication B) is {:.2f}, which is not significant (p > 0.05).".format(np.exp(cph.summary.loc['Treatment', 'coef'])))


# Get hazard ratios and confidence intervals
print(cph.summary['coef'])
print(cph.confidence_intervals_)