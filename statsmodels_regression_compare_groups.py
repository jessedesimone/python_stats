#!/usr/bin/env python3

'''
module for comparing regression slops/estimates between groups
'''

# Import Packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

os.chdir('/Users/jessedesimone/Desktop')

# Load the data from Excel
df = pd.read_excel('test.xlsx', sheet_name='Sheet1')

# Ensure group var is treated as a categorical variable
df['cluster_k3'] = df['cluster_k3'].astype('category')

# Fit the regression model with interaction terms between 'X' (independent) and 'G' (group variable)
model = smf.ols('APD_prob ~ TOT_DIFF_NORM * C(cluster_k3)', data=df).fit()

# Print the summary of the model (for comparison of regression estimates)
print(model.summary())

# Initialize the plot
plt.figure(figsize=(10, 6))

# Colors and labels for each group
colors = ['red', 'blue', 'green']
labels = df['cluster_k3'].cat.categories  # Use the categories of the group variable

# Loop through each group to plot regression lines
for i, group in enumerate(df['cluster_k3'].unique()):
    group_data = df[df['cluster_k3'] == group]
    
    # Calculate predicted y-values
    y_pred = model.predict(group_data)
    
    # Convert to numpy arrays if necessary
    x_vals = group_data['TOT_DIFF_NORM'].values
    y_pred = y_pred.values  # Ensure y_pred is a numpy array
    
    # Plot regression line
    plt.plot(x_vals, y_pred, color=colors[i], label=f'{labels[i]} regression line')

# Customize the plot
plt.scatter(df['TOT_DIFF_NORM'], df['APD_prob'], color='grey', alpha=0.5, label='Data points', s=20)  # Optional: scatter plot of all data points
plt.xlabel('X (Independent Variable)')
plt.ylabel('y (Dependent Variable)')
plt.title('Regression Lines for Each Group')
plt.legend()
plt.grid(True)
plt.show()