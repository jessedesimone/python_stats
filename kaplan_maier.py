#!/usr/bin/env python3

'''
Module for Survival Analysis via Kaplan-Maier Curve Estimation
Must define input file, and T, E, G variables

'''
# Import packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import median_survival_times

np.random.seed(42)  # For reproducibility

# Set working directory
os.chdir('/Users/jessedesimone/Desktop')

# Import data
infile = 'km_input.csv'     #set input file
df = pd.read_csv(infile)

# Define time and event variables
T = df["TIME"]        #timing variable
E = df["CDR_CDRSB_CONVERT"]       #event variable
G = df['GRP']       #grouping variable
grp_list=df['GRP'].unique().tolist()

# Initialize the Kaplan-Meier fitter
kmf = KaplanMeierFitter()

# Create empty lists for appending stats
median_list=[]
confidence_lower = []
confidence_upper = []

# Plot the survival curves for each group
plt.figure(figsize=(10, 6))
for group in grp_list:
    ix = G == group
    kmf.fit(T[ix], E[ix], label=group)
    kmf.plot_survival_function()
    #plt.xlim(0, 100)       #specify the xlimits
    '''Median Survival Time'''
    median_ = kmf.median_survival_time_     #get median survival time
    print(f'Median survival time for {group}: {median_}')
    median_list.append(median_)
    '''Confidence Interval'''
    ci = median_survival_times(kmf.confidence_interval_)       #get confidence interval
    lower_column = f'{group}_lower_0.95'
    upper_column = f'{group}_upper_0.95'
    
    if lower_column in ci.columns and upper_column in ci.columns:
        # Extract the closest time point to the median survival time from the confidence intervals
        # Find the index of the closest time point to the median
        median_time = median_
        if isinstance(median_time, (int, float)):
            # Ensure the median time is within the range of CI times
            ci_time_index = ci.index.get_loc(ci.index[np.argmin(np.abs(ci.index - median_time))])
            lower_limit = ci.iloc[ci_time_index][lower_column] if lower_column in ci.columns else None
            upper_limit = ci.iloc[ci_time_index][upper_column] if upper_column in ci.columns else None
        else:
            lower_limit = upper_limit = None
    else:
        lower_limit = upper_limit = None
    print(f'Lower Confidence Interval for {group}: {lower_limit}')
    print(f'Upper Confidence Interval for {group}: {upper_limit}')
    confidence_lower.append(lower_limit)
    confidence_upper.append(upper_limit)
'''plotting features'''    
plt.title('Kaplan-Meier Survival Curves for Groups', weight='bold')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.legend(loc='lower left')

# Get stats dataframe
d = {'variable':grp_list, 'median':median_list, 'confidence_upper':confidence_upper, 
     'confidence_lower':confidence_lower}
df_f = pd.DataFrame(d)
print(df_f)
df_f.to_csv(os.path.join('survival_stats.csv'), index=False)

# Perform the multivariate log-rank test to compare the survival curves across all groups
results = multivariate_logrank_test(T, G, E)
print(f'Multivariate log-rank test p-value: {results}')

# Add annotation with p-value to the right of the plot
plt.annotate(
    f'Log-Rank Test p-value: {results.p_value:.3f}',
    xy=(0.8, -0.1),  # Position (x, y) outside the plot
    xycoords='axes fraction',
    fontsize=8,
    ha='left',
    va='center',
    bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')
)

# Save the plot as a JPG file
plt.savefig('kaplan_maier_plot.jpg', format='jpg', dpi=300, bbox_inches='tight')

# Show the plot
plt.tight_layout()
plt.show()



# # Comparing two specific groups
# # Choose two groups to compare
# group1 = 'Q1'
# group2 = 'Q4'

# # Subset the data for the two groups
# df_subset = df[df['GRP'].isin([group1, group2])]
# T_subset = df_subset["TIME"]
# E_subset = df_subset["CDR_CDRSB_CONVERT"]
# G_subset = df_subset["GRP"]

# # Initialize the Kaplan-Meier fitter
# kmf = KaplanMeierFitter()

# # Plot the survival curves for each group
# plt.figure(figsize=(10, 6))
# for group in [group1, group2]:
#     ix = G == group
#     kmf.fit(T[ix], E[ix], label=group)
#     kmf.plot_survival_function()

# plt.title(f'Kaplan-Meier Survival Curves for {group1} and {group2}', weight='bold')
# plt.xlabel('Time')
# plt.ylabel('Survival Probability')
# plt.legend(loc='lower left')

# # Perform the log-rank test
# # Extract the time and event data for each group
# T_group1 = T_subset[G_subset == group1]
# E_group1 = E_subset[G_subset == group1]
# T_group2 = T_subset[G_subset == group2]
# E_group2 = E_subset[G_subset == group2]

# # Run the log-rank test
# results = logrank_test(T_group1, T_group2, event_observed_A=E_group1, event_observed_B=E_group2)

# # Print the p-value
# print(f'Log-Rank Test p-value: {results}')

# # Add annotation with p-value to the right of the plot
# plt.annotate(
#     f'Log-Rank Test p-value: {results.p_value:.3f}',
#     xy=(0.8, -0.1),  # Position (x, y) outside the plot
#     xycoords='axes fraction',
#     fontsize=8,
#     ha='left',
#     va='center',
#     bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')
# )

# # Save the plot as a JPG file
# plt.savefig('kaplan_maier_plot_2grp.jpg', format='jpg', dpi=300, bbox_inches='tight')

# # Show the plot
# plt.tight_layout()
# plt.show()