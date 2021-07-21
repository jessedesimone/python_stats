#!/usr/local/bin/python3.9
'''
Module for descriptive statistics

Change Log
==========
0.0.1 (2021-04-08)
----------
Initial commit
'''
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, skewnorm

# Generate random normal sample
data = np.random.normal(0, 10, 1000)

# Probability density function
f, ax1 = plt.subplots()
ax1.hist(data, bins='auto')
ax1.set_title('probability density (random)')
plt.tight_layout()

# Plot distribution
plt.figure(figsize=(9,5))
ax=sns.distplot(data, kde=False)
#create vertical line on mean
plt.axvline(np.mean(data), color="black", label="mean")
#create vertical lines to depict empirical distribution
#1 SD left and right of mean (68% of data)
plt.axvline(np.mean(data) + np.std(data), color = "blue", linestyle="dashed", label="+/-68%")
plt.axvline(np.mean(data) - np.std(data), color = "blue", linestyle="dashed")
#2 SD left and right of mean (95% of data)
plt.axvline(np.mean(data) + (np.std(data) * 2), color = "red", linestyle="dashed", label="+/-95%")
plt.axvline(np.mean(data) - (np.std(data) * 2), color = "red", linestyle="dashed")
#3 SD left and right of mean (99.7% of data)
plt.axvline(np.mean(data) + (np.std(data) * 3), color = "green", linestyle="dashed", label="+/-97.9%")
plt.axvline(np.mean(data) - (np.std(data) * 3), color = "green", linestyle="dashed")
plt.legend()
plt.title("Histogram of observations")
plt.xlabel("Observed value")
#plt.ylabel("KDE")                      #use if kde=True
plt.ylabel("Observation frequency")     #use if kde=False


# Descriptive statistics
np.mean(data)       #mean
np.median(data)     #median
stats.mode(data)    #mode
np.std(data)        #SD
stats.sem(data)     #SEM
np.var(data)        #variance
max(data)-min(data)     #range
