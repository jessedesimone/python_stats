#!/usr/local/bin/python3.9

'''perform Mann Whitney U Test between groups'''

#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Convert to DataFrame
data = np.array([High,Low,Placebo])
df = pd.DataFrame(data=data.transpose(), columns = ['High_dose','Low_dose','Placebo'])
df.info()

# Plot boxplot
df.boxplot(column = ['High_dose','Low_dose','Placebo'], grid=False)
plt.show()

# Generate some random numbers
np.random.seed(12345678)
High = np.random.normal(loc=100, scale=30.0, size=100)
Low = np.random.normal(loc=200, scale=60.0, size=100)
Placebo = np.random.normal(loc=50, scale=1.0, size=100)

# Mann-Whitney U Test
U, p = mannwhitneyu(High, Placebo)
print('Statistic=%.3f, P-Value=%.5f' % (U, p))
U, p = mannwhitneyu(Low, Placebo)
print('Statistic=%.3f, P-Value=%.5f' % (U, p))
U, p = mannwhitneyu(High, Low)
print('Statistic=%.3f, P-Value=%.5f' % (U, p))