#!/usr/local/bin/python3.9

'''perform Mann Whitney U Test between groups'''

#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Generate some random numbers
np.random.seed(12345678)
High = np.random.normal(loc=100, scale=30.0, size=100)
Low = np.random.normal(loc=200, scale=60.0, size=100)
Placebo = np.random.normal(loc=50, scale=1.0, size=100)

# Convert to DataFrame
data = np.array([High,Low,Placebo])
df = pd.DataFrame(data=data.transpose(), columns = ['High_dose','Low_dose','Placebo'])
df.info()

# Plot boxplot
df.boxplot(column = ['High_dose','Low_dose','Placebo'], grid=False)
plt.show()

# Mann-Whitney U Test
U, p = mannwhitneyu(High, Placebo, alternative="two-sided")
print('Statistic=%.3f, P-Value=%.5f' % (U, p))
U, p = mannwhitneyu(Low, Placebo, alternative="two-sided")
print('Statistic=%.3f, P-Value=%.5f' % (U, p))
U, p = mannwhitneyu(High, Low, alternative="two-sided")
print('Statistic=%.3f, P-Value=%.5f' % (U, p))

alpha = 0.05
if p < alpha:
    print("Reject the null hypothesis. There is a statistically significant difference between the two groups.")
else:
    print("Fail to reject the null hypothesis. There is no statistically significant difference between the two groups.")


##---------------From Dataframe---------------
df = pd.read_csv('/Users/jessedesimone/Desktop/Book24.csv')
grp1=df.PD.values
grp2=df.APD.values
#grp1 = grp1[~np.isnan(grp1)] #may need if grp1 or grp2 has NaN
# Mann-Whitney U Test
U, p = mannwhitneyu(grp1, grp2, alternative="two-sided")
print('Statistic=%.3f, P-Value=%.4f' % (U, p))