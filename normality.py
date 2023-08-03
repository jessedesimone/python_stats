#!/usr/local/bin/python3.9
'''
Module for checking normality of data

Change Log
==========
0.0.1 (2021-04-08)
----------
Initial commit
'''
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345678)
data = np.random.normal(loc=0, scale=3.0, size=10000)

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

#Visualize data in regard to empirical CDF
plt.figure(figsize=(9,5))
ax=sns.distplot(data, kde=True)

#plot SD vertical lines and cumultive distribution horizontal lines
plt.axhline(y = 0.025, color = 'y', linestyle='-')
plt.axvline(x = np.mean(data) - (2 * np.std(data)), color = 'y', linestyle='-')
plt.axhline(y = 0.975, color = 'y', linestyle='-')
plt.axvline(x = np.mean(data) + (2 * np.std(data)), color = 'y', linestyle='-')

from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
ecdf = ECDF(data)
print(ecdf.x)
print(ecdf.y)
plt.plot(ecdf.x, ecdf.y, color="black", linestyle="dashed", label="ECDF")
plt.axhline(y = 0.025, color = 'y', linestyle='-')
plt.axvline(x = np.mean(data) - (2 * np.std(data)), color = 'y', linestyle='-')
plt.axhline(y = 0.975, color = 'y', linestyle='-')
plt.axvline(x = np.mean(data) + (2 * np.std(data)), color = 'y', linestyle='-')
plt.legend()
plt.title("Empirical distribution CDF Plot")
plt.xlabel("Observed value")
plt.ylabel("KDE")

import scipy.stats as stats
import pylab
stats.probplot(data, dist="norm", plot=pylab)
plt.title("Normal Q-Q Plot")
pylab.show()

# Shapiro test
from scipy.stats import shapiro
stat, p = shapiro(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# Levene tests
from scipy.stats import levene
stat, p = levene(data, data2)       #need min 2 samples

# Skewness
# D'Agostino-Pearson test of skewness and kurtosis
ts, p = stats.normaltest(data)
print(stats.normaltest(data))

#Shape
n = 10000
start = 0
width = 20

a = 0
data_normal = skewnorm.rvs(size=n, a=a,loc = start, scale=width)

a = 3
data_skew = skewnorm.rvs(size=n, a=a,loc = start, scale=width)

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(data_normal, bins='auto')
ax1.set_title('probability density (random)')
ax2.hist(data_skew, bins='auto')
ax2.set_title('Skewed data')
plt.tight_layout()

sig1 = data_normal
print("mean : ", np.mean(sig1))
print("var  : ", np.var(sig1))
print("skew : ", skew(sig1))
print("kurt : ", kurtosis(sig1))

sig2 = data_skew
print("mean : ", np.mean(sig2))
print("var  : ", np.var(sig2))
print("skew : ", skew(sig2))
print("kurt : ", kurtosis(sig2))

# D'Agostino-Pearson test of skewness and kurtosis
ts, p = stats.normaltest(data_skew)
print(stats.normaltest(data_skew))