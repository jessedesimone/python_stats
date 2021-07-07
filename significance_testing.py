#!/usr/local/bin/python3.9
'''
Module for significance testing

Change Log
==========
0.0.1 (2021-04-11)
0.0.2 (2021-07-07)
----------
implemented loop option for multiple variables in dataframe
'''
import numpy as np
from scipy import stats

np.random.seed(7654567)  # fix seed to get the same result

'''
One-sample t-test
two-sided test for the null hypothesis that the expected value (mean) of a 
sample of independent observations a is equal to the given population mean, popmean
'''
rvs = stats.norm.rvs(loc=5, scale=10, size=(50))
stat, p = stats.ttest_1samp(rvs,5.0)
print(stat)
print(p)

'''
Independent-samples t-test
two-sided test for the null hypothesis that 2 independent samples have identical average 
(expected) values. This test assumes that the populations have identical variances by default.

Perform check for homogeneity of variance prior to running
i.e., Levene Test
'''
rvs1 = stats.norm.rvs(loc=5,scale=10,size=500)
rvs2 = stats.norm.rvs(loc=5,scale=10,size=500)
rvs3 = stats.norm.rvs(loc=5,scale=30,size=500)

#Equal variance
from scipy.stats import levene
levene(rvs1, rvs2)

stat, p = stats.ttest_ind(rvs1, rvs2, equal_var=True, alternative='two-sided')
print(stat, p)

# Unequal variance
levene(rvs1, rvs3)

stat, p = stats.ttest_ind(rvs1, rvs3, equal_var=False)
print(stat, p)


# Mann-Whitney U-test
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

stat, p = stats.mannwhitneyu(data_normal, data_skew, alternative='two-sided')
print(stat)
print('%.5f' % p)
results = stats.mannwhitneyu(data_normal, data_skew, alternative='two-sided')

'''
Paired samples t-test
two-sided test for the null hypothesis that 2 related 
or repeated samples have identical average (expected) values
'''
np.random.seed(12345678)
rvs1 = stats.norm.rvs(loc=5,scale=10,size=500)
rvs2 = (stats.norm.rvs(loc=5,scale=10,size=500) + stats.norm.rvs(scale=0.2,size=500))
stats.ttest_rel(rvs1,rvs2, alternative='two-sided')

'''
Wilcox signed-rank test
Tests the null hypothesis that two related paired samples come from the same distribution. 
In particular, it tests whether the distribution of the differences x - y is symmetric about zero. 
It is a non-parametric version of the paired T-test
'''
from numpy.random import randn
from scipy.stats import wilcoxon
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
stat, p = wilcoxon(data1, data2)
print('Statistic=%.3f, p=%.3f' % (stat, p))


# Code for interpretation
alpha = 0.05
if p > alpha:
	print('Fail to reject H0')
else:
	print('Reject H0')



'''
Loop through columns in dataframe
Interpret Hypothesis
FDR correction for multiple p-values

2 options:
1. parametric (normallly distributed) variables - run t-test
2. non-parametric (not normally distributed variables) - run Wilcox test
'''
# One-sample t-test (for parametric data)
col_names_par = ['', '', '']
true_mu = 0

pvals_par = []
for i in col_names_par:
	print('\n')
	w, p = stats.ttest_1samp(df[i], true_mu)
	print(i)
	print(w, p)
	pvals_par.append(p)
	alpha = 0.05
	if p > alpha:
		print('Fail to reject H0\nNot statistically different from zero')
	else:
		print('Reject H0\n*****Statistically different from zero*****')

# FDR correction for parametric p-values
multi.multipletests(pvals_par, alpha=0.05, method='fdr_bh',
					is_sorted=False, returnsorted=False)

# Wilcox test (for non-parametric data)
# col_names_npar
col_names_npar = ['', '', '']

pvals_npar = []
for i in col_names_npar:
	print('\n')
	w, p = wilcoxon(df[i])
	print(i)
	print(w, p)
	pvals_npar.append(p)
	alpha = 0.05
	if p > alpha:
		print('Fail to reject H0\nNot statistically different from zero')
	else:
		print('Reject H0\n*****Statistically different from zero*****')

# FDR correction for non-parametric p-values
multi.multipletests(pvals_npar, alpha=0.05, method='fdr_bh',
					is_sorted=False, returnsorted=False)