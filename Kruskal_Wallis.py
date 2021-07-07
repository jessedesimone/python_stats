#!/usr/local/bin/python3.9
'''
Compute the Kruskal-Wallis H-test for independent samples.

The Kruskal-Wallis H-test tests the null hypothesis that the population median of all of the groups are equal.
It is a non-parametric version of ANOVA. The test works on 2 or more independent samples, which may have
different sizes. Note that rejecting the null hypothesis does not indicate which of the groups differs.
Post hoc comparisons between groups are required to determine which groups are different.

Check for normality and homogeneity of variances
Levene's test
Shapiro test

Change Log
==========
0.0.1 (2021-04-08)
----------
Initial commit
'''

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

# Check assumptions
from scipy.stats import levene
from scipy.stats import shapiro
stat, p = levene(High, Low, Placebo)
print('Statistic=%.3f, P-Value=%.5f' % (stat, p))
col_names = [col for col in df.columns]
for col in col_names:
    stat, p = shapiro(df[col])
    print('Statistic=%.3f, P-Value=%.5f' % (stat, p))

# Kruskal Wallis test
test_stat, pval = stats.kruskal(High, Low, Placebo)
print('t-stat= ', test_stat, 'p-value= ', pval)

# Mann-Whitney U Test
from scipy.stats import mannwhitneyu
U, p = mannwhitneyu(High, Placebo)
print('Statistic=%.3f, P-Value=%.5f' % (U, p))
U, p = mannwhitneyu(Low, Placebo)
print('Statistic=%.3f, P-Value=%.5f' % (U, p))
U, p = mannwhitneyu(High, Low)
print('Statistic=%.3f, P-Value=%.5f' % (U, p))