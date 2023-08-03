#!/usr/local/bin/python3.9

'''
Module to perform Pearson or Spearman correlations for dataframe
'''
# Import packages
import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seabnorn as sns

# Configure directories
ROOT_DIR = '<path/to/root/dir>'

# Read data
df = pd.read_csv(ROOT_DIR + '<data file>.csv')

# Correlation matrix
df.corr(method='pearson')['CO2EMISSIONS'].abs().sort_values(ascending=False)
df.corr(method='spearman')['CO2EMISSIONS'].abs().sort_values(ascending=False)

# Correlation between 2 variables
a = df['<variable 1>']
b = df['<variable 2>']

test, p = stats.pearsonr(a,b)
print('Test Statistic : %.3f' % test, 'P-value : %.3f' % p)
test, p = stats.spearmanr(a,b)
print('Test Statistic : %.3f' % test, 'P-value : %.3f' % p)

#-----------run correlations in loop-----------
'''create lists for final dataframe'''
response_var = 'GM_FW'
col_list=df.columns.to_list()
col_list.remove(response_var)
t_list = []
p_unc_list = []

loop_cols = df.loc[:, df.columns!=response_var]
col_names = [col for col in loop_cols]
for col in col_names:
    print('+++++' + col + '+++++')
    test, p = stats.spearmanr(df[response_var], df[col])
    t_list.append(test)
    p_unc_list.append(p)
    print('Test Statistic : %.3f' % test, 'P-value : %.3f' % p)
    if p < 0.05:
        print('** < 0.05 unc **')
    print('\n')

    #FDR correction for multivariate permutation tests
import statsmodels as sm
from statsmodels.stats.multitest import fdrcorrection
a=sm.stats.multitest.fdrcorrection(p_unc_list, 
                                          alpha=0.05, 
                                          method='indep', 
                                          is_sorted=False)
fdr_list=a[1].tolist()

#create a final dataframe
'''create dataframe for statistics and uncorrected pvalues'''
d = {'variable':col_list, 'corr_coef':t_list, 'p_unc':p_unc_list, 'fdr':fdr_list}
df_fdr = pd.DataFrame(d)
print(df_fdr)
df_fdr.to_csv('/Users/jessedesimone/Desktop/fdr.csv')


#-----------visualization-----------
#plot scatter
plt.scatter(data = df, x='ENGINESIZE', y='CO2EMISSIONS')

#plot scatter with line slope
x = df['ENGINESIZE']
y = df['CO2EMISSIONS']
res = stats.linregress(x, y)
plt.plot(x, y, 'o', label='Data')
plt.plot(x, res.intercept + res.slope * x, 'r', label='Fitted Line')
plt.legend()
plt.show()

#seaborn regplot
sns.set(rc={'figure.figsize': (8, 5)})
sns.set(font_scale=1)
sns.set_style("ticks")
X = df['ENGINESIZE']
Y = df['CO2EMISSIONS']
ax = sns.regplot(X, Y, ci=95, scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax.set_title('Simple Linear Regression Plot', weight='bold')

#plot heatmap
sns.set_context('paper')
f, ax = plt.subplots(figsize=(10, 7))
cor = df.corr(method='pearson')
plt.title('Feature Correlation Matrix', weight='bold', fontsize=20)
ax = sns.heatmap(cor, vmax=1, annot=True, cmap=sns.color_palette("coolwarm", 20))
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')
# plt.savefig('CorrMat.png')
plt.show()

# define correlation function (spearman or pearson)
# def corrfunc(x, y, **kws):
#     r, p = stats.spearmanr(x, y)  # use stats.pearsonr for pearson
#     ax = plt.gca()
#     ax.annotate("r = {:.2f}".format(r),
#                 xy=(.1, .9), xycoords=ax.transAxes)
#     ax.annotate("p = {:.2f}".format(p),
#                 xy=(.6, .9), xycoords=ax.transAxes)

# Pairgrid plot
# NOTE: very computationally demanding; do not perform with high predictors, use correlation matrix above
mpl.rcParams['axes.labelsize']=5
g = sns.PairGrid(df, palette=["red"], height=1, aspect=1.5)
g.map_upper(plt.scatter, s=1)
g.map_diag(sns.distplot, kde=False)
g.map_lower(sns.kdeplot, cmap="gist_heat")
g.map_lower(corrfunc)
g.map_upper(corrfunc)