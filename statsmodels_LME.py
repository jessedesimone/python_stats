import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats import multitest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('/Users/jessedesimone/Desktop')

# Load data
df = pd.read_excel('Book2.xlsx')

# Ensure 'GRP' column is in your data; if not, double-check your spreadsheet
if 'GRP' not in df.columns:
    raise ValueError("Column 'GRP' not found in the DataFrame.")

# Make sure types are consistent
df['SUBJ'] = df['SUBJ'].astype(str)
df['GRP'] = df['GRP'].astype('category')
df['PREDICTION'] = df['PREDICTION'].astype('category')
df['SEX'] = df['SEX'].astype('category')

# Fit linear mixed-effects model
model = mixedlm("DLB_PROB ~ GRP + AGE", df, groups=df["SUBJ"])
result = model.fit()

# Summary of model
print(result.summary())

# Extract p-values for group coefficients only
group_pvals = result.pvalues[['GRP[T.DLB]', 'GRP[T.IBD]', 'GRP[T.PD]', 'GRP[T.RBD]']]

# Apply FDR correction
rejected, pvals_corrected = multitest.fdrcorrection(group_pvals, alpha=0.05)

# Print corrected p-values
print("FDR-corrected p-values for group comparisons:")
for param, pval in zip(group_pvals.index, pvals_corrected):
    print(f"{param}: {pval}")

# Tukey post hoc for GRP (group effect)
tukey = pairwise_tukeyhsd(endog=df['DLB_PROB'], groups=df['GRP'], alpha=0.05)
print(tukey)

# # Optional: Plot group differences - boxplit
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='GRP', y='DLB_PROB', data=df)
# sns.stripplot(x='GRP', y='DLB_PROB', data=df, color='black', alpha=0.5)
# plt.title('DLB_PROB by Group')
# plt.tight_layout()
# plt.show()

# # Optional: Plot group differences - reg plot
# plt.figure(figsize=(10, 6))
# sns.regplot(data=df, x='AGE', y='DLB_PROB', scatter_kws={'s': 100, 'alpha': 0.7}, line_kws={'color': 'red'}, label='Age')
# sns.regplot(data=df, x='SEX', y='DLB_PROB', scatter_kws={'s': 100, 'alpha': 0.7}, line_kws={'color': 'blue'}, label='Sex')
# plt.title('Regression Plot of DLB_PROB with Covariates: Age and Sex')
# plt.xlabel('Covariates (Age and Sex)')
# plt.ylabel('DLB_PROB')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()