#!/usr/local/bin/python3.9

#import packages
import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#read data
df = pd.read_csv('/Users/jessedesimone/Desktop/test_ancova.csv')

#run ancova model
model = ols('Amygdala_R_FW ~ C(grp_id) + age + C(sex) + educ + C(apoe)', data=df).fit()
model_sum = sm.stats.anova_lm(model, typ=3)
print(model_sum)

#tukey comparisons
mc = pairwise_tukeyhsd(df['Amygdala_R_FW'],df['grp_id']); mc_results = mc.summary(); print(mc_results)