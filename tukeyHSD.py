#!/usr/local/bin/python3.9

'''
perform pairwise tukey comparisons
'''

#pingouin package
import pingouin as pg
pt = pg.pairwise_tukey(data=df,dv='<response_variable>',between='grp_id')

#statsmodels package
from statsmodels.stats.multicomp import pairwise_tukeyhsd
mc = pairwise_tukeyhsd(df['<response_variable>'],df['grp_id'])
mc_results = mc.summary()
print(mc_results)

#scipy package
import numpy as np
from scipy.stats import tukey_hsd
group0 = [24.5, 23.5, 26.4, 27.1, 29.9]
group1 = [28.4, 34.2, 29.5, 32.2, 30.1]
group2 = [26.1, 28.3, 24.3, 26.2, 27.8]
res = tukey_hsd(group0, group1, group2)
print(res)