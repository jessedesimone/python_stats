#!/usr/local/bin/python3.9
'''
Compute the Kruskal-Wallis H-test for non-parametric independent samples
'''
#import packages
import os
import pandas as pd
from scipy import stats

#create data
group1 = [7, 14, 14, 13, 12, 9, 6, 14, 12, 8]
group2 = [15, 17, 13, 15, 15, 13, 9, 12, 10, 8]
group3 = [6, 8, 8, 9, 5, 14, 13, 8, 10, 9]

#---------single variable---------
test_stat, pval = stats.kruskal(group1, group2, group3)
print('t-stat= ', test_stat, 'p-value= ', pval)

