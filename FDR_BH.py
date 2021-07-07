#!/usr/local/bin/python3.9
"""
Copyright (C) 2021 Jesse DeSimone, Ph.D.

Change Log
=============
0.0.1 (2021-04-09)
-------------
Initial commit

Usage:
    user input {int} number of uncorrected p-values to be entered
    user input {float} uncorrected pvalues

Output:
    dataframe with uncorrected and correcte values
    FDR_out.csv


Other options:

# Input list of uncorrected P-values
pv = input('\nEnter p-valus uncorrected: ')     #input string of uncorrected p-values
pvals = [float(s) for s in pv.split(',')]       #convert string to list of floats

# Convert DataFrame array to list
infile = '<path/to/file>'
df=pd.read_excel(infile)
pvals=df['pvals'].tolist()

# Use statsmodels package
import statsmodels.stats.multitest as multi
multi.multipletests(pvals, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
multitest_methods_names = {'b': 'Bonferroni',
                           's': 'Sidak',
                           'h': 'Holm',
                           'hs': 'Holm-Sidak',
                           'sh': 'Simes-Hochberg',
                           'ho': 'Hommel',
                           'fdr_bh': 'FDR Benjamini-Hochberg',
                           'fdr_by': 'FDR Benjamini-Yekutieli',
                           'fdr_tsbh': 'FDR 2-stage Benjamini-Hochberg',
                           'fdr_tsbky': 'FDR 2-stage Benjamini-Krieger-Yekutieli',
                           'fdr_gbs': 'FDR adaptive Gavrilov-Benjamini-Sarkar'
                           }

"""
# PACKAGES
import numpy as np
import pandas as pd

list_of_p = []
# Define Runner function
def number_req():
    num_of_in = int(input('Enter <int> number of uncorrected p-values: '))
    while True:
        if len(list_of_p) == num_of_in:
            break
        else:
            unc_p = float(input('Enter <float> uncorrected p-value: '))
            list_of_p.append(unc_p)

number_req()

# Define FDR correction function
def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)  #convert p to array
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

newvals = p_adjust_bh(list_of_p)    #submit list of p-values for correction
newvals.tolist()      #convert array to list

# Output results to DataFrame
d = {'Uncorrected':list_of_p, 'FDR':newvals}
df = pd.DataFrame(d)
print(df)
df.to_csv('FDR_out.csv')
