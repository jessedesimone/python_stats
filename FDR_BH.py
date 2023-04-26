#!/usr/local/bin/python3.9

'''
Option 1
run the number_req() function and manually input all uncorrected p-values
then run p_adj_bh() function to perform FDR corrected for uncorrected p-values

Option 2
create csv file with 2 columns
column 1 is a list of variable names with column name 'variables'
column 2 is the associated uncorrected p-value for each variable with column name 'p_unc'
run function

'''
# PACKAGES
import numpy as np
import pandas as pd
import statsmodels as sm


#------------------------- option 1 -------------------------

# Define Runner function
list_of_p = []
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
df.to_csv('/Users/jessedesimone/Desktop/FDR_out.csv')

#------------------------- option 2 -------------------------
df = pd.read_csv('/path/to/pvals.csv')
var_list=df['variable'].tolist(); var_list
p_unc_list=df['p_unc'].tolist(); p_unc_list

a=sm.stats.multitest.fdrcorrection(p_unc_list, 
                                          alpha=0.05, 
                                          method='indep', 
                                          is_sorted=False)
fdr_list=a[1].tolist()

'''create dataframe for statistics and uncorrected pvalues'''
d = {'variable':var_list, 'p_unc':p_unc_list, 'fdr':fdr_list}
df_f = pd.DataFrame(d)
print(df_f)
df_f.to_csv('/path/to/fdr_out.csv')