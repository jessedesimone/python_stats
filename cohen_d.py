#!/usr/bin/env python3

'''
calculate Cohen's d statistic for effect size
calculate d for each colname in dataframe
required input is grouped dataframe with individual values
each column is a different variable to be tested
create a final dataframe with mean, sd, pooled sd, cohen d for each colname
user only required to update grp_var and grp names
'''

#import packages
import os
import pandas as pd
import math

#configure directories and files
IN_DIR='/path/to/infile/directory'
OUT_DIR='/path/to/outfile/directory'
infile='<input file>.csv'
outfile='<output file>.csv'
fname=os.path.join(IN_DIR,infile)
oname=os.path.join(OUT_DIR,outfile)

#read data
df = pd.read_csv(fname)

#define group names
grp_var=''
grp1=''
grp2=''

#define cohen function
def cohen(data, grouping_var, group1, group2, column_name):
    m1=data.loc[df[grouping_var] == group1, column_name].mean()
    m2=data.loc[df[grouping_var] == group2, column_name].mean()
    m_diff=m1-m2
    s1=data.loc[df[grouping_var] == group1, column_name].std()
    s2=data.loc[df[grouping_var] == group2, column_name].std()
    n1=data[grouping_var].value_counts()[group1]
    n2=data[grouping_var].value_counts()[group2]
    s_pool=math.sqrt(((n1-1)*s1*s1+(n2-1)*s2*s2)/(n1+n2-2))
    cohen=m_diff/s_pool
    return m1,s1,n1,m2,s2,n2,m_diff,s_pool,cohen

#create lists for final dataframe
var_list=df.columns.tolist(); var_list.remove(grp_var)      #variable list
m1_list=[]
s1_list=[]
n1_list=[]
m2_list=[]
s2_list=[]
n2_list=[]
m_diff_list=[]
s_pool_list=[]
cohen_list=[]

#run cohen function for all columns in dataframe except grouping variable
df_c=df         #create dataframe copy
loop_cols = df_c.loc[:, df_c.columns!=grp_var]
col_names = [col for col in loop_cols]
for col in col_names:
    print('*** running cohen test for ', col)
    c=cohen(df, grp_var, grp1, grp2, col)
    m1=c[0]
    s1=c[1]
    n1=c[2]
    m2=c[3]
    s2=c[4]
    n2=c[5]
    m_diff=c[6]
    s_pool=c[7]
    coh=c[8]
    m1_list.append(m1)
    s1_list.append(s1)
    n1_list.append(n1)
    m2_list.append(m2)
    s2_list.append(s2)
    n2_list.append(n2)
    m_diff_list.append(m_diff)
    s_pool_list.append(s_pool)
    cohen_list.append(coh)

#create new dataframe with values for all colnames and output to csv file
d = {'variable':var_list, 'mean_1':m1_list, 'mean_2':m2_list, 'mean_diff':m_diff_list,
     'sd_1':s1_list, 'sd_2':s2_list, 'pooled_sd':s_pool_list, 'Cohen d':cohen_list}
df_f = pd.DataFrame(d)
print(df_f)
df_f.to_csv(oname, index=False)