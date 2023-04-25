import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind

#read data
df=pd.read_csv('/path/to/csv')

#subset the dataframe
df_check=df.filter(regex='var1|var2|var n')

#describe data
df_check.groupby(['<grouping var>']).describe()

#independent samples t-test
'''
perform for single variable
update grouping variable
'''
test='<var to test>'
group1 = df_check[df_check['<grouping var>']=='<grouping sub var>']
group2 = df_check[df_check['<grouping var>']=='<grouping sub var>']
t_stat, p_value = ttest_ind(group1[test], group2[test])
print("T-statistic value: ", t_stat); print("P-Value: ", p_value)

'''
perform ttest on multiple variables
update grouping variable
'''
#create lists for final dataframe
col_list=df_check.columns.to_list()
col_list.remove('grp')
t_list = []
p_unc_list = []

loop_cols = df_check.loc[:, df_check.columns!='<grouping var>']
col_names = [col for col in loop_cols]
for col in col_names:
    group1 = df_check[df_check['<grouping var>']=='<grouping sub var>']
    group2 = df_check[df_check['<grouping var>']=='<grouping sub var>']
    t_stat, p_value = ttest_ind(group1[col], group2[col])
    t_list.append(t_stat)
    p_unc_list.append(p_value)
    print("")
    print("*** Variable: ", col)
    print("T-statistic value: ", t_stat); print("P-Value: ", p_value)

d = {'variable':col_list, 't_stat':t_list, 'p_unc':p_unc_list}
df_f = pd.DataFrame(d)
print(df_f)