import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind

#read data
df=pd.read_csv('/path/to/csv')

#subset the dataframe
df_check=df.filter(regex='grp')

#describe data
df_check.groupby(['grp']).describe()

#independent samples t-test
'''
perform for single variable
update grouping variable
'''
test='suvr'
group1 = df_check[df_check['grp']=='high_risk']
group2 = df_check[df_check['grp']=='low_risk']
t_stat, p_value = ttest_ind(group1[test], group2[test])
print("T-statistic value: ", t_stat); print("P-Value: ", p_value)

'''
perform ttest on multiple variables
update grouping variable
'''
loop_cols = df_check.loc[:, df_check.columns!='quest_dx']
col_names = [col for col in loop_cols]
for col in col_names:
    group1 = df_check[df_check['quest_dx']=='high_risk']
    group2 = df_check[df_check['quest_dx']=='low_risk']


    stat, p = shapiro(df_check[col])
    print(col)
    print('Shapiro test: Statistic=%.3f, P-Value=%.5f' % (stat, p))
    if p < 0.05:
        print('** violates normality assumption **')
    print('\n')