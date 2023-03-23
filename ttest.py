import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind

df=pd.read_csv('/Users/jessedesimone/Documents/postdoc_uf/analysis/ab4240_imaging/fmriconnmap/df_pred.csv')

df_check=df.filter(regex='grp|cdr')
df_check=df_check.drop(['cdr_lab'], axis=1)

#describe data


#independent samples t-test
group1 = df_check[df_check['grp']=='high_risk']
group2 = df_check[df_check['grp']=='low_risk']
t_stat, p_value = ttest_ind(group1['cdr'], group2['cdr'])
print("T-statistic value: ", t_stat); print("P-Value: ", p_value)