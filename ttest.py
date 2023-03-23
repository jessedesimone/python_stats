import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind

#read data
df=pd.read_csv('/path/to/csv')

#subset data
df_check=df.filter(regex='grp|apoe_lab')
#df_check=df_check.drop(['ab42_40_label'], axis=1)
#df_check=df_check.drop(['cdr_lab', 'moca_lab', 'mmse_lab'], axis=1)

#describe data
df_check.groupby(['grp']).describe()

#independent samples t-test
test='suvr'
group1 = df_check[df_check['grp']=='high_risk']
group2 = df_check[df_check['grp']=='low_risk']
t_stat, p_value = ttest_ind(group1[test], group2[test])
print("T-statistic value: ", t_stat); print("P-Value: ", p_value)