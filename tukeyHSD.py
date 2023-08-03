#!/usr/local/bin/python3.9

'''
perform pairwise tukey comparisons
'''

df=pd.read_csv('<path/to/csv>')

#pingouin package
import pingouin as pg
pt = pg.pairwise_tukey(data=df,dv='<response_variable>',between='grp_id')

#loop across colums and subscript p-values for each comparison (example assumes 3 groups)
var_list=df['variable', 'variable']
col_names = [col for col in var_list]
tuk_list_1_2=[]
tuk_list_1_3=[]
tuk_list_2_3=[]
for col in col_names:
    print(col)
    pt = pg.pairwise_tukey(data=df,dv=col,between='grp_id')
    print(pt)
    tuk1=pt["p-tukey"][0]; tuk_list_1_2.append(tuk1) 
    tuk2=pt["p-tukey"][1]; tuk_list_1_3.append(tuk2)
    tuk3=pt["p-tukey"][2]; tuk_list_2_3.append(tuk3)

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