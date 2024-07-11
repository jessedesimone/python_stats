#!/usr/local/bin/python3.9

#import packages
import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#----------------single variable----------------
#read data
df = pd.read_csv('/Users/jessedesimone/Desktop/test_ancova.csv')

#run ancova model
model = ols('Amygdala_R_FW ~ C(grp_id) + age + C(sex) + educ + C(apoe)', data=df).fit()
model_sum = sm.stats.anova_lm(model, typ=3)
print(model_sum)

#tukey comparisons
mc = pairwise_tukeyhsd(df['Amygdala_R_FW'],df['grp_id']); mc_results = mc.summary(); print(mc_results)

#----------------loop variables----------------
'''
perform an ancova analysis with tukey post-hoc
loop through columns of a dataframe
each colume is a separate response variable to be tested
'''

#import packages
import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#create output directory
root='/Users/jessedesimone/Desktop/test'
os.makedirs(root)

#read data
df = pd.read_csv('/Users/jessedesimone/Desktop/test_ancova.csv')        #original dataframe
df2=df.drop(['age', 'sex', 'educ', 'apoe'], axis=1)     #create a copy of dataframe without covariates

#define ancova function
def run_ancova_model():
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    col_list=df2.columns.to_list()
    col_list.remove('grp_id')
    p_unc_list = []
    loop_cols = df2.loc[:, df2.columns!='grp_id']
    col_names = [col for col in loop_cols]
    '''run model for each variable in col_names'''
    for col in col_names:
        print(col)
        model = ols('{} ~ C(grp_id) + age + C(sex)'.format(col), data=df).fit()
        print(model.summary())
        model_sum = sm.stats.anova_lm(model, typ=2)
        print(model_sum)
        p=float(model_sum["PR(>F)"][0])      #get p value for group effect !!!!!0 if using type II; 1 if using type III
        p=("{:.5f}".format(p))
        p_unc_list.append(p)
    d = {'variable':col_list, 'p_unc':p_unc_list}
    df_f = pd.DataFrame(d)
    df_f.to_csv(os.path.join(root, "aov_p_unc.csv"), index=False)

#define fdr function
'''run fdr correction for list of p values created above'''
def run_fdr_correct():
    df = pd.read_csv(root + "/aov_p_unc.csv")
    var_list=df['variable'].tolist(); var_list
    p_unc_list=df['p_unc'].tolist(); p_unc_list
    import statsmodels as sm
    a=sm.stats.multitest.fdrcorrection(p_unc_list, 
                                            alpha=0.05, 
                                            method='indep', 
                                            is_sorted=False)
    fdr_list=a[1].tolist()
    d = {'variable':var_list, 'p_unc':p_unc_list, 'fdr':fdr_list}
    df_f = pd.DataFrame(d)
    print(df_f)
    df_f.to_csv(os.path.join(root, "aov_p_fdr.csv"), index=False)

#define tukey function
def run_tukey():
    import pingouin as pg
    df1 = df
    df2 = pd.read_csv(root + "/aov_p_fdr.csv")
    var_list=df2['variable'].tolist()
    p_unc_list=df2['p_unc'].tolist()
    fdr_list=df2['fdr'].tolist()
    col_names = [col for col in var_list]
    mean_a_list=[]
    mean_b_list=[]
    mean_c_list=[]
    diff_list_1=[]
    diff_list_2=[]
    diff_list_3=[]
    tuk_list_1=[]
    tuk_list_2=[]
    tuk_list_3=[]
    for col in col_names:
        print(col)
        pt = pg.pairwise_tukey(data=df1,dv=col,between='grp_id')
        print(pt)
        mean_a=pt['mean(A)'][0]; mean_a_list.append(mean_a)
        mean_b=pt['mean(B)'][0]; mean_b_list.append(mean_b)
        mean_c=pt['mean(B)'][1]; mean_c_list.append(mean_c)  
        diff1=pt["diff"][0]; diff_list_1.append(diff1) 
        diff2=pt["diff"][1]; diff_list_2.append(diff2) 
        diff3=pt["diff"][2]; diff_list_3.append(diff3) 
        tuk1=pt["p-tukey"][0]; tuk_list_1.append(tuk1) 
        tuk2=pt["p-tukey"][1]; tuk_list_2.append(tuk2)
        tuk3=pt["p-tukey"][2]; tuk_list_3.append(tuk3)
    '''create final dataframe'''
    d = {'variable':var_list, 'p_unc':p_unc_list, 'fdr':fdr_list,
         'mean_a':mean_a_list, 'mean_b':mean_b_list, 'mean_c':mean_c_list,
        'a_vs_b_diff':diff_list_1, 'a_vs_c_diff':diff_list_2, 'b_vs_c_diff':diff_list_3,
        'a_vs_b_tukey':tuk_list_1, 'a_vs_c_tukey':tuk_list_2, 'b_vs_c_tukey':tuk_list_3}
    df_f = pd.DataFrame(d)
    print(df_f)
    df_f.to_csv(os.path.join(root, "aov_tukey_final.csv"), index=False)

run_ancova_model()
run_fdr_correct()
run_tukey()