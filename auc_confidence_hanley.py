''' 
Hanley and McNeil method to calculate the standard error and confidence interval for AUC 

'''
# import packages 
import numpy as np
from scipy.stats import norm
'''
dmri_msa_psp_v_pd (MSA/PSP Probability)
dmri_psp_v_msa (PSP Probability)
dmri_pd_v_msa (PD Probability)
dmri_pd_v_psp (PD Probability)
PD=32
APD=45
MSA=13
PSP=32

'''
# define input parameters
n1 = 39  # Number of positive samples
n0 = 5  # Number of negative samples
auc= 0.974
confidence_level = 0.95

q1 = auc / (2 - auc)
q2 = 2 * auc**2 / (1 + auc)

se_auc = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0))

alpha = 1 - confidence_level
z = norm.ppf(1 - alpha / 2)

lower_bound = auc - z * se_auc
upper_bound = auc + z * se_auc

print('Lower', round(lower_bound,3))
print('Upper', round(upper_bound,3))