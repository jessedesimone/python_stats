import pandas as pd
import os

os.chdir('/Users/jessedesimone/Desktop')
df = pd.read_excel('Book3.xlsx', sheet_name='6SITE_TEST')
df.info()

# # Replace missing values with the median within each group
# for col in ['MOCA']:
#     df[col] = df.groupby('FINAL_DX')[col].transform(lambda x: x.fillna(x.median()))

# df.info()

# # df.to_excel('fill_median.xlsx')

# Group by DX and calculate required stats
summary = df.groupby('FINAL_DX').agg(
    COUNT=('SUBJ_ID', 'count'),
    AGE_mean=('AGE', 'mean'),
    AGE_std=('AGE', 'std'),
    SEX_BIN_counts=('SEX_BIN', lambda x: x.value_counts().to_dict()),
    DUR_mean=('DX_DUR_Y', 'mean'),
    DUR_std=('DX_DUR_Y', 'std'),
    UPDRS_mean=('UPDRS_III', 'mean'),
    UPDRS_std=('UPDRS_III', 'std'),
    MOCA_mean=('MOCA', 'mean'),
    MOCA_std=('MOCA', 'std')    
).reset_index()

# Round numerical results for better readability
summary = summary.round(2)
print(summary)
summary.to_excel('dx_summary.xlsx', index=False)