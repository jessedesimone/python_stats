import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load the Excel file
file_path = '/Users/jessedesimone/Desktop/matched.xlsx'
df = pd.read_excel(file_path)

# Function to exclude outliers
def exclude_outliers(df, col, group_col='CLIN_CONVERT'):
    # Calculate group-wise mean and standard deviation
    group_stats = df.groupby(group_col)[col].agg(['mean', 'std']).reset_index()
    group_stats['lower_bound'] = group_stats['mean'] - 2 * group_stats['std']
    group_stats['upper_bound'] = group_stats['mean'] + 2 * group_stats['std']
    # Merge bounds with original DataFrame
    df = df.merge(group_stats[[group_col, 'lower_bound', 'upper_bound']], on=group_col)
    # Exclude outliers
    df = df[(df[col] >= df['lower_bound']) & (df[col] <= df['upper_bound'])]
    # Drop the bounds columns
    df = df.drop(columns=['lower_bound', 'upper_bound'])
    return df

# Define ANCOVA function
def run_ancova(df, col, covariates=['AGE', 'SEX', 'APOE4', 'TIME', 'CDR', 'Cohort'], group_col='CLIN_CONVERT'):
    # Define the formula for ANCOVA
    formula = f"{col} ~ {group_col} + {' + '.join(covariates)}"
    # Fit the model
    model = ols(formula, data=df).fit()
    # Get the ANCOVA table
    ancova_table = sm.stats.anova_lm(model, typ=2)
    return ancova_table

# Columns to analyze
columns_to_analyze = ['PTAU217_B','Amygdala_L_FW', 'Amygdala_R_FW', 'Angular_L_FW', 'Angular_R_FW', 'Calcarine_L_FW', 'Calcarine_R_FW', 'Caudate_L_FW', 'Caudate_R_FW', 'Cingulum_Ant_L_FW', 'Cingulum_Ant_R_FW', 'Cingulum_Mid_L_FW', 'Cingulum_Mid_R_FW', 'Cingulum_Post_L_FW', 'Cingulum_Post_R_FW', 'Cuneus_L_FW', 'Cuneus_R_FW', 'Dorsal_Mesopontine_FW', 'Entorhinal_Cortex_L_FW', 'Entorhinal_Cortex_R_FW', 'Frontal_Inf_Oper_L_FW', 'Frontal_Inf_Oper_R_FW', 'Frontal_Inf_Orb_L_FW', 'Frontal_Inf_Orb_R_FW', 'Frontal_Inf_Tri_L_FW', 'Frontal_Inf_Tri_R_FW', 'Frontal_Med_Orb_L_FW', 'Frontal_Med_Orb_R_FW', 'Frontal_Mid_L_FW', 'Frontal_Mid_Orb_L_FW', 'Frontal_Mid_Orb_R_FW', 'Frontal_Mid_R_FW', 'Frontal_Sup_L_FW', 'Frontal_Sup_Medial_L_FW', 'Frontal_Sup_Medial_R_FW', 'Frontal_Sup_Orb_L_FW', 'Frontal_Sup_Orb_R_FW', 'Frontal_Sup_R_FW', 'Fusiform_L_FW', 'Fusiform_R_FW', 'Heschl_L_FW', 'Heschl_R_FW', 'Hippocampus_L_FW', 'Hippocampus_R_FW', 'Insula_L_FW', 'Insula_R_FW', 'Lingual_L_FW', 'Lingual_R_FW', 'Occipital_Inf_L_FW', 'Occipital_Inf_R_FW', 'Occipital_Mid_L_FW', 'Occipital_Mid_R_FW', 'Occipital_Sup_L_FW', 'Occipital_Sup_R_FW', 'Pallidum_L_FW', 'Pallidum_R_FW', 'Paracentral_Lobule_L_FW', 'Paracentral_Lobule_R_FW', 'ParaHippocampal_L_FW', 'ParaHippocampal_R_FW', 'Parietal_Inf_L_FW', 'Parietal_Inf_R_FW', 'Parietal_Sup_L_FW', 'Parietal_Sup_R_FW', 'Pons_FW', 'Postcentral_L_FW', 'Postcentral_R_FW', 'Precentral_L_FW', 'Precentral_R_FW', 'Precuneus_L_FW', 'Precuneus_R_FW', 'Putamen_L_FW', 'Putamen_R_FW', 'Rectus_L_FW', 'Rectus_R_FW', 'Retrosplenial_Cortex_L_FW', 'Retrosplenial_Cortex_R_FW', 'Rolandic_Oper_L_FW', 'Rolandic_Oper_R_FW', 'Supp_Motor_Area_L_FW', 'Supp_Motor_Area_R_FW', 'SupraMarginal_L_FW', 'SupraMarginal_R_FW', 'Temporal_Inf_L_FW', 'Temporal_Inf_R_FW', 'Temporal_Mid_L_FW', 'Temporal_Mid_R_FW', 'Temporal_Pole_Mid_L_FW', 'Temporal_Pole_Mid_R_FW', 'Temporal_Pole_Sup_L_FW', 'Temporal_Pole_Sup_R_FW', 'Temporal_Sup_L_FW', 'Temporal_Sup_R_FW', 'Thalamus_L_FW', 'Thalamus_R_FW', 'Vermis_10_FW', 'Vermis_1_2_FW', 'Vermis_3_FW', 'Vermis_4_5_FW', 'Vermis_6_FW', 'Vermis_7_FW', 'Vermis_8_FW', 'Vermis_9_FW', 'Ch1_2_L_FW', 'Ch1_2_R_FW', 'Ch3_L_FW', 'Ch3_R_FW', 'Ch4_L_FW', 'Ch4p_L_FW', 'Ch4p_R_FW', 'Ch4_R_FW', 'TCATT_Angular_Gyrus_Final_FW', 'TCATT_Anterior_Orbital_Gyrus_Final_FW', 'TCATT_Calcarine_Sulcus_Final_FW', 'TCATT_Cuneus_Final_FW', 'TCATT_Gyrus_Rectus_Final_FW', 'TCATT_Inferior_Frontal_Gyrus_Pars_Opercularis_Final_FW', 'TCATT_Inferior_Frontal_Gyrus_Pars_Orbitalis_Final_FW', 'TCATT_Inferior_Frontal_Gyrus_Pars_Triangularis_Final_FW', 'TCATT_Inferior_Occipital_Final_FW', 'TCATT_Inferior_Parietal_Lobule_Final_FW', 'TCATT_Inferior_Temporal_Gyrus_Final_FW', 'TCATT_Lateral_Orbital_Gyrus_Final_FW', 'TCATT_Lingual_Gyrus_Final_FW', 'TCATT_M1_Final_FW', 'TCATT_Medial_Frontal_Gyrus_Final_FW', 'TCATT_Medial_Orbital_Gyrus_Final_FW', 'TCATT_Medial_Orbitofrontal_Gyrus_Final_FW', 'TCATT_Middle_Frontal_Gyrus_Final_FW', 'TCATT_Middle_Occipital_Final_FW', 'TCATT_Middle_Temporal_Gyrus_Final_FW', 'TCATT_Olfactory_Cortex_Final_FW', 'TCATT_Paracentral_Final_FW', 'TCATT_PMd_Final_FW', 'TCATT_PMv_Final_FW', 'TCATT_preSMA_Final_FW', 'TCATT_S1_Final_FW', 'TCATT_SMA_Final_FW', 'TCATT_Superior_Frontal_Gyrus_Final_FW', 'TCATT_Superior_Occipital_Final_FW', 'TCATT_Superior_Parietal_Lobule_Final_FW', 'TCATT_Superior_Temporal_Gyrus_Final_FW', 'TCATT_Supramarginal_Gyrus_Final_FW']

# Exclude outliers and run ANCOVA for each column
ancova_results = {}
p_values = {}
for col in columns_to_analyze:
    df_filtered = exclude_outliers(df, col)
    ancova_table = run_ancova(df_filtered, col)
    ancova_results[col] = ancova_table
    p_values[col] = ancova_table.loc['CLIN_CONVERT', 'PR(>F)']

# Save the results to an Excel file
# column headers must be less <=25 characters
# with pd.ExcelWriter('/Users/jessedesimone/Desktop/ancova_results.xlsx') as writer:
#     for col, result in ancova_results.items():
#         result.to_excel(writer, sheet_name=f'ANCOVA_{col}')

# Create DataFrame of p-values with column names and values next to each other
p_values_df = pd.DataFrame(list(p_values.items()), columns=['Column', 'p-value'])

# Save p-values to a new Excel sheet
p_values_df.to_excel('/Users/jessedesimone/Desktop/ancova_p_values.xlsx', index=False)