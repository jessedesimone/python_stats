import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/Users/jessedesimone/Desktop')

# Read dataframe
df = pd.read_excel('complete_cases.xlsx')

# Create misdiagnosis column
df['misdiagnosed'] = 1 - df['Accuracy']

z = 1.96  # 95% CI z-score

def compute_aggregate_rr(df_subset, group_col='Group', event_col='misdiagnosed'):
    """
    Compute aggregate relative risk and 95% CI (log-normal) for two groups.
    Returns (p1, p2, rr, ci_low, ci_high, group_names)
    """
    group_counts = df_subset.groupby(group_col)[event_col].agg(['sum', 'count'])
    group_names = group_counts.index.tolist()

    if len(group_names) < 2:
        return None  # Not enough groups to compare

    a = group_counts.loc[group_names[0], 'sum']
    n1 = group_counts.loc[group_names[0], 'count']
    b = group_counts.loc[group_names[1], 'sum']
    n2 = group_counts.loc[group_names[1], 'count']

    # Handle zero events to avoid division by zero or log(0)
    if a == 0 or b == 0:
        # Add continuity correction
        a += 0.5
        b += 0.5
        n1 += 1
        n2 += 1

    p1 = a / n1
    p2 = b / n2
    rr = p1 / p2

    log_rr = np.log(rr)
    se_log_rr = np.sqrt((1/a - 1/n1) + (1/b - 1/n2))

    ci_low = np.exp(log_rr - z * se_log_rr)
    ci_high = np.exp(log_rr + z * se_log_rr)

    return p1, p2, rr, ci_low, ci_high, group_names

# --- Overall aggregate RR ---

result_all = compute_aggregate_rr(df)
if result_all:
    p1, p2, rr_aggregate, ci_low, ci_high, group_names = result_all
    print(f"\n[Aggregate RR - All Cases]")
    print(f"Misdiagnosis Rate ({group_names[0]}): {p1:.3f}")
    print(f"Misdiagnosis Rate ({group_names[1]}): {p2:.3f}")
    print(f"Relative Risk ({group_names[0]} vs {group_names[1]}): {rr_aggregate:.3f} (95% CI: {ci_low:.2f}–{ci_high:.2f})")
else:
    print("Not enough groups to calculate aggregate RR for all cases.")

# --- Per-Diagnosis subtype aggregate RR ---

dx_types = df['AIDP DX'].unique()

# Store results for plotting
plot_data = {
    'Diagnosis': ['All Cases'],
    'RR': [rr_aggregate if result_all else np.nan],
    'CI_Lower': [ci_low if result_all else np.nan],
    'CI_Upper': [ci_high if result_all else np.nan]
}

for dx in dx_types:
    df_dx = df[df['AIDP DX'] == dx]
    result = compute_aggregate_rr(df_dx)
    if result:
        p1_dx, p2_dx, rr_dx, ci_low_dx, ci_high_dx, groups_dx = result
        print(f"\n===== Diagnosis: {dx} =====")
        print(f"Misdiagnosis Rate ({groups_dx[0]}): {p1_dx:.3f}")
        print(f"Misdiagnosis Rate ({groups_dx[1]}): {p2_dx:.3f}")
        print(f"Relative Risk ({groups_dx[0]} vs {groups_dx[1]}): {rr_dx:.3f} (95% CI: {ci_low_dx:.2f}–{ci_high_dx:.2f})")
        
        plot_data['Diagnosis'].append(dx)
        plot_data['RR'].append(rr_dx)
        plot_data['CI_Lower'].append(ci_low_dx)
        plot_data['CI_Upper'].append(ci_high_dx)
    else:
        print(f"\n===== Diagnosis: {dx} =====")
        print("  ⚠️ One group missing - skipping RR calculation.")
        plot_data['Diagnosis'].append(dx)
        plot_data['RR'].append(np.nan)
        plot_data['CI_Lower'].append(np.nan)
        plot_data['CI_Upper'].append(np.nan)



# --- Plot aggregate RR ---

# Example structure: plot_data from your earlier calculations
# plot_data = {
#     'Diagnosis': [...],
#     'RR': [...],
#     'CI_Lower': [...],
#     'CI_Upper': [...]
# }

diagnoses = plot_data['Diagnosis']
rr_values = plot_data['RR']

# Color dots by RR >1 or <1
colors = ['red' if rr > 1 else 'blue' for rr in rr_values]

fig, ax = plt.subplots(figsize=(8, len(diagnoses)*0.5 + 2))

y_pos = np.arange(len(diagnoses))

ax.scatter(rr_values, y_pos, color=colors, s=100, edgecolor='k', zorder=3)

# Vertical line at RR=1 (no difference)
ax.axvline(1, color='gray', linestyle='--')

ax.set_yticks(y_pos)
ax.set_yticklabels(diagnoses)
ax.invert_yaxis()  # top diagnosis on top

ax.set_xlabel('Relative Risk (%)')
ax.set_title('Relative Risk of Misdiagnosis')

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Higher risk in Clinical-only (RR > 1)',
           markerfacecolor='red', markersize=10, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='Higher risk in Clinical + AIDP (RR < 1)',
           markerfacecolor='blue', markersize=10, markeredgecolor='k')
]
ax.legend(handles=legend_elements, loc='best')

plt.tight_layout()
plt.show()
