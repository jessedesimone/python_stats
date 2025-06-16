from statsmodels.stats.power import NormalIndPower
import numpy as np

# Parameters
alpha = 0.025         # One-sided significance level
power_analysis = NormalIndPower()

p1 = 0.80             # Null hypothesis hypothesis (e.g., tolerable sensitivity/specificity)
p2 = 0.90             # Alternative hypothesis (what you're trying to detect)
n = 100                # Sample size (number of relevant test samples)

# Compute effect size (Cohen's h)
def cohen_h(p1, p2):
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

effect_size = cohen_h(p2, p1)  # note: use p2 vs. p1 for directionality

# Calculate achieved power
power = power_analysis.power(effect_size=effect_size, nobs1=n, alpha=alpha, alternative='larger')
print(f"Power to detect increase from {p1} to {p2} with n={n}, alpha={alpha}: {power:.3f}")

# Solve for required sample size to achieve 80% power
required_n = power_analysis.solve_power(effect_size=effect_size, power=0.80, alpha=alpha, alternative='larger')
print(f"Required sample size for 80% power: {np.ceil(required_n)}")
