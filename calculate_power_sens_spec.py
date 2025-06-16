# | Test # | Method / Function                                      | Test Type / Model                | Input Interpretation                                                              | Key Assumptions                                                                                          | Strengths & Notes                                                                                    | Output Use                                               |
# | ------ | ------------------------------------------------------ | -------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
# | **1**  | `NormalIndPower` (from `statsmodels`)                  | Two-sample z-test power approx   | Effect size: Cohen's h; Sample size = n in 1 group (assumed equal n for 2 groups) | Assumes two independent groups with equal n; uses arcsin transform (Cohen's h) for effect size           | Widely used, good approximation for moderate to large samples; easy to solve for power or n          | Power for difference between two proportions             |
# | **2**  | `GofChisquarePower` (from `statsmodels`)               | Goodness-of-fit test power       | Effect size: Cohen's h; n\_bins=2 (binary outcome)                                | Chi-square approximation for goodness-of-fit with 2 bins; tests if observed proportions differ from null | Useful for testing fit to known distribution; less common for direct 2-proportion comparison         | Power for goodness-of-fit test (2 categories)            |
# | **3**  | `power_proportions_ztest` (from `statsmodels`)         | Two-sample proportions z-test    | Inputs: prop1, prop2, nobs = per group sample size                                | Tests difference between two proportions with equal sample size per group (two independent samples)      | Designed for two independent groups; includes built-in power and sample size calculations            | Power for difference between two independent proportions |
# | **4**  | Manual normal approximation (using `scipy.stats.norm`) | One-sample z-test for proportion | Single sample size n; Null proportion p0; alternative p1                          | One-sided test; normal approximation to binomial for one sample                                          | Clear interpretation for one-sample test; no dependency on two groups or effect size transformations | Power for one-sample proportion test against fixed null  |

# ------------
# TEST 1
from statsmodels.stats.power import NormalIndPower
import numpy as np

# Parameters
alpha = 0.025         # One-sided significance level
power_analysis = NormalIndPower()

p1 = 0.80             # Null hypothesis hypothesis (e.g., tolerable sensitivity/specificity)
p2 = 0.90            # Alternative hypothesis (what you're trying to detect)
n = 94                # Sample size (number of relevant test samples)

# Compute effect size (Cohen's h)
def cohen_h(p1, p2):
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

effect_size = cohen_h(p2, p1)  # note: use p2 vs. p1 for directionality

# Calculate achieved power
power = power_analysis.power(effect_size=effect_size, nobs1=n, alpha=alpha, alternative='larger')
print(f"Power to detect increase from {p1} to {p2} with n={n}, alpha={alpha}: {power:.3f}")

# # Solve for required sample size to achieve 80% power
# required_n = power_analysis.solve_power(effect_size=effect_size, power=0.80, alpha=alpha, alternative='larger')
# print(f"Required sample size for 80% power: {np.ceil(required_n)}")

# ------------
# TEST 2

from statsmodels.stats.power import GofChisquarePower
import numpy as np

# Parameters
p0 = 0.80           # Null hypothesis proportion
p1 = 0.90           # Alternative hypothesis proportion
alpha = 0.025       # One-sided significance level
n = 94              # Sample size for known test
target_power = 0.80 # Desired power level for sample size calculation

# Compute Cohen's h
def cohen_h(p1, p0):
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p0))

h = cohen_h(p1, p0)

# Initialize power analysis object
power_analysis = GofChisquarePower()

# Calculate power with n_bins=2 (binary outcome)
power = power_analysis.power(effect_size=h, nobs=n, alpha=alpha, n_bins=2)

print(f"[Power] With n={n}, power to detect increase from {p0} to {p1} at alpha={alpha}: {power:.3f}")

# Calculate required sample size for target power
required_n = power_analysis.solve_power(effect_size=h, power=target_power, alpha=alpha, n_bins=2)
print(f"[Sample Size] To achieve {target_power*100:.0f}% power, required sample size is: {np.ceil(required_n):.0f}")

# ------------
# TEST 3

from statsmodels.stats.proportion import power_proportions_ztest
import numpy as np

# Parameters
p0 = 0.80           # Null hypothesis proportion
p1 = 0.90           # Alternative proportion
alpha = 0.025       # One-sided significance level
n = 94              # Sample size for known test
target_power = 0.80 # Desired power level

# Calculate power given sample size
power = power_proportions_ztest(count=None, nobs=n, prop1=p1, prop2=p0, alpha=alpha, alternative='larger')
print(f"[Power] With n={n}, power to detect increase from {p0} to {p1}: {power:.3f}")

# Function to solve for required sample size
def solve_n_for_power(p0, p1, alpha, power, alternative='larger', tol=1e-6, max_iter=1000):
    from scipy.optimize import bisect
    
    def power_diff(n):
        return power_proportions_ztest(count=None, nobs=n, prop1=p1, prop2=p0, alpha=alpha, alternative=alternative) - power
    
    # Search between 2 and 1e6
    n_required = bisect(power_diff, 2, 1_000_000, xtol=tol, maxiter=max_iter)
    return n_required

# Calculate required sample size for target power
required_n = solve_n_for_power(p0, p1, alpha, target_power)
print(f"[Sample Size] To achieve {target_power*100:.0f}% power, required sample size is: {np.ceil(required_n):.0f}")

# ------------
# TEST 4

from scipy.stats import norm
import math

# Parameters
p0 = 0.8      # Null hypothesis proportion
p1 = 0.9      # Alternative proportion
alpha = 0.025 # Significance level (one-sided)
n = 126       # Sample size

# Standard error under null
SE0 = math.sqrt(p0 * (1 - p0) / n)

# Critical z for alpha (one-sided)
z_alpha = norm.ppf(1 - alpha)

# Z score of alternative proportion under null
z = (p1 - p0) / SE0

# Power calculation
power = 1 - norm.cdf(z_alpha - z)

print(f"Statistical power: {power:.4f}")