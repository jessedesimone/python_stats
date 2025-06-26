import math
from scipy.stats import norm

def cohen_h(p1, p2):
    """Calculate Cohen's h effect size for two proportions."""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))

def calculate_power(test_type, n, alpha=0.05, effect_size=None, p1=None, p2=None, paired=False, use_cohen_h=False):
    """
    Calculate statistical power given sample size for clinical trials.
    
    Parameters:
        test_type (str): Type of test - 'means' or 'proportions'.
        n (int): Sample size per group (or total for paired design).
        alpha (float): Significance level (default 0.05).
        effect_size (float): Cohen's d (for 'means') or difference in proportions (for 'proportions').
        p1 (float): Proportion in group 1 (for 'proportions' test).
        p2 (float): Proportion in group 2 (for 'proportions' test).
        paired (bool): Whether design is paired (for 'means').
        use_cohen_h (bool): Whether to use Cohen's h for proportions.
    
    Returns:
        power (float): Statistical power (between 0 and 1).
    """
    z_alpha = norm.ppf(1 - alpha / 2)

    if test_type == 'means':
        if effect_size is None:
            raise ValueError("For 'means' test, effect_size (Cohen's d) must be provided.")

        if paired:
            z_beta = effect_size * math.sqrt(n) - z_alpha
        else:
            z_beta = effect_size * math.sqrt(n / 2) - z_alpha

        power = norm.cdf(z_beta)

    elif test_type == 'proportions':
        if p1 is None or p2 is None:
            raise ValueError("For 'proportions' test, p1 and p2 must be provided.")

        if use_cohen_h:
            h = abs(cohen_h(p1, p2))
            z_beta = h * math.sqrt(n / 2) - z_alpha
            power = norm.cdf(z_beta)
        else:
            # Raw difference case
            p_avg = (p1 + p2) / 2
            effect_size = abs(p1 - p2)
            sd1 = math.sqrt(2 * p_avg * (1 - p_avg))
            sd2 = math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
            
            numerator = effect_size * math.sqrt(n / 2)
            denominator = z_alpha * sd1

            z_beta = (numerator - denominator) / sd2
            power = norm.cdf(z_beta)
    else:
        raise ValueError("Invalid test_type. Choose 'means' or 'proportions'.")

    return power

# Example usage:
if __name__ == "__main__":
    # # Means test example: sample size = 64 per group, effect size d=0.5
    # power_means = calculate_power(test_type='means', n=64, alpha=0.05, effect_size=0.5, paired=False)
    # print(f"Power for means test: {power_means:.3f}")

    # Proportions test example: sample size = 398 per group, p1=0.8, p2=0.9, raw difference
    power_props_raw = calculate_power(test_type='proportions', n=750, alpha=0.05, p1=0.75, p2=0.9, use_cohen_h=False)
    print(f"Power for proportions test (raw difference): {power_props_raw:.3f}")

    # # Proportions test example with Cohen's h
    # power_props_cohen = calculate_power(test_type='proportions', n=750, alpha=0.05, p1=0.8, p2=0.9, use_cohen_h=True)
    # print(f"Power for proportions test (Cohen's h): {power_props_cohen:.3f}")

    # # Paired means test: total sample size 30, effect size 0.3
    # power_paired = calculate_power(test_type='means', n=30, alpha=0.05, effect_size=0.3, paired=True)
    # print(f"Power for paired means test: {power_paired:.3f}")
