import math
from scipy.stats import norm

def cohen_h(p1, p2):
    """Calculate Cohen's h effect size for two proportions."""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))

def calculate_sample_size(test_type, alpha=0.05, power=0.8, effect_size=None, p1=None, p2=None, paired=False, use_cohen_h=False):
    """
    Calculate sample size for clinical trials.
    
    Parameters:
        test_type (str): Type of test - 'means' or 'proportions'.
        alpha (float): Significance level (default is 0.05).
        power (float): Statistical power (default is 0.8).
        effect_size (float): Cohen's d (for 'means' test) or difference in proportions (for 'proportions' test).
        p1 (float): Proportion in group 1 (for 'proportions' test).
        p2 (float): Proportion in group 2 (for 'proportions' test).
        paired (bool): Whether the design is paired (for 'means' test).
    
    Returns:
        n (float): Required sample size per group (for two-sample tests) or total sample size (for paired/one-sample tests).
    """
    z_alpha = norm.ppf(1 - alpha / 2)  # Z-value for significance level
    z_beta = norm.ppf(power)            # Z-value for power
    
    if test_type == 'means':
        if effect_size is None:
            raise ValueError("For 'means' test, effect_size (Cohen's d) must be provided.")
        
        if paired:
            n = ((z_alpha + z_beta) / effect_size) ** 2
        else:
            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    
    elif test_type == 'proportions':
        if p1 is None or p2 is None:
            raise ValueError("For 'proportions' test, p1 and p2 must be provided.")
        
        if use_cohen_h:
            h = abs(cohen_h(p1, p2))
            n = 2 * ((z_alpha + z_beta) / h) ** 2
        else:
            p_avg = (p1 + p2) / 2
            effect_size = abs(p1 - p2)
            n = 2 * ((z_alpha * math.sqrt(2 * p_avg * (1 - p_avg)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / effect_size) ** 2
    
    else:
        raise ValueError("Invalid test_type. Choose 'means' or 'proportions'.")
    
    return math.ceil(n)

# Example usage:
if __name__ == "__main__":
    # Two-sample t-test for means using Cohen's d
    n_means = calculate_sample_size(test_type='means', alpha=0.05, power=0.8, effect_size=0.5, paired=False)
    print(f"Sample size per group for means comparison: {n_means}")
    
    # Two-sample comparison of proportions
    # Using raw difference
    n_raw = calculate_sample_size(test_type='proportions', p1=0.80, p2=0.90, alpha=0.05, use_cohen_h=False)
    print(f"Sample size per group for proportions (raw difference): {n_raw}")
    
    # Using Cohen's h
    n_cohen = calculate_sample_size(test_type='proportions', p1=0.80, p2=0.90, alpha = 0.05, use_cohen_h=True)
    print(f"Sample size per group for proportions (Cohen's h): {n_cohen}")
    
    # Paired t-test using Cohen's d
    n_paired = calculate_sample_size(test_type='means', alpha=0.05, power=0.8, effect_size=0.3, paired=True)
    print(f"Total sample size for paired design: {n_paired}")
    
    
