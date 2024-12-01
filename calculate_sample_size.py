import math
from scipy.stats import norm

def calculate_sample_size(test_type, alpha=0.05, power=0.8, effect_size=None, p1=None, p2=None, paired=False):
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
    z_beta = norm.ppf(power)          # Z-value for power
    
    if test_type == 'means':
        if effect_size is None:
            raise ValueError("For 'means' test, effect_size (Cohen's d) must be provided.")
        
        if paired:
            # Sample size for paired design (one sample)
            n = ((z_alpha + z_beta) / effect_size) ** 2
        else:
            # Sample size for two-sample comparison
            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    
    elif test_type == 'proportions':
        if p1 is None or p2 is None:
            raise ValueError("For 'proportions' test, both p1 and p2 must be provided.")
        
        p_avg = (p1 + p2) / 2  # Average proportion
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
    n_props = calculate_sample_size(test_type='proportions', alpha=0.05, power=0.8, p1=0.6, p2=0.8)
    print(f"Sample size per group for proportions comparison: {n_props}")
    
    # Paired t-test using Cohen's d
    n_paired = calculate_sample_size(test_type='means', alpha=0.05, power=0.8, effect_size=0.3, paired=True)
    print(f"Total sample size for paired design: {n_paired}")
