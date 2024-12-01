from scipy.stats import norm
import math

def calculate_power(test_type, n, alpha=0.05, effect_size=None, p1=None, p2=None, paired=False):
    """
    Calculate the statistical power of a clinical trial.
    
    Parameters:
        test_type (str): Type of test - 'means' or 'proportions'.
        n (int): Sample size per group (for two-sample tests) or total sample size (for paired/one-sample tests).
        alpha (float): Significance level (default is 0.05).
        effect_size (float): Cohen's d (for 'means' test) or difference in proportions (for 'proportions' test).
        p1 (float): Proportion in group 1 (for 'proportions' test).
        p2 (float): Proportion in group 2 (for 'proportions' test).
        paired (bool): Whether the design is paired (for 'means' test).
    
    Returns:
        power (float): Statistical power of the test.
    """
    z_alpha = norm.ppf(1 - alpha / 2)  # Z-value for significance level

    if test_type == 'means':
        if effect_size is None:
            raise ValueError("For 'means' test, effect_size (Cohen's d) must be provided.")
        
        if paired:
            # Effective sample size for paired design
            n_effective = n
        else:
            # Effective sample size for two-sample comparison
            n_effective = n / 2

        # Calculate z_beta
        z_beta = math.sqrt(n_effective) * effect_size - z_alpha
    
    elif test_type == 'proportions':
        if p1 is None or p2 is None:
            raise ValueError("For 'proportions' test, both p1 and p2 must be provided.")
        
        p_avg = (p1 + p2) / 2  # Average proportion
        effect_size = abs(p1 - p2)
        pooled_sd = math.sqrt(2 * p_avg * (1 - p_avg))
        
        # Effective sample size for two-sample comparison
        n_effective = n / 2

        # Calculate z_beta
        z_beta = (math.sqrt(n_effective) * effect_size) / pooled_sd - z_alpha
    
    else:
        raise ValueError("Invalid test_type. Choose 'means' or 'proportions'.")
    
    # Calculate power
    power = norm.cdf(z_beta)
    return power

# Example usage:
if __name__ == "__main__":
    # Calculate power for means comparison (two-sample t-test)
    power_means = calculate_power(test_type='means', n=50, alpha=0.05, effect_size=0.5, paired=False)
    print(f"Power for means comparison: {power_means:.3f}")
    
    # Calculate power for proportions comparison
    power_props = calculate_power(test_type='proportions', n=60, alpha=0.05, p1=0.6, p2=0.8)
    print(f"Power for proportions comparison: {power_props:.3f}")
    
    # Calculate power for paired t-test
    power_paired = calculate_power(test_type='means', n=30, alpha=0.05, effect_size=0.3, paired=True)
    print(f"Power for paired design: {power_paired:.3f}")
