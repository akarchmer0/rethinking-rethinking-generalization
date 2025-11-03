"""Statistical testing utilities."""

import numpy as np
from scipy import stats
from typing import Callable, Dict, List, Tuple, Any
import warnings


def bootstrap_confidence_intervals(metric_function: Callable, data: Any,
                                   n_bootstrap: int = 1000,
                                   confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for a metric.
    
    Args:
        metric_function: Function that computes the metric on data
        data: Data to bootstrap (can be list, array, etc.)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0 to 1)
    
    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    # Compute point estimate
    point_estimate = metric_function(data)
    
    # Bootstrap resampling
    bootstrap_estimates = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, n, replace=True)
        if isinstance(data, np.ndarray):
            sample = data[indices]
        else:
            sample = [data[i] for i in indices]
        
        estimate = metric_function(sample)
        bootstrap_estimates.append(estimate)
    
    # Compute confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound = np.percentile(bootstrap_estimates, upper_percentile)
    
    return point_estimate, lower_bound, upper_bound


def hypothesis_test_smoothness(smoothness1: List[float], smoothness2: List[float],
                               test_type: str = 'ttest') -> Dict[str, float]:
    """
    Test if two groups differ significantly in smoothness.
    
    Args:
        smoothness1: Smoothness measurements for group 1
        smoothness2: Smoothness measurements for group 2
        test_type: Type of test ('ttest', 'mannwhitney', or 'permutation')
    
    Returns:
        Dictionary with test statistic and p-value
    """
    if test_type == 'ttest':
        # Independent samples t-test
        statistic, p_value = stats.ttest_ind(smoothness1, smoothness2)
        test_name = "t-test"
    
    elif test_type == 'mannwhitney':
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(smoothness1, smoothness2, 
                                                alternative='two-sided')
        test_name = "Mann-Whitney U"
    
    elif test_type == 'permutation':
        # Permutation test
        statistic, p_value = permutation_test(smoothness1, smoothness2)
        test_name = "Permutation"
    
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    return {
        'test': test_name,
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }


def permutation_test(group1: List[float], group2: List[float],
                    n_permutations: int = 10000) -> Tuple[float, float]:
    """
    Perform permutation test.
    
    Args:
        group1: First group of measurements
        group2: Second group of measurements
        n_permutations: Number of permutations
    
    Returns:
        Tuple of (test_statistic, p_value)
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    # Observed difference in means
    observed_diff = np.abs(np.mean(group1) - np.mean(group2))
    
    # Combine groups
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    # Permutation test
    count = 0
    for _ in range(n_permutations):
        # Shuffle and split
        np.random.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        
        # Compute difference
        perm_diff = np.abs(np.mean(perm_group1) - np.mean(perm_group2))
        
        if perm_diff >= observed_diff:
            count += 1
    
    p_value = count / n_permutations
    
    return observed_diff, p_value


def correlation_analysis(x: np.ndarray, y: np.ndarray,
                        method: str = 'spearman') -> Dict[str, float]:
    """
    Compute correlation with significance test.
    
    Args:
        x: First variable
        y: Second variable
        method: Correlation method ('pearson' or 'spearman')
    
    Returns:
        Dictionary with correlation coefficient and p-value
    """
    x = np.array(x)
    y = np.array(y)
    
    if method == 'pearson':
        corr, p_value = stats.pearsonr(x, y)
    elif method == 'spearman':
        corr, p_value = stats.spearmanr(x, y)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    return {
        'method': method,
        'correlation': float(corr),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }


def effect_size_cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        group1: First group of measurements
        group2: Second group of measurements
    
    Returns:
        Cohen's d
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return float(d)


def multiple_comparison_correction(p_values: List[float], 
                                   method: str = 'bonferroni') -> List[float]:
    """
    Correct p-values for multiple comparisons.
    
    Args:
        p_values: List of p-values
        method: Correction method ('bonferroni' or 'holm')
    
    Returns:
        Corrected p-values
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    if method == 'bonferroni':
        # Bonferroni correction
        corrected = p_values * n
        corrected = np.minimum(corrected, 1.0)
    
    elif method == 'holm':
        # Holm-Bonferroni correction
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        corrected = np.zeros_like(p_values)
        for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p_values)):
            corrected[idx] = min(p * (n - i), 1.0)
    
    else:
        raise ValueError(f"Unknown correction method: {method}")
    
    return corrected.tolist()


def anova_test(groups: List[List[float]]) -> Dict[str, float]:
    """
    Perform one-way ANOVA test.
    
    Args:
        groups: List of groups to compare
    
    Returns:
        Dictionary with F-statistic and p-value
    """
    f_stat, p_value = stats.f_oneway(*groups)
    
    return {
        'test': 'One-way ANOVA',
        'f_statistic': float(f_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }


def kruskal_wallis_test(groups: List[List[float]]) -> Dict[str, float]:
    """
    Perform Kruskal-Wallis test (non-parametric alternative to ANOVA).
    
    Args:
        groups: List of groups to compare
    
    Returns:
        Dictionary with H-statistic and p-value
    """
    h_stat, p_value = stats.kruskal(*groups)
    
    return {
        'test': 'Kruskal-Wallis',
        'h_statistic': float(h_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }


def compute_summary_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive summary statistics.
    
    Args:
        data: Array of measurements
    
    Returns:
        Dictionary of summary statistics
    """
    data = np.array(data)
    
    return {
        'mean': float(np.mean(data)),
        'median': float(np.median(data)),
        'std': float(np.std(data, ddof=1)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75)),
        'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data))
    }

