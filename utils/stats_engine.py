"""
utils/stats_engine.py
Core statistical functions for A/B test analysis.
Covers: two-proportion z-test, Mann-Whitney U, CUPED variance reduction,
        sequential testing, and Bayesian posterior estimation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, Any


def two_proportion_ztest(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Two-proportion z-test for conversion rate comparison.

    Returns:
        dict with z_stat, p_value, confidence interval, significance, lift
    """
    p_c = control_conversions / control_n
    p_t = treatment_conversions / treatment_n

    pooled_p = (control_conversions + treatment_conversions) / (control_n + treatment_n)
    se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / control_n + 1 / treatment_n))

    z_stat = (p_t - p_c) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    ci_low  = (p_t - p_c) - z_alpha * se
    ci_high = (p_t - p_c) + z_alpha * se

    return {
        "control_rate":    round(p_c * 100, 3),
        "treatment_rate":  round(p_t * 100, 3),
        "absolute_lift":   round((p_t - p_c) * 100, 3),
        "relative_lift":   round((p_t - p_c) / p_c * 100, 2) if p_c > 0 else 0,
        "z_statistic":     round(z_stat, 4),
        "p_value":         round(p_value, 6),
        "ci_low":          round(ci_low * 100, 3),
        "ci_high":         round(ci_high * 100, 3),
        "is_significant":  p_value < alpha,
        "alpha":           alpha,
    }


def mann_whitney_revenue(
    control_revenue: np.ndarray,
    treatment_revenue: np.ndarray
) -> Dict[str, Any]:
    """
    Mann-Whitney U test for revenue distributions (non-parametric,
    handles zero-inflated distributions from non-converters).
    """
    u_stat, p_value = stats.mannwhitneyu(
        treatment_revenue, control_revenue, alternative='greater'
    )
    return {
        "u_statistic":        round(u_stat, 2),
        "p_value":            round(p_value, 6),
        "is_significant":     p_value < 0.05,
        "control_median_rev": round(np.median(control_revenue), 2),
        "treatment_median_rev": round(np.median(treatment_revenue), 2),
        "control_mean_rev":   round(np.mean(control_revenue), 2),
        "treatment_mean_rev": round(np.mean(treatment_revenue), 2),
    }


def cuped_adjustment(
    control_metric: np.ndarray,
    treatment_metric: np.ndarray,
    control_covariate: np.ndarray,
    treatment_covariate: np.ndarray
) -> Dict[str, Any]:
    """
    CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.
    Adjusts metric using pre-experiment covariate to reduce noise.

    Theta = Cov(Y, X) / Var(X)
    Y_cuped = Y - theta * (X - E[X])
    """
    X_mean = np.mean(np.concatenate([control_covariate, treatment_covariate]))

    theta_c = np.cov(control_metric, control_covariate)[0, 1] / np.var(control_covariate)
    theta_t = np.cov(treatment_metric, treatment_covariate)[0, 1] / np.var(treatment_covariate)

    c_cuped = control_metric   - theta_c * (control_covariate   - X_mean)
    t_cuped = treatment_metric - theta_t * (treatment_covariate - X_mean)

    raw_lift   = (treatment_metric.mean() - control_metric.mean()) / control_metric.mean() * 100
    cuped_lift = (t_cuped.mean() - c_cuped.mean()) / c_cuped.mean() * 100

    var_reduction = (1 - np.var(t_cuped - c_cuped) / np.var(treatment_metric - control_metric)) * 100

    return {
        "raw_lift_pct":           round(raw_lift, 2),
        "cuped_lift_pct":         round(cuped_lift, 2),
        "variance_reduction_pct": round(var_reduction, 1),
        "theta_control":          round(theta_c, 4),
        "theta_treatment":        round(theta_t, 4),
    }


def sample_size_calculator(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> Dict[str, Any]:
    """
    Calculate required sample size per variant.

    baseline_rate: current conversion rate (0-1)
    minimum_detectable_effect: smallest lift worth detecting (e.g. 0.01 = 1pp)
    """
    treatment_rate = baseline_rate + minimum_detectable_effect
    pooled = (baseline_rate + treatment_rate) / 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)

    n = (
        (z_alpha * np.sqrt(2 * pooled * (1 - pooled)) +
         z_beta  * np.sqrt(baseline_rate * (1 - baseline_rate) +
                           treatment_rate * (1 - treatment_rate))) ** 2
        / (minimum_detectable_effect ** 2)
    )

    return {
        "required_n_per_variant": int(np.ceil(n)),
        "total_required_n":       int(np.ceil(n * 2)),
        "baseline_rate_pct":      round(baseline_rate * 100, 2),
        "mde_pct":                round(minimum_detectable_effect * 100, 2),
        "alpha":                  alpha,
        "power":                  power,
    }


def bayesian_ab(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    n_samples: int = 100_000
) -> Dict[str, Any]:
    """
    Bayesian A/B test using Beta-Binomial model.
    Prior: Beta(1, 1) = uniform (non-informative)
    Posterior: Beta(alpha + conversions, beta + non-conversions)
    """
    prior_alpha, prior_beta = 1, 1

    c_posterior = np.random.beta(
        prior_alpha + control_conversions,
        prior_beta  + (control_n - control_conversions),
        n_samples
    )
    t_posterior = np.random.beta(
        prior_alpha + treatment_conversions,
        prior_beta  + (treatment_n - treatment_conversions),
        n_samples
    )

    prob_treatment_wins = (t_posterior > c_posterior).mean()
    expected_lift       = ((t_posterior - c_posterior) / c_posterior).mean() * 100

    return {
        "prob_treatment_wins":    round(prob_treatment_wins * 100, 2),
        "expected_lift_pct":      round(expected_lift, 2),
        "posterior_control_mean": round(c_posterior.mean() * 100, 3),
        "posterior_treatment_mean": round(t_posterior.mean() * 100, 3),
        "credible_interval_95_low":  round(np.percentile(t_posterior - c_posterior, 2.5) * 100, 3),
        "credible_interval_95_high": round(np.percentile(t_posterior - c_posterior, 97.5) * 100, 3),
    }


def run_full_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Run the complete A/B test analysis pipeline on a DataFrame."""
    control   = df[df['group'] == 'control']
    treatment = df[df['group'] == 'treatment']

    freq = two_proportion_ztest(
        control['converted'].sum(),   len(control),
        treatment['converted'].sum(), len(treatment)
    )
    rev = mann_whitney_revenue(
        control['revenue'].values,
        treatment['revenue'].values
    )
    bayes = bayesian_ab(
        control['converted'].sum(),   len(control),
        treatment['converted'].sum(), len(treatment)
    )
    size = sample_size_calculator(
        baseline_rate=control['converted'].mean(),
        minimum_detectable_effect=0.01
    )

    return {
        "frequentist": freq,
        "revenue":     rev,
        "bayesian":    bayes,
        "sample_size": size,
        "recommendation": "SHIP IT ✅" if freq["is_significant"] and freq["relative_lift"] > 0
                          else "DO NOT SHIP ❌"
    }


if __name__ == "__main__":
    df = pd.read_csv("data/ab_test_raw.csv")
    results = run_full_analysis(df)
    import json
    print(json.dumps(results, indent=2))
