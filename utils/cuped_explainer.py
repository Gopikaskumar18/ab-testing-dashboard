"""
utils/cuped_explainer.py
═══════════════════════════════════════════════════════════════════════════
CUPED: Controlled-experiment Using Pre-Experiment Data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Paper: Deng, A. et al. (2013) — Microsoft Research
Used by: Microsoft, Airbnb, Netflix, LinkedIn to increase experiment power

WHY CUPED?
─────────────────────────────────────────────────────────────────────────
Standard A/B tests need large sample sizes to detect small effects.
CUPED reduces the variance of the metric by subtracting out variance that
can be explained by something we already know about the user (pre-experiment
behaviour) — *before* the experiment started.

By removing this "explainable noise", what remains is a cleaner signal
from the treatment, letting us detect the same effect with fewer users
(or more confidently with the same number of users).

MATH INTUITION
─────────────────────────────────────────────────────────────────────────
Raw metric Y:    Y_i = true_effect + user_noise + random_noise

We observe X (pre-experiment metric, e.g. purchases last 30 days).
X explains some of user_noise because past behaviour predicts future behaviour.

CUPED metric:    Y_cuped_i = Y_i - θ * (X_i - E[X])

where θ = Cov(Y, X) / Var(X)   ← OLS coefficient of Y regressed on X

The term θ * (X_i - E[X]) is our best linear prediction of user_noise from X.
Subtracting it out reduces variance without biasing the treatment effect estimate,
because X is measured *before* the experiment (it can't be affected by treatment).

Variance reduction = 1 - Var(Y_cuped) / Var(Y) = ρ²   (squared correlation)

If X correlates 30% with Y → ρ² = 0.09 → 9% variance reduction (modest)
If X correlates 60% with Y → ρ² = 0.36 → 36% variance reduction (strong)
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def cuped(
    y_control: np.ndarray,
    y_treatment: np.ndarray,
    x_control: np.ndarray,
    x_treatment: np.ndarray,
    alpha: float = 0.05
) -> dict:
    """
    Apply CUPED variance reduction and return before/after comparison.

    Parameters
    ----------
    y_control    : metric values for control group
    y_treatment  : metric values for treatment group
    x_control    : pre-experiment covariate for control (e.g. purchases last 30d)
    x_treatment  : pre-experiment covariate for treatment
    alpha        : significance level

    Returns
    -------
    dict with raw results, CUPED results, and variance reduction stats
    """
    # Grand mean of covariate (pooled, so neither group is biased)
    x_grand_mean = np.concatenate([x_control, x_treatment]).mean()

    # Theta: estimated via OLS on control group only
    # Using only control avoids treatment effect contaminating theta
    theta = np.cov(y_control, x_control)[0, 1] / np.var(x_control)

    # Apply CUPED adjustment
    y_c_cuped = y_control   - theta * (x_control   - x_grand_mean)
    y_t_cuped = y_treatment - theta * (x_treatment - x_grand_mean)

    def t_test(a, b):
        """Welch's t-test for difference in means."""
        n_a, n_b   = len(a), len(b)
        mean_diff  = b.mean() - a.mean()
        pooled_se  = np.sqrt(a.var(ddof=1)/n_a + b.var(ddof=1)/n_b)
        t_stat     = mean_diff / pooled_se
        dof        = (a.var(ddof=1)/n_a + b.var(ddof=1)/n_b)**2 / (
                        (a.var(ddof=1)/n_a)**2/(n_a-1) +
                        (b.var(ddof=1)/n_b)**2/(n_b-1)
                     )
        p_val      = 2 * (1 - stats.t.cdf(abs(t_stat), df=dof))
        ci_low     = mean_diff - stats.t.ppf(1-alpha/2, dof) * pooled_se
        ci_high    = mean_diff + stats.t.ppf(1-alpha/2, dof) * pooled_se
        return {
            'mean_control':   round(a.mean(), 4),
            'mean_treatment': round(b.mean(), 4),
            'diff':           round(mean_diff, 4),
            'se':             round(pooled_se, 6),
            'ci_low':         round(ci_low, 4),
            'ci_high':        round(ci_high, 4),
            't_stat':         round(t_stat, 4),
            'p_value':        round(p_val, 8),
            'significant':    p_val < alpha
        }

    raw   = t_test(y_control, y_treatment)
    cuped_res = t_test(y_c_cuped, y_t_cuped)

    var_reduction = (1 - y_c_cuped.var() / y_control.var()) * 100
    correlation   = np.corrcoef(y_control, x_control)[0, 1]

    # Effective sample size gain from CUPED
    # Same power with fewer users: N_cuped = N_raw * (1 - rho^2)
    n_reduction_pct = correlation**2 * 100

    return {
        'raw':                   raw,
        'cuped':                 cuped_res,
        'theta':                 round(theta, 6),
        'correlation_y_x':       round(correlation, 4),
        'variance_reduction_pct': round(var_reduction, 1),
        'effective_n_saving_pct': round(n_reduction_pct, 1),
        'interpretation': (
            f"CUPED reduced metric variance by {var_reduction:.1f}% "
            f"(θ={theta:.4f}, ρ={correlation:.3f}). "
            f"Equivalent to running the experiment with {n_reduction_pct:.0f}% fewer users "
            f"for the same statistical power."
        )
    }


def run_example():
    """Reproduce the checkout experiment results with CUPED."""
    np.random.seed(42)
    df = pd.read_csv('data/ab_test_raw.csv')

    control   = df[df['group'] == 'control']
    treatment = df[df['group'] == 'treatment']

    print("=" * 60)
    print("CUPED Analysis: Checkout Optimization Experiment")
    print("=" * 60)
    print(f"Users: {len(control):,} control | {len(treatment):,} treatment")
    print()

    result = cuped(
        y_control    = control['converted'].values.astype(float),
        y_treatment  = treatment['converted'].values.astype(float),
        x_control    = control['pre_exp_purchases'].values.astype(float),
        x_treatment  = treatment['pre_exp_purchases'].values.astype(float),
    )

    print("── RAW t-test ──────────────────────────────────────")
    r = result['raw']
    print(f"  Control CVR:    {r['mean_control']*100:.2f}%")
    print(f"  Treatment CVR:  {r['mean_treatment']*100:.2f}%")
    print(f"  Lift:           {(r['mean_treatment']-r['mean_control'])*100:+.2f}pp")
    print(f"  95% CI:         [{r['ci_low']*100:.3f}pp, {r['ci_high']*100:.3f}pp]")
    print(f"  p-value:        {r['p_value']:.6f}  {'✅' if r['significant'] else '❌'}")
    print(f"  Std error:      {r['se']:.6f}")
    print()

    print("── CUPED-adjusted t-test ───────────────────────────")
    c = result['cuped']
    print(f"  Control CVR:    {c['mean_control']*100:.2f}%  (adjusted)")
    print(f"  Treatment CVR:  {c['mean_treatment']*100:.2f}%  (adjusted)")
    print(f"  Lift:           {(c['mean_treatment']-c['mean_control'])*100:+.2f}pp")
    print(f"  95% CI:         [{c['ci_low']*100:.3f}pp, {c['ci_high']*100:.3f}pp]")
    print(f"  p-value:        {c['p_value']:.6f}  {'✅' if c['significant'] else '❌'}")
    print(f"  Std error:      {c['se']:.6f}  (reduced vs raw)")
    print()

    print("── CUPED stats ─────────────────────────────────────")
    print(f"  Theta (θ):                {result['theta']}")
    print(f"  Correlation (ρ):          {result['correlation_y_x']}")
    print(f"  Variance reduction:       {result['variance_reduction_pct']}%")
    print(f"  Effective N saving:       {result['effective_n_saving_pct']}%")
    print()
    print(f"  ➤ {result['interpretation']}")
    print()
    print("NOTE: With this dataset pre_exp_purchases has low correlation")
    print("to conversion (users are randomly assigned). In production,")
    print("a 30–60% correlated covariate (e.g. past 30-day purchases)")
    print("typically yields 10–40% variance reduction.")

    return result


if __name__ == '__main__':
    run_example()
