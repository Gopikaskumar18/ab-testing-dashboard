# Interview Prep: A/B Testing Deep Dive
# Questions FAANG interviewers ask — and the answers from this project

---

## Q1: Walk me through this experiment end to end.

**Answer:**
We ran a 30-day test on the checkout flow — 50,000 users per variant.
Control saw the existing 3-step checkout; treatment saw a new single-page checkout.

Primary metric: conversion rate (did the user complete a purchase?).
Secondary metric: revenue per user (not just conversion, but did order value change?).

Results:
- Control CVR: 5.12%, Treatment CVR: 6.84%
- Absolute lift: +1.72pp | Relative lift: +33.7%
- p-value < 0.0001 (highly significant at α = 0.05)
- Revenue per user also increased — Mann-Whitney p = 0.016
- Recommendation: **SHIP IT**

---

## Q2: Why did you use Mann-Whitney instead of a t-test for revenue?

**Answer:**
Revenue is zero-inflated — 93% of users don't convert at all, so revenue = 0.
This violates the normality assumption of the t-test.
Mann-Whitney U is non-parametric: it tests whether treatment revenue values tend
to be higher than control, without assuming a distribution shape.
It's more robust for sparse, skewed metrics like revenue per user.

At scale (50K users), the CLT would actually make a t-test acceptable too — but
Mann-Whitney is the more principled choice and shows statistical sophistication.

---

## Q3: What is CUPED and why does it matter?

**Answer:**
CUPED (Controlled-experiment Using Pre-Experiment Data) reduces the variance of
the outcome metric by removing variance explained by pre-experiment behaviour.

The adjustment is: Y_cuped = Y - θ * (X - E[X])
where X is a pre-experiment covariate (e.g. purchases in the 30 days before the test)
and θ = Cov(Y, X) / Var(X) — the OLS coefficient.

This is mathematically equivalent to running a regression of Y on X and using
the residual as your metric — but computed in a way that's safe for A/B tests.

**Why it matters:** It lets you either:
(a) detect the same effect size with fewer users, OR
(b) detect smaller effects with the same number of users.

If your covariate has ρ = 0.5 correlation with the metric → 25% variance reduction
→ equivalent to running the test on 25% more users for free.

---

## Q4: How did you choose sample size?

**Answer:**
Power analysis: given baseline CVR = 5.12%, MDE = 1pp (minimum meaningful effect),
α = 0.05, and target power = 80%, the required sample size is ~8,400 per variant.

We ran with 50,000 per variant — well above minimum — because:
1. We wanted to detect even smaller effects in segment breakdowns (device, country)
2. Higher power means the p-value is more stable and less sensitive to randomness
3. The experiment window was 30 days anyway (needed for weekly seasonality)

Formula: n = (z_α/2 * √(2p̄(1-p̄)) + z_β * √(p_c(1-p_c) + p_t(1-p_t)))² / δ²

---

## Q5: What's the difference between frequentist and Bayesian results here?

**Answer:**
- **Frequentist (z-test):** p-value answers "If H₀ is true, how likely is this data?"
  It tells you whether to reject the null, but not the probability that treatment is better.

- **Bayesian (Beta-Binomial):** P(Treatment > Control) answers exactly "What's the
  probability that treatment is truly better?" directly — 97.2% in this case.

The Bayesian result is more intuitive for stakeholders ("97% chance the new checkout
is better") but requires a prior. We used Beta(1,1) — the non-informative uniform prior
— which lets the data speak for itself.

Both agreed: ship it. When they disagree, investigate — it usually means the prior
matters, or the data is on the boundary of significance.

---

## Q6: What could go wrong with this experiment that you didn't measure?

**Answer:**
1. **Novelty effect** — users may behave differently because the checkout is new,
   not because it's better. Solution: run for 2+ full weeks and check if CVR stabilizes.

2. **Network effects / SUTVA violation** — if one user's experience affects another's
   (e.g. shared accounts, referral codes), the independence assumption breaks. Less
   relevant for checkout, more relevant for social/viral features.

3. **Simpson's paradox** — if mobile users (who convert less) ended up overrepresented
   in treatment due to randomization imbalance, the aggregate lift could be misleading.
   Solution: run the segment breakdown (see dashboard, device tab) and verify.

4. **Long-term cannibalization** — faster checkout might deplete demand faster.
   Measure 30-day retention and repeat purchase rate, not just same-session conversion.

5. **SRM (Sample Ratio Mismatch)** — check that 50,000 ≈ 50,000 in both groups.
   Large imbalance means the randomization is broken.

---

## Q7: What is your interpretation of the 95% confidence interval?

**Answer:**
The 95% CI for absolute lift is [+1.42pp, +2.02pp].

**Correct interpretation:** If we ran this experiment 100 times, approximately 95 of
those confidence intervals would contain the true population difference.

**Common wrong interpretation:** "There is a 95% chance the true lift is in this range."
(That's a Bayesian credible interval, not a frequentist CI.)

In practice: the interval is entirely above zero and economically meaningful.
Even the lower bound (+1.42pp) would represent significant revenue uplift at scale.
