"""Read-count likelihood models for longitudinal VCF-Evo trajectories.

The primary model treats alternate observations as Binomial(DP, p_t) with
logit(p_t) = intercept + growth_rate * time.  This uses the sequencing counts
rather than regressing on pre-computed AF values and naturally gives more
weight to high-depth observations.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import beta, chi2, norm


def _clean_counts(time: Iterable, alt_depth: Iterable, depth: Iterable):
    x = np.asarray(time, dtype=float)
    k = np.asarray(alt_depth, dtype=float)
    n = np.asarray(depth, dtype=float)
    keep = np.isfinite(x) & np.isfinite(k) & np.isfinite(n) & (n > 0) & (k >= 0) & (k <= n)
    return x[keep], k[keep], n[keep]


def binomial_vaf_interval(alt_depth: float, depth: float, confidence: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson confidence interval for one observed VAF."""
    if not np.isfinite(alt_depth) or not np.isfinite(depth) or depth <= 0 or alt_depth < 0 or alt_depth > depth:
        return np.nan, np.nan
    k, n = int(round(alt_depth)), int(round(depth))
    alpha = 1.0 - confidence
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def add_binomial_vaf_intervals(
    df: pd.DataFrame,
    alt_depth_col: str = "alt_depth",
    depth_col: str = "dp",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Add count-derived VAF and exact binomial confidence intervals."""
    if alt_depth_col not in df.columns or depth_col not in df.columns:
        raise ValueError(f"Required count columns not found: {alt_depth_col}, {depth_col}")
    out = df.copy()
    k = pd.to_numeric(out[alt_depth_col], errors="coerce")
    n = pd.to_numeric(out[depth_col], errors="coerce")
    out["count_vaf"] = np.where(n > 0, k / n, np.nan)
    intervals = [binomial_vaf_interval(a, d, confidence) for a, d in zip(k, n)]
    out["count_vaf_ci_low"] = [v[0] for v in intervals]
    out["count_vaf_ci_high"] = [v[1] for v in intervals]
    return out


def _log_binomial_coeff(k, n):
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def _binomial_loglike(params, x, k, n, slope: bool = True):
    if slope:
        eta = params[0] + params[1] * x
    else:
        eta = np.repeat(params[0], len(x))
    # Stable log likelihood from eta directly.
    ll = _log_binomial_coeff(k, n) + k * (-np.logaddexp(0.0, -eta)) + (n - k) * (-np.logaddexp(0.0, eta))
    return float(np.sum(ll))


def _initial_intercept(k, n):
    # Jeffreys-style smoothing avoids +/-inf when all observations are 0 or n.
    p = (float(np.sum(k)) + 0.5) / (float(np.sum(n)) + 1.0)
    p = float(np.clip(p, 1e-8, 1.0 - 1e-8))
    return math.log(p / (1.0 - p))


def _fisher_covariance(intercept, slope, x, n):
    eta = intercept + slope * x
    p = expit(eta)
    w = n * p * (1.0 - p)
    X = np.column_stack([np.ones(len(x)), x])
    info = X.T @ (w[:, None] * X)
    try:
        return np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return np.full((2, 2), np.nan)


def fit_constant_binomial(time, alt_depth, depth) -> dict:
    """Fit a time-invariant Binomial VAF model."""
    x, k, n = _clean_counts(time, alt_depth, depth)
    if len(x) == 0:
        return {"model": "constant_binomial", "n": 0, "vaf": np.nan, "log_likelihood": np.nan, "aic": np.nan}
    intercept0 = _initial_intercept(k, n)
    result = minimize(lambda b: -_binomial_loglike(b, x, k, n, slope=False), x0=[intercept0], method="BFGS")
    intercept = float(result.x[0])
    ll = _binomial_loglike([intercept], x, k, n, slope=False)
    return {
        "model": "constant_binomial",
        "n": len(x),
        "total_depth": int(np.sum(n)),
        "total_alt": int(np.sum(k)),
        "intercept": intercept,
        "vaf": float(expit(intercept)),
        "log_likelihood": ll,
        "aic": float(2 - 2 * ll),
        "converged": bool(result.success),
    }


def fit_binomial_growth(time, alt_depth, depth, confidence: float = 0.95) -> dict:
    """Fit logit(VAF) = intercept + growth_rate*time using Binomial counts.

    ``growth_rate`` is in log-odds units per supplied time unit. For low VAFs,
    log-odds and log(VAF) are nearly equal, so ``log(2)/growth_rate`` is also
    reported as ``approx_vaf_doubling_time`` when growth is positive.
    """
    x, k, n = _clean_counts(time, alt_depth, depth)
    if len(x) < 2 or np.unique(x).size < 2:
        return {
            "model": "binomial_logit_growth", "n": len(x), "growth_rate": np.nan,
            "growth_rate_ci_low": np.nan, "growth_rate_ci_high": np.nan,
            "growth_rate_pvalue": np.nan, "approx_vaf_doubling_time": np.nan,
            "log_likelihood": np.nan, "aic": np.nan,
        }

    intercept0 = _initial_intercept(k, n)
    # Approximate AF slope gives the optimizer a useful starting direction.
    af = (k + 0.5) / (n + 1.0)
    try:
        slope0 = float(np.polyfit(x, np.log(af / (1.0 - af)), 1)[0])
    except Exception:
        slope0 = 0.0

    result = minimize(
        lambda b: -_binomial_loglike(b, x, k, n, slope=True),
        x0=[intercept0, slope0],
        method="BFGS",
    )
    intercept, growth_rate = map(float, result.x)
    ll = _binomial_loglike([intercept, growth_rate], x, k, n, slope=True)
    cov = _fisher_covariance(intercept, growth_rate, x, n)
    se = float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) and cov[1, 1] >= 0 else np.nan
    zcrit = float(norm.ppf(0.5 + confidence / 2.0))
    ci_low = growth_rate - zcrit * se if np.isfinite(se) else np.nan
    ci_high = growth_rate + zcrit * se if np.isfinite(se) else np.nan
    z = growth_rate / se if np.isfinite(se) and se > 0 else np.nan
    pvalue = float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    doubling = math.log(2.0) / growth_rate if growth_rate > 0 else np.nan

    t0, t1 = float(np.min(x)), float(np.max(x))
    p0, p1 = float(expit(intercept + growth_rate * t0)), float(expit(intercept + growth_rate * t1))
    return {
        "model": "binomial_logit_growth",
        "n": len(x),
        "total_depth": int(np.sum(n)),
        "total_alt": int(np.sum(k)),
        "intercept": intercept,
        "growth_rate": growth_rate,
        "growth_rate_se": se,
        "growth_rate_ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
        "growth_rate_ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
        "growth_rate_pvalue": pvalue,
        "approx_vaf_doubling_time": float(doubling) if np.isfinite(doubling) else np.nan,
        "fitted_first_vaf": p0,
        "fitted_last_vaf": p1,
        "log_likelihood": ll,
        "aic": float(4 - 2 * ll),
        "converged": bool(result.success),
    }


def compare_constant_vs_growth(time, alt_depth, depth, confidence: float = 0.95) -> dict:
    """Compare constant-VAF and time-varying Binomial models."""
    null = fit_constant_binomial(time, alt_depth, depth)
    growth = fit_binomial_growth(time, alt_depth, depth, confidence=confidence)
    if not np.isfinite(null.get("log_likelihood", np.nan)) or not np.isfinite(growth.get("log_likelihood", np.nan)):
        lr = p = delta_aic = np.nan
    else:
        lr = max(0.0, 2.0 * (growth["log_likelihood"] - null["log_likelihood"]))
        p = float(chi2.sf(lr, 1))
        delta_aic = float(null["aic"] - growth["aic"])
    return {
        **growth,
        "constant_vaf": null.get("vaf", np.nan),
        "constant_log_likelihood": null.get("log_likelihood", np.nan),
        "constant_aic": null.get("aic", np.nan),
        "likelihood_ratio": lr,
        "lr_pvalue": p,
        "delta_aic_growth_vs_constant": delta_aic,
        "preferred_model": "growth" if np.isfinite(delta_aic) and delta_aic > 2 else "constant_or_uncertain",
    }


def fit_read_count_trajectories(
    df: pd.DataFrame,
    min_points: int = 2,
    alt_depth_col: str = "alt_depth",
    depth_col: str = "dp",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Fit count-based models to every participant-specific trajectory."""
    required = {"variant_id", "time", alt_depth_col, depth_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    key = "trajectory_id" if "trajectory_id" in df.columns else "variant_id"
    rows = []
    for trajectory_id, group in df.groupby(key):
        group = group.sort_values("time")
        valid = group.dropna(subset=["time", alt_depth_col, depth_col]).copy()
        valid = valid[(valid[depth_col] > 0) & (valid[alt_depth_col] >= 0) & (valid[alt_depth_col] <= valid[depth_col])]
        if len(valid) < min_points or valid["time"].nunique() < 2:
            continue
        fit = compare_constant_vs_growth(valid["time"], valid[alt_depth_col], valid[depth_col], confidence)
        row = {
            "trajectory_id": trajectory_id,
            "variant_id": valid["variant_id"].iloc[0],
            "participant_id": valid["participant_id"].iloc[0] if "participant_id" in valid else None,
            "gene": valid["gene"].dropna().iloc[0] if "gene" in valid and valid["gene"].notna().any() else None,
            "protein_change": valid["protein_change"].dropna().iloc[0] if "protein_change" in valid and valid["protein_change"].notna().any() else None,
            "alt_depth_field": alt_depth_col,
            "first_time": float(valid["time"].iloc[0]),
            "last_time": float(valid["time"].iloc[-1]),
            **fit,
        }
        if "x95mdaf" in valid:
            row["n_above_x95mdaf"] = int((valid["vaf"] >= valid["x95mdaf"]).fillna(False).sum())
        rows.append(row)
    return pd.DataFrame(rows)
