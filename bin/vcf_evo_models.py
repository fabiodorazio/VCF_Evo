"""Simple, dependency-light trajectory models for VCF-Evo."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _clean_xy(time, vaf):
    x = np.asarray(time, dtype=float)
    y = np.asarray(vaf, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def fit_linear_growth(time, vaf) -> dict:
    """Fit VAF = intercept + slope*time using ordinary least squares."""
    x, y = _clean_xy(time, vaf)
    if len(x) < 2:
        return {"model": "linear", "n": len(x), "slope": np.nan, "intercept": np.nan, "r2": np.nan}
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "model": "linear",
        "n": len(x),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
    }


def fit_exponential_growth(time, vaf, min_vaf: float = 1e-6) -> dict:
    """Fit VAF(t)=VAF0*exp(r*t); return r and VAF doubling time."""
    x, y = _clean_xy(time, vaf)
    keep = y > min_vaf
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return {"model": "exponential", "n": len(x), "growth_rate": np.nan, "vaf0": np.nan, "doubling_time": np.nan, "r2_log": np.nan}

    growth_rate, log_vaf0 = np.polyfit(x, np.log(y), 1)
    pred_log = log_vaf0 + growth_rate * x
    observed_log = np.log(y)
    ss_res = float(np.sum((observed_log - pred_log) ** 2))
    ss_tot = float(np.sum((observed_log - np.mean(observed_log)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    doubling = math.log(2) / growth_rate if growth_rate > 0 else np.nan
    return {
        "model": "exponential",
        "n": len(x),
        "growth_rate": float(growth_rate),
        "vaf0": float(np.exp(log_vaf0)),
        "doubling_time": float(doubling) if np.isfinite(doubling) else np.nan,
        "r2_log": r2,
    }


def fit_variant_trajectories(df: pd.DataFrame, min_points: int = 3) -> pd.DataFrame:
    """Fit linear and exponential models to every variant trajectory."""
    required = {"variant_id", "time", "vaf"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    rows = []
    key = "trajectory_id" if "trajectory_id" in df.columns else "variant_id"
    for trajectory_id, group in df.groupby(key):
        group = group.sort_values("time")
        valid = group.dropna(subset=["time", "vaf"])
        if len(valid) < min_points:
            continue
        lin = fit_linear_growth(valid["time"], valid["vaf"])
        exp = fit_exponential_growth(valid["time"], valid["vaf"])
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "variant_id": valid["variant_id"].iloc[0],
                "participant_id": valid["participant_id"].iloc[0] if "participant_id" in valid else None,
                "gene": valid["gene"].dropna().iloc[0] if "gene" in valid and valid["gene"].notna().any() else None,
                "protein_change": valid["protein_change"].dropna().iloc[0] if "protein_change" in valid and valid["protein_change"].notna().any() else None,
                "n": len(valid),
                "first_time": float(valid["time"].iloc[0]),
                "last_time": float(valid["time"].iloc[-1]),
                "first_vaf": float(valid["vaf"].iloc[0]),
                "last_vaf": float(valid["vaf"].iloc[-1]),
                "linear_slope": lin["slope"],
                "linear_r2": lin["r2"],
                "exp_growth_rate": exp["growth_rate"],
                "doubling_time": exp["doubling_time"],
                "exp_r2_log": exp["r2_log"],
            }
        )
    return pd.DataFrame(rows)


def best_single_change_point(time, vaf, min_segment: int = 2) -> dict:
    """Find a single piecewise-linear change point by minimum squared error.

    This is intentionally simple and interpretable; it is suitable for screening,
    not for claiming a statistically validated intervention effect.
    """
    x, y = _clean_xy(time, vaf)
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)
    if n < 2 * min_segment:
        return {"change_time": np.nan, "sse": np.nan, "pre_slope": np.nan, "post_slope": np.nan}

    best = None
    for split in range(min_segment, n - min_segment + 1):
        x1, y1 = x[:split], y[:split]
        x2, y2 = x[split:], y[split:]
        s1, i1 = np.polyfit(x1, y1, 1)
        s2, i2 = np.polyfit(x2, y2, 1)
        sse = float(np.sum((y1 - (i1 + s1 * x1)) ** 2) + np.sum((y2 - (i2 + s2 * x2)) ** 2))
        candidate = {
            "change_time": float((x[split - 1] + x[split]) / 2.0),
            "sse": sse,
            "pre_slope": float(s1),
            "post_slope": float(s2),
        }
        if best is None or sse < best["sse"]:
            best = candidate
    return best
