"""Longitudinal variant matching and trajectory construction for VCF-Evo."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"variant_id", "time", "timepoint", "vaf"}


def validate_longitudinal_table(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Longitudinal table is missing columns: {sorted(missing)}")
    if df["time"].isna().any():
        raise ValueError("Time values must not be missing")


def trajectory_column(df: pd.DataFrame) -> str:
    return "trajectory_id" if "trajectory_id" in df.columns else "variant_id"


def build_trajectory_matrix(df: pd.DataFrame, value: str = "vaf") -> pd.DataFrame:
    validate_longitudinal_table(df)
    if value not in df.columns:
        raise ValueError(f"Column {value!r} is not present")
    key = trajectory_column(df)
    return df.pivot_table(index=key, columns="time", values=value, aggfunc="first").sort_index(axis=1).sort_index()


def classify_variant_history(df: pd.DataFrame, detection_threshold: float = 0.01) -> pd.DataFrame:
    """Classify participant-specific trajectories by a consistent VAF threshold.

    X95MDAF is reported separately when present, but is deliberately not used as
    the default call threshold for GSE178936.
    """
    validate_longitudinal_table(df)
    key = trajectory_column(df)
    rows = []
    grouping = ["participant_id", key] if "participant_id" in df.columns and key != "participant_id" else [key]
    for keys, group in df.groupby(grouping):
        group = group.sort_values("time")
        times = sorted(group["time"].unique())
        if len(times) < 2:
            continue
        by_time = group.groupby("time")["vaf"].first().reindex(times)
        detected = by_time.fillna(0.0) >= detection_threshold
        first, last = bool(detected.iloc[0]), bool(detected.iloc[-1])
        n_detected = int(detected.sum())
        if first and last and n_detected == len(times): category = "persistent"
        elif not first and last: category = "emerging"
        elif first and not last: category = "disappearing"
        else: category = "intermittent"
        row = {
            "trajectory_id": group[key].iloc[0], "variant_id": group["variant_id"].iloc[0],
            "history": category, "n_observed_timepoints": len(times), "n_timepoints_detected": n_detected,
            "first_vaf": float(by_time.iloc[0]) if pd.notna(by_time.iloc[0]) else np.nan,
            "last_vaf": float(by_time.iloc[-1]) if pd.notna(by_time.iloc[-1]) else np.nan,
        }
        if "participant_id" in group: row["participant_id"] = group["participant_id"].iloc[0]
        for col in ["gene", "protein_change"]:
            if col in group: row[col] = group[col].dropna().iloc[0] if group[col].notna().any() else None
        if "x95mdaf" in group:
            row["n_above_x95mdaf"] = int((group["vaf"] >= group["x95mdaf"]).fillna(False).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def acquisition_table(df: pd.DataFrame, detection_threshold: float = 0.01) -> pd.DataFrame:
    validate_longitudinal_table(df)
    key = trajectory_column(df)
    detected = df[df["vaf"].fillna(0.0) >= detection_threshold].copy()
    if detected.empty: return detected
    idx = detected.groupby(key)["time"].idxmin()
    cols = [c for c in ["participant_id", "trajectory_id", "variant_id", "gene", "chrom", "pos", "ref", "alt", "time", "timepoint", "vaf", "dp", "alt_depth"] if c in detected]
    return detected.loc[idx, cols].sort_values([c for c in ["participant_id", "time"] if c in cols]).reset_index(drop=True)
