"""Plotting utilities for longitudinal VCF-Evo results.

The functions in this module deliberately operate on VCF-Evo's canonical
longitudinal tables and model outputs rather than on raw VCF records. This
keeps plotting independent of the original input format (VCF, TSV, MAF, etc.).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit


def _ensure_output_dir(output_dir) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _label_for_group(group: pd.DataFrame) -> str:
    gene = None
    protein = None
    if "gene" in group.columns and group["gene"].notna().any():
        gene = str(group.loc[group["gene"].notna(), "gene"].iloc[0])
    if "protein_change" in group.columns and group["protein_change"].notna().any():
        protein = str(group.loc[group["protein_change"].notna(), "protein_change"].iloc[0])
    if gene and protein:
        return f"{gene} {protein}"
    if gene:
        return gene
    if "variant_id" in group.columns:
        return str(group["variant_id"].iloc[0])
    return str(group.name)


def plot_participant_trajectories(
    longitudinal: pd.DataFrame,
    participant_id: str,
    output_path,
    min_max_vaf: float = 0.0,
    max_trajectories: int = 20,
    use_count_intervals: bool = True,
) -> Path:
    """Plot longitudinal VAF trajectories for one participant.

    The trajectories with the largest observed maximum VAF are shown when the
    participant has more than ``max_trajectories`` variants. Exact binomial
    observation intervals are drawn when available.
    """
    required = {"participant_id", "trajectory_id", "time", "vaf"}
    missing = required.difference(longitudinal.columns)
    if missing:
        raise ValueError(f"Missing columns for trajectory plot: {sorted(missing)}")
    data = longitudinal[longitudinal["participant_id"].astype(str) == str(participant_id)].copy()
    if data.empty:
        raise ValueError(f"Participant {participant_id!r} not found")

    ranked = (
        data.groupby("trajectory_id")["vaf"]
        .max()
        .sort_values(ascending=False)
    )
    ranked = ranked[ranked >= min_max_vaf].head(max_trajectories)
    data = data[data["trajectory_id"].isin(ranked.index)].copy()
    if data.empty:
        raise ValueError("No trajectories pass the plotting threshold")

    fig, ax = plt.subplots(figsize=(10, 6))
    for trajectory_id, group in data.groupby("trajectory_id", sort=False):
        group = group.sort_values("time")
        ycol = "count_vaf" if "count_vaf" in group.columns and group["count_vaf"].notna().any() else "vaf"
        x = pd.to_numeric(group["time"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(group[ycol], errors="coerce").to_numpy(float)
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        if not len(x):
            continue
        line = ax.plot(x, y * 100.0, marker="o", label=_label_for_group(group))[0]
        if use_count_intervals and {"count_vaf_ci_low", "count_vaf_ci_high"}.issubset(group.columns):
            lo = pd.to_numeric(group.loc[keep, "count_vaf_ci_low"], errors="coerce").to_numpy(float)
            hi = pd.to_numeric(group.loc[keep, "count_vaf_ci_high"], errors="coerce").to_numpy(float)
            ok = np.isfinite(lo) & np.isfinite(hi)
            if ok.any():
                ax.vlines(x[ok], lo[ok] * 100.0, hi[ok] * 100.0, alpha=0.45, linewidth=1.0)
    ax.set_xlabel("Time")
    ax.set_ylabel("Variant allele frequency (%)")
    ax.set_title(f"VCF-Evo trajectories: {participant_id}")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fitted_growth_curve(
    longitudinal: pd.DataFrame,
    model_row: pd.Series | dict,
    output_path,
    n_grid: int = 200,
) -> Path:
    """Plot observed count-derived VAF and a fitted binomial-logit trajectory."""
    model = dict(model_row)
    trajectory_id = model.get("trajectory_id")
    if trajectory_id is None:
        raise ValueError("model_row must contain trajectory_id")
    data = longitudinal[longitudinal["trajectory_id"] == trajectory_id].sort_values("time").copy()
    if data.empty:
        raise ValueError(f"Trajectory {trajectory_id!r} not found")
    if not np.isfinite(model.get("intercept", np.nan)) or not np.isfinite(model.get("growth_rate", np.nan)):
        raise ValueError("model_row must contain finite intercept and growth_rate")

    x = pd.to_numeric(data["time"], errors="coerce").to_numpy(float)
    if "count_vaf" in data.columns and data["count_vaf"].notna().any():
        y = pd.to_numeric(data["count_vaf"], errors="coerce").to_numpy(float)
    else:
        y = pd.to_numeric(data["vaf"], errors="coerce").to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    grid = np.linspace(float(np.min(x)), float(np.max(x)), n_grid)
    fitted = expit(float(model["intercept"]) + float(model["growth_rate"]) * grid)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y * 100.0, zorder=3, label="Observed")
    if {"count_vaf_ci_low", "count_vaf_ci_high"}.issubset(data.columns):
        lo = pd.to_numeric(data.loc[keep, "count_vaf_ci_low"], errors="coerce").to_numpy(float)
        hi = pd.to_numeric(data.loc[keep, "count_vaf_ci_high"], errors="coerce").to_numpy(float)
        ok = np.isfinite(lo) & np.isfinite(hi)
        if ok.any():
            ax.vlines(x[ok], lo[ok] * 100.0, hi[ok] * 100.0, alpha=0.55, linewidth=1.2)
    ax.plot(grid, fitted * 100.0, linewidth=2.0, label="Binomial-logit fit")
    ax.set_xlabel("Time")
    ax.set_ylabel("Variant allele frequency (%)")
    title = _label_for_group(data)
    ax.set_title(f"{title} — growth={model['growth_rate']:.3g}/time unit")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_growth_rate_forest(
    read_count_models: pd.DataFrame,
    output_path,
    participant_id: Optional[str] = None,
    max_models: int = 30,
    sort_by: str = "growth_rate",
) -> Path:
    """Forest plot of count-based growth-rate estimates and 95% intervals."""
    required = {"growth_rate", "growth_rate_ci_low", "growth_rate_ci_high"}
    missing = required.difference(read_count_models.columns)
    if missing:
        raise ValueError(f"Missing columns for forest plot: {sorted(missing)}")
    data = read_count_models.copy()
    if participant_id is not None:
        data = data[data["participant_id"].astype(str) == str(participant_id)]
    data = data.dropna(subset=["growth_rate", "growth_rate_ci_low", "growth_rate_ci_high"])
    if data.empty:
        raise ValueError("No finite growth-rate models to plot")
    data = data.sort_values(sort_by).tail(max_models).copy()

    labels = []
    for _, row in data.iterrows():
        label = str(row.get("gene") or row.get("variant_id") or row.get("trajectory_id"))
        protein = row.get("protein_change")
        if pd.notna(protein) and str(protein):
            label += f" {protein}"
        labels.append(label)
    y = np.arange(len(data))
    rate = data["growth_rate"].to_numpy(float)
    lo = data["growth_rate_ci_low"].to_numpy(float)
    hi = data["growth_rate_ci_high"].to_numpy(float)

    fig_height = max(4.5, 0.32 * len(data) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.errorbar(rate, y, xerr=np.vstack([rate - lo, hi - rate]), fmt="o", capsize=3)
    ax.axvline(0.0, linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Growth rate (log-odds / time unit), 95% CI")
    title = "VCF-Evo clone growth rates"
    if participant_id is not None:
        title += f": {participant_id}"
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_history_summary(histories: pd.DataFrame, output_path) -> Path:
    """Plot counts of persistent/emerging/disappearing/intermittent trajectories."""
    if "history" not in histories.columns:
        raise ValueError("histories table must contain a history column")
    counts = histories["history"].value_counts().reindex(
        ["persistent", "emerging", "disappearing", "intermittent"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Number of trajectories")
    ax.set_title("VCF-Evo trajectory states")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_standard_plots(
    longitudinal: pd.DataFrame,
    histories: pd.DataFrame,
    read_count_models: pd.DataFrame,
    output_dir,
    participants: Optional[Sequence[str]] = None,
    max_participants: int = 20,
    max_trajectories_per_participant: int = 20,
) -> list[Path]:
    """Generate a standard VCF-Evo plot set and return written paths."""
    out = _ensure_output_dir(output_dir)
    written = []
    if not histories.empty:
        written.append(plot_history_summary(histories, out / "trajectory_state_summary.png"))
    if participants is None and "participant_id" in longitudinal.columns:
        participants = list(longitudinal["participant_id"].dropna().astype(str).unique()[:max_participants])
    for participant in participants or []:
        participant_dir = _ensure_output_dir(out / "participants" / str(participant))
        written.append(
            plot_participant_trajectories(
                longitudinal,
                participant,
                participant_dir / "vaf_trajectories.png",
                max_trajectories=max_trajectories_per_participant,
            )
        )
        if not read_count_models.empty and "participant_id" in read_count_models.columns:
            models = read_count_models[read_count_models["participant_id"].astype(str) == str(participant)]
            finite = models.dropna(subset=["growth_rate", "growth_rate_ci_low", "growth_rate_ci_high"])
            if not finite.empty:
                written.append(plot_growth_rate_forest(finite, participant_dir / "growth_rate_forest.png", participant_id=participant))
                top = finite.assign(abs_rate=finite["growth_rate"].abs()).sort_values("abs_rate", ascending=False).head(5)
                for _, row in top.iterrows():
                    safe = str(row["variant_id"]).replace(":", "_").replace(">", "-")
                    written.append(plot_fitted_growth_curve(longitudinal, row, participant_dir / f"fit_{safe}.png"))
    return written
