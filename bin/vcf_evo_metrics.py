"""Population- and participant-level longitudinal metrics for VCF-Evo."""
from __future__ import annotations
import numpy as np
import pandas as pd


def shannon_diversity(vafs) -> float:
    x=np.asarray(vafs,dtype=float); x=x[np.isfinite(x)&(x>0)]
    if len(x)==0 or x.sum()<=0: return np.nan
    p=x/x.sum(); return float(-np.sum(p*np.log(p)))


def gini(values) -> float:
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)&(x>=0)]
    if len(x)==0 or np.allclose(x,0): return np.nan
    x=np.sort(x); n=len(x); cumulative=np.sum(np.arange(1,n+1)*x)
    return float((2*cumulative)/(n*x.sum())-(n+1)/n)


def timepoint_summary(df: pd.DataFrame, detection_threshold: float=0.01) -> pd.DataFrame:
    required={"time","timepoint","variant_id","vaf"}; missing=required.difference(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    group_cols=[c for c in ["participant_id","time","timepoint"] if c in df.columns]
    rows=[]
    for keys, group in df.groupby(group_cols, sort=True):
        keys=(keys,) if not isinstance(keys,tuple) else keys; meta=dict(zip(group_cols,keys))
        detected=group[group["vaf"].fillna(0.0)>=detection_threshold]
        row={**meta,"n_variants":int(group["variant_id"].nunique()),"n_detected":int(detected["variant_id"].nunique()),
             "mean_vaf":float(detected["vaf"].mean()) if len(detected) else np.nan,
             "max_vaf":float(detected["vaf"].max()) if len(detected) else np.nan,
             "shannon_diversity":shannon_diversity(detected["vaf"]),"vaf_gini":gini(detected["vaf"])}
        if "x95mdaf" in group: row["n_above_x95mdaf"]=int((group["vaf"]>=group["x95mdaf"]).fillna(False).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def mutation_acquisition_rate(df: pd.DataFrame, detection_threshold: float=0.01) -> float:
    times=sorted(df["time"].dropna().unique())
    if len(times)<2 or times[-1]==times[0]: return np.nan
    key="trajectory_id" if "trajectory_id" in df.columns else "variant_id"
    detected=df[df["vaf"].fillna(0.0)>=detection_threshold]
    first_seen=detected.groupby(key)["time"].min(); new_after=int((first_seen>times[0]).sum())
    return float(new_after/(times[-1]-times[0]))


def participant_acquisition_rates(df: pd.DataFrame, detection_threshold: float=0.01) -> pd.DataFrame:
    if "participant_id" not in df.columns:
        return pd.DataFrame([{"participant_id":"single_subject","mutation_acquisition_rate":mutation_acquisition_rate(df,detection_threshold)}])
    rows=[]
    for participant, group in df.groupby("participant_id"):
        rows.append({"participant_id":participant,"mutation_acquisition_rate":mutation_acquisition_rate(group,detection_threshold),
                     "first_time":group["time"].min(),"last_time":group["time"].max(),"n_waves":group["time"].nunique()})
    return pd.DataFrame(rows)
