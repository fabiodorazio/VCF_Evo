"""VCF-Evo input adapters and canonical longitudinal schema."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
try:
    from pysam import VariantFile
except ImportError:  # TSV-only use does not require pysam
    VariantFile = None


@dataclass(frozen=True)
class TimepointSpec:
    path: str
    time: float
    label: Optional[str] = None
    sample: Optional[str] = None


def normalise_chromosome(chrom: str) -> str:
    value = str(chrom).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    return f"chr{value}"


def variant_key(chrom: str, pos: int, ref: str, alt: str) -> str:
    return f"{normalise_chromosome(chrom)}:{int(pos)}:{str(ref).upper()}>{str(alt).upper()}"


def add_trajectory_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "participant_id" not in out.columns:
        out["participant_id"] = "single_subject"
    out["trajectory_id"] = out["participant_id"].astype(str) + "|" + out["variant_id"].astype(str)
    return out


def _first_numeric(value):
    if value is None:
        return np.nan
    if isinstance(value, (tuple, list)):
        return value[0] if value else np.nan
    return value


def _extract_ad(sample_data, alt_index: int):
    ad = sample_data.get("AD")
    if ad is None or len(ad) <= alt_index:
        return np.nan, np.nan
    return ad[0], ad[alt_index]


def _extract_vaf(sample_data, alt_index: int):
    ref_depth, alt_depth = _extract_ad(sample_data, alt_index)
    if not pd.isna(ref_depth) and not pd.isna(alt_depth):
        total = ref_depth + alt_depth
        if total > 0:
            return float(alt_depth) / float(total)
    af = sample_data.get("AF")
    if af is None:
        return np.nan
    if isinstance(af, (tuple, list)):
        i = alt_index - 1
        return float(af[i]) if i < len(af) and af[i] is not None else np.nan
    return float(af)


def read_timepoint_vcf(spec: TimepointSpec) -> pd.DataFrame:
    if VariantFile is None:
        raise ImportError("pysam is required for VCF input; GSE178936 TSV input does not require it")
    path = Path(spec.path)
    rows = []
    with VariantFile(str(path), "r") as vcf:
        samples = list(vcf.header.samples)
        if not samples:
            raise ValueError(f"VCF has no sample columns: {path}")
        sample_name = spec.sample or samples[0]
        if sample_name not in samples:
            raise ValueError(f"Sample {sample_name!r} not found in {path}. Available: {samples}")
        label = spec.label or path.stem
        for record in vcf:
            if not record.alts:
                continue
            sample_data = record.samples[sample_name]
            gt = sample_data.get("GT")
            gt_text = None if gt is None else "/".join("." if x is None else str(x) for x in gt)
            dp = _first_numeric(sample_data.get("DP"))
            gq = _first_numeric(sample_data.get("GQ"))
            for alt_index, alt in enumerate(record.alts, start=1):
                ref_depth, alt_depth = _extract_ad(sample_data, alt_index)
                rows.append({
                    "participant_id": sample_name,
                    "variant_id": variant_key(record.chrom, record.pos, record.ref, alt),
                    "chrom": normalise_chromosome(record.chrom), "pos": int(record.pos),
                    "ref": record.ref, "alt": alt, "time": float(spec.time),
                    "timepoint": label, "sample": sample_name, "qual": record.qual,
                    "filter": ";".join(record.filter.keys()) if record.filter.keys() else ".",
                    "gt": gt_text, "dp": dp, "gq": gq,
                    "ref_depth": ref_depth, "alt_depth": alt_depth,
                    "vaf": _extract_vaf(sample_data, alt_index), "source": "vcf",
                })
    return add_trajectory_id(pd.DataFrame(rows))


def read_longitudinal_vcfs(specs: Iterable[TimepointSpec]) -> pd.DataFrame:
    frames = [read_timepoint_vcf(spec) for spec in specs]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["participant_id", "variant_id", "time", "timepoint"]).reset_index(drop=True)


GSE178936_REQUIRED = {
    "participant_id", "wave", "chromosome", "position", "reference", "mutation", "DP", "AF"
}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"NA": np.nan, "": np.nan}), errors="coerce")


def read_gse178936_table(
    path: str,
    time_map: Optional[Mapping[float, float]] = None,
    participant: Optional[str] = None,
) -> pd.DataFrame:
    """Read the GSE178936 processed ARCHER TSV/TSV.GZ into VCF-Evo schema.

    ``wave`` is used as the time coordinate by default. Supply ``time_map`` to
    replace waves with elapsed years/age once a validated mapping is available.
    X95MDAF is retained as assay metadata; it is not assumed to be a hard call
    threshold because the GEO processed file deliberately contains tracked
    measurements below the cohort inclusion VAF threshold.
    """
    raw = pd.read_csv(path, sep="\t", compression="infer", low_memory=False)
    missing = GSE178936_REQUIRED.difference(raw.columns)
    if missing:
        raise ValueError(f"GSE178936 table is missing columns: {sorted(missing)}")
    if participant is not None:
        raw = raw[raw["participant_id"].astype(str) == str(participant)].copy()
        if raw.empty:
            raise ValueError(f"Participant {participant!r} not found")

    wave = _numeric(raw["wave"])
    time = wave.map(time_map) if time_map is not None else wave
    if time_map is not None and time.isna().any():
        missing_waves = sorted(wave[time.isna()].dropna().unique().tolist())
        raise ValueError(f"time_map has no values for waves: {missing_waves}")

    out = pd.DataFrame({
        "participant_id": raw["participant_id"].astype(str),
        "time": time.astype(float),
        "timepoint": raw["wave"].map(lambda x: f"wave_{x}"),
        "wave": wave,
        "chrom": raw["chromosome"].map(normalise_chromosome),
        "pos": _numeric(raw["position"]).astype("Int64"),
        "ref": raw["reference"].astype(str).str.upper(),
        "alt": raw["mutation"].astype(str).str.upper(),
        "dp": _numeric(raw["DP"]),
        "alt_depth": _numeric(raw["AO"]) if "AO" in raw else np.nan,
        "unique_alt_depth": _numeric(raw["UAO"]) if "UAO" in raw else np.nan,
        "vaf": _numeric(raw["AF"]),
        "x95mdaf": _numeric(raw["X95MDAF"]) if "X95MDAF" in raw else np.nan,
        "quality": _numeric(raw["quality"]) if "quality" in raw else np.nan,
        "gene": raw["PreferredSymbol"] if "PreferredSymbol" in raw else None,
        "variant_type": raw["type"] if "type" in raw else None,
        "protein_change": raw["HGVSp"] if "HGVSp" in raw else None,
        "coding_change": raw["HGVSc"] if "HGVSc" in raw else None,
        "consequence": raw["consequence"] if "consequence" in raw else None,
        "variant_classification": raw["Variant_Classification"] if "Variant_Classification" in raw else None,
        "cosmic_id": raw["COSMICID"] if "COSMICID" in raw else None,
        "gnomad_af": _numeric(raw["gnomAD_AF"]) if "gnomAD_AF" in raw else np.nan,
        "source": "GSE178936",
    })
    out["variant_id"] = [variant_key(c, p, r, a) for c, p, r, a in zip(out.chrom, out.pos, out.ref, out.alt)]
    out["ref_depth"] = out["dp"] - out["alt_depth"]
    out["above_x95mdaf"] = np.where(
        out["x95mdaf"].notna() & out["vaf"].notna(), out["vaf"] >= out["x95mdaf"], pd.NA
    )
    out = add_trajectory_id(out)
    return out.sort_values(["participant_id", "variant_id", "time"]).reset_index(drop=True)
