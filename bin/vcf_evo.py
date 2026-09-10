"""Command-line interface for longitudinal VCF-Evo analysis."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from vcf_evo_io import TimepointSpec, read_gse178936_table, read_longitudinal_vcfs
from vcf_evo_longitudinal import classify_variant_history
from vcf_evo_metrics import participant_acquisition_rates, timepoint_summary
from vcf_evo_models import fit_variant_trajectories
from vcf_evo_readcounts import add_binomial_vaf_intervals, fit_read_count_trajectories
from vcf_evo_plotting import generate_standard_plots


def parse_timepoint(value: str) -> TimepointSpec:
    parts=value.split(":")
    if len(parts)<2: raise argparse.ArgumentTypeError("Expected PATH:TIME[:LABEL[:SAMPLE]]")
    path,time=parts[0],parts[1]; label=parts[2] if len(parts)>=3 and parts[2] else None; sample=parts[3] if len(parts)>=4 and parts[3] else None
    try: time=float(time)
    except ValueError as exc: raise argparse.ArgumentTypeError(f"Invalid time value: {time}") from exc
    return TimepointSpec(path=path,time=time,label=label,sample=sample)


def load_time_map(path):
    if not path: return None
    table=pd.read_csv(path)
    if not {"wave","time"}.issubset(table.columns): raise ValueError("time map must contain columns: wave,time")
    return dict(zip(pd.to_numeric(table.wave),pd.to_numeric(table.time)))


def get_args():
    parser=argparse.ArgumentParser(prog="vcf-evo")
    src=parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--timepoint",action="append",type=parse_timepoint,help="VCF observation PATH:TIME[:LABEL[:SAMPLE]]; repeat for each visit")
    src.add_argument("--gse178936",help="GSE178936 processed .tsv or .tsv.gz file")
    parser.add_argument("--participant",help="Optional participant_id filter for tabular input")
    parser.add_argument("--time-map",help="CSV with wave,time to convert wave to elapsed years/age")
    parser.add_argument("-o","--output-dir",default="../outputs/vcf_evo")
    parser.add_argument("--detection-threshold",type=float,default=0.01)
    parser.add_argument("--min-model-points",type=int,default=3)
    parser.add_argument("--read-count-min-points",type=int,default=2)
    parser.add_argument("--use-unique-alt-depth",action="store_true",help="Use UAO rather than AO for the binomial read-count model")
    parser.add_argument("--confidence",type=float,default=0.95,help="Confidence level for count-based intervals (default: 0.95)")
    parser.add_argument("--plots", action="store_true", help="Generate standard VCF-Evo longitudinal plots")
    parser.add_argument("--plot-max-participants", type=int, default=20, help="Maximum participants to plot in cohort mode")
    parser.add_argument("--plot-max-trajectories", type=int, default=20, help="Maximum trajectories per participant plot")
    return parser.parse_args()


def main():
    args=get_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    if args.gse178936:
        longitudinal=read_gse178936_table(args.gse178936,time_map=load_time_map(args.time_map),participant=args.participant)
    else:
        if len(args.timepoint)<2: raise SystemExit("VCF-Evo requires at least two --timepoint arguments")
        longitudinal=read_longitudinal_vcfs(args.timepoint)
    alt_depth_col="unique_alt_depth" if args.use_unique_alt_depth else "alt_depth"
    if alt_depth_col in longitudinal.columns and "dp" in longitudinal.columns:
        longitudinal=add_binomial_vaf_intervals(longitudinal,alt_depth_col=alt_depth_col,confidence=args.confidence)
    histories=classify_variant_history(longitudinal,args.detection_threshold)
    trajectories=fit_variant_trajectories(longitudinal,args.min_model_points)
    if alt_depth_col in longitudinal.columns and "dp" in longitudinal.columns:
        read_count_models=fit_read_count_trajectories(longitudinal,args.read_count_min_points,alt_depth_col=alt_depth_col,confidence=args.confidence)
    else:
        read_count_models=pd.DataFrame()
    summary=timepoint_summary(longitudinal,args.detection_threshold)
    rates=participant_acquisition_rates(longitudinal,args.detection_threshold)
    longitudinal.to_csv(out/"longitudinal_variants.csv",index=False)
    histories.to_csv(out/"variant_histories.csv",index=False)
    trajectories.to_csv(out/"trajectory_models.csv",index=False)
    read_count_models.to_csv(out/"read_count_models.csv",index=False)
    summary.to_csv(out/"timepoint_summary.csv",index=False)
    rates.to_csv(out/"participant_acquisition_rates.csv",index=False)
    if args.plots:
        plot_participants = [args.participant] if args.participant else None
        written = generate_standard_plots(
            longitudinal, histories, read_count_models, out/"plots",
            participants=plot_participants,
            max_participants=args.plot_max_participants,
            max_trajectories_per_participant=args.plot_max_trajectories,
        )
        print(f"Generated {len(written)} plot(s) in {out/'plots'}")
    print(f"VCF-Evo results written to {out}")

if __name__=="__main__": main()
