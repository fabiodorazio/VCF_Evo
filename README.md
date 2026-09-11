
This package is made of two main parts:

## 1. VCF Analyser
The code is structured in different modules that are orchestrated by the main.py file and it is built to
work on multi-chromosomal vcf files. The remaining part of the code is split across the files listed below:

- Contains functions to parse and transform the original vcf content to return info about the input
file and tables in csv format
- Contains functions to adapt the csv files for plotting and generate png outputs
- Contains side functions
- Optional function to return a mutational signature report
The main.py file requests arguments from the user and parses them into the functions that are imported
from the other modules. The whole program can be run with default arguments using the command
below:

### Run Analyser
```bash
python main.py ../inputs/chr20.variants_20072022.vcf
```


## 2. VCF Evo
Clonal haematopoiesis becomes increasingly prevalent with age, clone size changes longitudinally, and the growth behaviour differs substantially by driver gene.   More recent work also reinforces the connection between clonal haematopoiesis, ageing and inflammatory biology [Hajishengallis & Chavakis, 2026](https://www.nature.com/articles/s41580-025-00936-y?utm_source=chatgpt.com).

VCF-Evo is a framework for analysing how somatic variants change through time.

Most variant-analysis pipelines treat a sequencing sample as a static snapshot. They identify mutations, annotate them, calculate mutation burden, examine substitution patterns, and sometimes infer mutational signatures.
Longitudinal sequencing contains another layer of information. When the same individual is sequenced repeatedly, the allele frequency of a somatic mutation can be followed through time. These measurements provide information about the behaviour of the cell population carrying that mutation.

## GSE178936 input

The importer understands the ARCHER processed TSV/TSV.GZ columns including `participant_id`, `wave`, `chromosome`, `position`, `reference`, `mutation`, `DP`, `AO`, `UAO`, `AF`, annotations, and `X95MDAF`.

```bash
python bin/vcf_evo.py \
  --gse178936 inputs/GSE178936_LBC_ARCHER.1PCT_VAF.Feb22.non-synonymous.tsv.gz \
  -o outputs/gse178936_evo
```

Analyse one participant while developing:

```bash
python bin/vcf_evo.py \
  --gse178936 inputs/GSE178936_LBC_ARCHER.1PCT_VAF.Feb22.non-synonymous.tsv.gz \
  --participant CHIP_LBC21_001 \
  -o outputs/CHIP_LBC21_001
```


`X95MDAF` is preserved as assay metadata and `above_x95mdaf` is reported, but it is not automatically used as the hard detection threshold because the processed GEO dataset intentionally contains longitudinal measurements below its main inclusion threshold.

### Canonical schema

Both adapters produce participant-specific `trajectory_id` values (`participant_id|variant_id`) so the same genomic mutation observed in different people is never merged. The genomic ID is `chr:position:REF>ALT`.

### Outputs

- `longitudinal_variants.csv` — canonical observations and read-count/annotation metadata
- `variant_histories.csv` — persistent/emerging/disappearing/intermittent trajectories
- `trajectory_models.csv` — linear/exponential fits per participant + variant
- `timepoint_summary.csv` — burden/diversity by participant and timepoint
- `participant_acquisition_rates.csv` — newly detected trajectories per time unit

### Serial VCF input

Existing VCF input is retained:

```bash
python bin/vcf_evo.py \
  --timepoint inputs/person_T0.vcf:0:T0 \
  --timepoint inputs/person_T1.vcf:1:T1 \
  --timepoint inputs/person_T2.vcf:2:T2 \
  -o outputs/vcf_evo
```

The current change-point model remains a screening statistic. Research-grade inference should next use AO/DP (and possibly UAO) in binomial/beta-binomial likelihoods with explicit error and censoring models.

### Read-count likelihood modelling

VCF-Evo now fits longitudinal clone growth directly from alternate-read counts and total depth. By default the GSE178936 analysis uses `AO`/`DP`. Pass `--use-unique-alt-depth` to use `UAO`/`DP` instead.

For each observation, `longitudinal_variants.csv` now includes:

- `count_vaf` = ALT reads / DP
- `count_vaf_ci_low`, `count_vaf_ci_high` = exact 95% binomial interval (configurable with `--confidence`)

For each trajectory, `read_count_models.csv` fits:

- a constant-VAF binomial model
- a time-varying binomial logistic model, `logit(VAF) = intercept + growth_rate * time`
- `growth_rate` and its standard error / confidence interval / Wald p-value
- an approximate VAF doubling time for positive low-VAF growth
- likelihood-ratio comparison against a constant clone
- AIC and `delta_aic_growth_vs_constant`

At the low allele fractions typical of CHIP, the fitted log-odds growth rate closely approximates the exponential log-VAF growth rate, while the logistic formulation remains mathematically bounded between 0 and 1.

Example:

```bash
python bin/vcf_evo.py \
  --gse178936 inputs/GSE178936_LBC_ARCHER.1PCT_VAF.Feb22.non-synonymous.tsv.gz \
  --participant CHIP_LBC21_001 \
  --confidence 0.95 \
  -o outputs/CHIP_LBC21_001
```

`read_count_models.csv` should be preferred over the original AF-regression `trajectory_models.csv` for inference. The latter is retained as a descriptive/backward-compatible output.

## Plotting

VCF-Evo now includes `bin/vcf_evo_plotting.py`. Plotting operates on the canonical longitudinal tables, so it works for either GSE178936 TSV input or VCF input.

Generate the standard plot bundle with:

```bash
python bin/vcf_evo.py \
  --gse178936 inputs/GSE178936_LBC_ARCHER.1PCT_VAF.Feb22.non-synonymous.tsv.gz \
  --participant CHIP_LBC21_001 \
  --plots \
  -o outputs/CHIP_LBC21_001
```

The `plots/` directory contains:

- `trajectory_state_summary.png` — persistent/emerging/disappearing/intermittent counts.
- `participants/<id>/vaf_trajectories.png` — longitudinal VAF trajectories with exact binomial observation intervals when available.
- `participants/<id>/growth_rate_forest.png` — count-model growth rates and 95% CIs.
- `participants/<id>/fit_<variant>.png` — fitted binomial-logit curves for up to five trajectories with the largest absolute growth rates.

For full-cohort runs, plotting is capped at 20 participants and 20 trajectories per participant by default. Change this with `--plot-max-participants` and `--plot-max-trajectories`.


### Run Evo
```bash
python bin/vcf_evo.py --gse178936 inputs/GSE178936_LBC_ARCHER.1PCT_VAF.Feb22.non-synonymous.tsv --participant CHIP_LBC21_001 --plots -o outputs/CHIP_LBC21_001
  
```
