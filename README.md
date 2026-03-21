# Duplicate Project Matcher

Cross-source project deduplication (entity resolution) for `.csv` and Excel (`.xlsx`, `.xls`) files.

## What it does

- Reads two input files (`source A` and `source B`)
- Normalizes key fields (`NAME`, address, city/state/zip, coordinates, numeric fields)
- Uses blocking to avoid `O(N^2)` comparisons
- Uses hierarchical matching: `state+city` block -> geo closeness gate -> text/entity evidence
- Calculates fuzzy + geo + numeric similarity features
- Computes weighted match score
- Produces outputs for:
  - `auto_match`
  - `review`
  - `non_match`

## Blocking strategy

The script generates candidate pairs from state/city-first blocks:

- `state + city`
- `state + city + zip5`
- `state + city + first5(name)`

Then it applies a geo gate:

- If both records have coordinates, keep only pairs within `--geo-candidate-max-distance-m` (default `5000`).
- If coordinates are missing on either side, pair is kept for text/entity checks.

## Scoring formula

Default weighted score:

```text
0.35 * name_similarity
+ 0.25 * address_similarity
+ 0.15 * city_match
+ 0.10 * geo_similarity
+ 0.10 * estimated_value_similarity
+ 0.05 * floors_similarity
+ 0.10 * owner_similarity
+ 0.10 * contractor_similarity
```

Scoring is adaptive: missing fields are excluded from denominator instead of forcing zero.

Threshold defaults:

- `auto_match >= 0.90`
- `review >= 0.80`

## Install

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python matcher.py --source-a data/source_a.xlsx --source-b data/source_b.csv --output-dir output
```

Optional:

```powershell
python matcher.py `
  --source-a data/source_a.csv `
  --source-b data/source_b.csv `
  --id-col-a PROJECT_ID `
  --id-col-b RECORD_ID `
  --auto-threshold 0.9 `
  --review-threshold 0.8 `
  --geo-candidate-max-distance-m 5000 `
  --export-llm-review `
  --print-column-mapping `
  --max-candidates 500000 `
  --output-dir output
```

## Output files

Created under `output/`:

- `matches_scored.csv` (all candidate pairs + features + score + decision)
- `matches_auto.csv` (high confidence)
- `matches_review.csv` (manual review queue)
- `summary.json` (counts)
- `matches_llm_review.csv` (only if `--export-llm-review` is set)

## Notes

- If expected columns are missing, the script tries alias names and gracefully falls back when possible.
- For better precision on large datasets, add name embeddings in a second phase.

## Hybrid Pipeline (Splink + EnsembleLink)

Use `splink_pipeline.py` for a two-stage architecture:

1. Stage 1 Retrieval (Splink)
- Broad candidate generation with `state + city` blocking and probabilistic scoring.
- Keep top-N candidates per left record (`--retrieval-top-k`).
- By default, only the top 1000 rows from each source are processed (`--max-rows 1000`). Use `--max-rows 0` to process all rows.

2. Stage 2 Reranking (EnsembleLink)
- Semantic reranking using `ensemblelink` if installed.
- Automatic fallback to heuristic semantic score if `ensemblelink` is not available.

Run:

```powershell
python splink_pipeline.py `
  --source-a ws2_project_1000_records_20260313.xlsx `
  --source-b cc_1000_records_20260313.xlsx `
  --max-rows 1000 `
  --retrieval-threshold 0.70 `
  --retrieval-top-k 50 `
  --final-threshold 0.90 `
  --final-top-k 1 `
  --output-dir output_hybrid
```

Optional:

```powershell
python splink_pipeline.py `
  --source-a ws2_project_1000_records_20260313.xlsx `
  --source-b cc_1000_records_20260313.xlsx `
  --device cuda `
  --ensemble-model-name ensemble-link-large-v2 `
  --output-dir output_hybrid
```

Output files in `output_hybrid/`:

- `stage1_topk_candidates.csv`
- `stage2_reranked_candidates.csv`
- `hybrid_final_matches.csv`
- `hybrid_clusters.csv`
- `hybrid_summary.json`
- `splink_retrieval_model.json` (or `splink_retrieval_settings.json` fallback)
