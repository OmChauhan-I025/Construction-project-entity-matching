from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import splink.duckdb.comparison_library as cl
from rapidfuzz import fuzz
from splink.duckdb.linker import DuckDBLinker

CANONICAL_COLUMNS = {
    "id": ["ID", "PROJECT_ID", "PROJECT ID", "RECORD_ID", "RECORD ID"],
    "name": ["NAME", "PROJECT_NAME", "PROJECT NAME"],
    "address": ["ADDRESS_LINE_1__C", "ADDRESS_LINE_1", "ADDRESS", "ADDRESS LINE 1"],
    "city": ["CITY__C", "CITY"],
    "state": ["STATE__C", "STATE"],
    "zip": ["ZIP__C", "ZIP", "POSTAL_CODE", "ZIPCODE"],
    "lat": ["LOCATION__LATITUDE__S", "LATITUDE", "LAT", "LOCATION LATITUDE S"],
    "lon": ["LOCATION__LONGITUDE__S", "LONGITUDE", "LON", "LNG", "LOCATION LONGITUDE S"],
    "owner": ["OWNER", "OWNER__C", "OWNER C", "C OWNER", "OWNER_H_PARENT"],
    "contractor": ["CONTRACTOR", "CONTRACTOR__C", "CONTRACTOR C", "GENERAL_CONTRACTOR", "GC"],
    "estimated_value": ["ESTIMATED_VALUE__C", "ESTIMATED_VALUE", "C_ESTIMATE", "ESTIMATE"],
}

STREET_ABBREVIATIONS = {
    " STREET ": " ST ",
    " ROAD ": " RD ",
    " AVENUE ": " AVE ",
    " BOULEVARD ": " BLVD ",
    " DRIVE ": " DR ",
    " LANE ": " LN ",
    " COURT ": " CT ",
    " PLACE ": " PL ",
    " HIGHWAY ": " HWY ",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid record linkage: Splink retrieval + EnsembleLink/fallback reranking"
    )
    parser.add_argument("--source-a", required=True, help="Source A path (.csv/.xlsx/.xls)")
    parser.add_argument("--source-b", required=True, help="Source B path (.csv/.xlsx/.xls)")
    parser.add_argument("--output-dir", default="output_hybrid", help="Output directory")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1000,
        help="Maximum rows to read from each source (0 = all rows)",
    )

    parser.add_argument(
        "--retrieval-threshold",
        type=float,
        default=0.15,
        help="Stage-1 Splink retrieval threshold (broader candidate net)",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=120,
        help="Max candidates kept per left-side project after Stage-1 retrieval",
    )

    parser.add_argument(
        "--final-threshold",
        type=float,
        default=0.72,
        help="Final hybrid score threshold for output matches",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=1,
        help="Final number of matches kept per left-side project after reranking",
    )

    parser.add_argument(
        "--ensemble-model-name",
        default="ensemble-link-large-v2",
        help="EnsembleLink model name",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Model device for EnsembleLink (e.g., cpu, cuda)",
    )
    parser.add_argument(
        "--disable-ensemblelink",
        action="store_true",
        help="Skip EnsembleLink import/model load and use heuristic reranking",
    )

    parser.add_argument(
        "--max-training-pairs",
        type=int,
        default=1_000_000,
        help="Max random pairs used for Splink u estimation",
    )
    parser.add_argument(
        "--print-column-mapping",
        action="store_true",
        help="Print detected source column mappings",
    )
    return parser.parse_args()


def read_input_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    raise ValueError(f"Unsupported format: {suffix}. Use .csv, .xlsx, or .xls")


def normalize_column_name(col: str) -> str:
    return "".join(ch for ch in col.upper().strip() if ch.isalnum() or ch == "_")


def find_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    normalized_map = {normalize_column_name(c): c for c in df.columns}
    for alias in aliases:
        alias_key = normalize_column_name(alias)
        if alias_key in normalized_map:
            return normalized_map[alias_key]
    return None


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    selected: Dict[str, Optional[str]] = {}
    for canonical, aliases in CANONICAL_COLUMNS.items():
        selected[canonical] = find_column(df, aliases)
    return selected


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).upper().strip()
    return " ".join(text.split())


def normalize_address(value: object) -> str:
    text = f" {normalize_text(value)} "
    for source, target in STREET_ABBREVIATIONS.items():
        text = text.replace(source, target)
    return " ".join(text.strip().split())


def normalize_zip(value: object) -> str:
    text = normalize_text(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:5]


def coerce_float(value: object) -> float:
    if pd.isna(value):
        return math.nan
    text = str(value).replace(",", "").replace("$", "").strip()
    try:
        return float(text)
    except ValueError:
        return math.nan


def preprocess(df: pd.DataFrame, source_tag: str, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    if mapping["id"] is None:
        out["project_id"] = [f"{source_tag}_{i}" for i in range(len(df))]
    else:
        values = df[mapping["id"]]
        out["project_id"] = values.where(values.notna(), "").astype(str).str.strip()
        empty_mask = out["project_id"].eq("")
        out.loc[empty_mask, "project_id"] = [f"{source_tag}_{i}" for i in out.index[empty_mask]]

    out["name"] = df[mapping["name"]].map(normalize_text) if mapping["name"] else ""
    out["address"] = df[mapping["address"]].map(normalize_address) if mapping["address"] else ""
    out["city"] = df[mapping["city"]].map(normalize_text) if mapping["city"] else ""
    out["state"] = df[mapping["state"]].map(normalize_text) if mapping["state"] else ""
    out["zip5"] = df[mapping["zip"]].map(normalize_zip) if mapping["zip"] else ""

    out["lat"] = df[mapping["lat"]].map(coerce_float) if mapping["lat"] else math.nan
    out["lon"] = df[mapping["lon"]].map(coerce_float) if mapping["lon"] else math.nan

    out["owner"] = df[mapping["owner"]].map(normalize_text) if mapping["owner"] else ""
    out["contractor"] = df[mapping["contractor"]].map(normalize_text) if mapping["contractor"] else ""
    out["estimated_value"] = (
        df[mapping["estimated_value"]].map(coerce_float) if mapping["estimated_value"] else math.nan
    )

    out = out.reset_index(drop=True)
    out["source_dataset"] = source_tag
    out["uid"] = out["source_dataset"] + "_" + out.index.astype(str)
    return out


def build_splink_settings(has_owner: bool, has_contractor: bool, has_geo: bool) -> Dict:
    comparisons = [
        cl.exact_match("state", m_probability_exact_match=0.97, m_probability_else=0.03),
        cl.exact_match("city", m_probability_exact_match=0.97, m_probability_else=0.03),
        cl.exact_match("zip5", m_probability_exact_match=0.90, m_probability_else=0.10),
        cl.jaro_winkler_at_thresholds(
            "name",
            [0.95, 0.90, 0.80],
            term_frequency_adjustments=False,
            include_exact_match_level=True,
            m_probability_exact_match=0.98,
            m_probability_or_probabilities_jw=[0.90, 0.80, 0.60],
            m_probability_else=0.02,
        ),
        cl.jaro_winkler_at_thresholds(
            "address",
            [0.95, 0.90, 0.80],
            include_exact_match_level=True,
            m_probability_exact_match=0.95,
            m_probability_or_probabilities_jw=[0.85, 0.70, 0.50],
            m_probability_else=0.05,
        ),
    ]

    if has_owner:
        comparisons.append(
            cl.jaro_winkler_at_thresholds(
                "owner",
                [0.95, 0.85],
                include_exact_match_level=True,
                m_probability_exact_match=0.92,
                m_probability_or_probabilities_jw=[0.75, 0.45],
                m_probability_else=0.08,
            )
        )

    if has_contractor:
        comparisons.append(
            cl.jaro_winkler_at_thresholds(
                "contractor",
                [0.95, 0.85],
                include_exact_match_level=True,
                m_probability_exact_match=0.92,
                m_probability_or_probabilities_jw=[0.75, 0.45],
                m_probability_else=0.08,
            )
        )

    if has_geo:
        comparisons.append(
            cl.distance_in_km_at_thresholds(
                "lat",
                "lon",
                [0.2, 1.0, 5.0],
                include_exact_match_level=False,
                m_probability_or_probabilities_km=[0.95, 0.75, 0.35],
                m_probability_else=0.05,
            )
        )

    return {
        "link_type": "link_only",
        "unique_id_column_name": "uid",
        "source_dataset_column_name": "source_dataset",
        "retain_intermediate_calculation_columns": False,
        "retain_matching_columns": True,
        "blocking_rules_to_generate_predictions": [
            "l.state = r.state and l.city = r.city",
        ],
        "comparisons": comparisons,
        "probability_two_random_records_match": 1e-6,
    }


def build_text_for_semantic(row: pd.Series, suffix: str) -> str:
    parts = [
        str(row.get(f"name_{suffix}", "") or ""),
        str(row.get(f"address_{suffix}", "") or ""),
        str(row.get(f"city_{suffix}", "") or ""),
        str(row.get(f"state_{suffix}", "") or ""),
        str(row.get(f"zip5_{suffix}", "") or ""),
        str(row.get(f"owner_{suffix}", "") or ""),
        str(row.get(f"contractor_{suffix}", "") or ""),
    ]
    return " | ".join(p for p in parts if p and p != "nan")


def normalize_project_phrase(text: str) -> str:
    t = text.upper().strip()
    t = re.sub(r"\bT\s*([0-9]+)\b", r"TERMINAL \1", t)
    t = re.sub(r"\bPH\s*([0-9]+)\b", r"PHASE \1", t)
    t = t.replace("EXP", "EXPANSION")
    t = t.replace("RENOV", "RENOVATION")
    t = " ".join(t.split())
    return t


def heuristic_semantic_score(row: pd.Series) -> float:
    name_l = normalize_project_phrase(str(row.get("name_l", "") or ""))
    name_r = normalize_project_phrase(str(row.get("name_r", "") or ""))
    addr_l = normalize_project_phrase(str(row.get("address_l", "") or ""))
    addr_r = normalize_project_phrase(str(row.get("address_r", "") or ""))

    name_ratio = fuzz.ratio(name_l, name_r) / 100.0 if name_l and name_r else 0.0
    name_token = fuzz.token_set_ratio(name_l, name_r) / 100.0 if name_l and name_r else 0.0
    name_sort = fuzz.token_sort_ratio(name_l, name_r) / 100.0 if name_l and name_r else 0.0
    name_partial = fuzz.partial_ratio(name_l, name_r) / 100.0 if name_l and name_r else 0.0

    addr_ratio = fuzz.ratio(addr_l, addr_r) / 100.0 if addr_l and addr_r else 0.0
    addr_token = fuzz.token_set_ratio(addr_l, addr_r) / 100.0 if addr_l and addr_r else 0.0

    zip_match = 1.0 if row.get("zip5_l", "") and row.get("zip5_l", "") == row.get("zip5_r", "") else 0.0
    city_match = 1.0 if row.get("city_l", "") and row.get("city_l", "") == row.get("city_r", "") else 0.0

    # Adaptive weighting avoids over-penalizing records where one side is missing address/zip.
    weighted_total = 0.0
    active_weight = 0.0

    if name_l and name_r:
        name_blend = max(
            0.40 * name_ratio + 0.25 * name_token + 0.20 * name_sort + 0.15 * name_partial,
            name_partial,
        )
        weighted_total += 0.60 * name_blend
        active_weight += 0.60
    if addr_l and addr_r:
        weighted_total += 0.20 * max(addr_ratio, addr_token)
        active_weight += 0.20
    if row.get("zip5_l", "") and row.get("zip5_r", ""):
        weighted_total += 0.15 * zip_match
        active_weight += 0.15
    if row.get("city_l", "") and row.get("city_r", ""):
        weighted_total += 0.10 * city_match
        active_weight += 0.10

    if active_weight == 0:
        return 0.0

    score = weighted_total / active_weight
    return max(0.0, min(1.0, score))


def try_ensemblelink_rerank(
    candidates: pd.DataFrame,
    model_name: str,
    device: str,
    disable_ensemblelink: bool,
) -> Tuple[pd.DataFrame, str]:
    if disable_ensemblelink:
        out = candidates.copy()
        out["ensemble_score"] = out.apply(heuristic_semantic_score, axis=1)
        return out, "heuristic_disabled"

    try:
        from ensemblelink import EnsembleLinker  # type: ignore

        model = EnsembleLinker(model_name=model_name, device=device)

        payload = candidates.copy()
        payload["query_text"] = payload.apply(lambda r: build_text_for_semantic(r, "l"), axis=1)
        payload["target_text"] = payload.apply(lambda r: build_text_for_semantic(r, "r"), axis=1)

        reranked = model.rerank(
            pairs=payload,
            query_col="query_text",
            target_col="target_text",
            top_k=None,
        )

        if isinstance(reranked, pd.DataFrame) and "ensemble_score" in reranked.columns:
            return reranked, "ensemblelink"

        out = payload.copy()
        if isinstance(reranked, pd.DataFrame) and "score" in reranked.columns:
            out["ensemble_score"] = reranked["score"].astype(float)
        else:
            out["ensemble_score"] = out.apply(heuristic_semantic_score, axis=1)
        return out, "ensemblelink_fallback_score_col"
    except Exception:
        out = candidates.copy()
        out["ensemble_score"] = out.apply(heuristic_semantic_score, axis=1)
        return out, "heuristic_import_or_runtime_fallback"


def rank_candidates_per_left(df: pd.DataFrame, score_col: str, top_k: int) -> pd.DataFrame:
    if df.empty:
        return df
    ranked = df.sort_values(["uid_l", score_col], ascending=[True, False]).copy()
    ranked["rank_within_left"] = ranked.groupby("uid_l").cumcount() + 1
    return ranked[ranked["rank_within_left"] <= top_k].copy()


def build_supplemental_fuzzy_candidates(
    a: pd.DataFrame,
    b: pd.DataFrame,
    existing_pairs: set[Tuple[str, str]],
    per_left_limit: int = 30,
    fuzzy_threshold: float = 0.68,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    b_by_block: Dict[Tuple[str, str], pd.DataFrame] = {}
    for (state, city), gdf in b.groupby(["state", "city"], dropna=False):
        b_by_block[(str(state), str(city))] = gdf

    for left in a.itertuples(index=False):
        block_key = (str(left.state), str(left.city))
        right_block = b_by_block.get(block_key)
        if right_block is None or right_block.empty:
            continue

        scored_right: List[Tuple[float, pd.Series]] = []
        for _, right in right_block.iterrows():
            pair_key = (str(left.uid), str(right["uid"]))
            if pair_key in existing_pairs:
                continue

            name_token = fuzz.token_set_ratio(str(left.name), str(right["name"])) / 100.0
            name_partial = fuzz.partial_ratio(str(left.name), str(right["name"])) / 100.0
            addr_ratio = fuzz.ratio(str(left.address), str(right["address"])) / 100.0

            fuzzy_score = max(0.60 * name_token + 0.30 * name_partial + 0.10 * addr_ratio, name_partial)
            if fuzzy_score < fuzzy_threshold:
                continue
            scored_right.append((fuzzy_score, right))

        if not scored_right:
            continue

        scored_right.sort(key=lambda x: x[0], reverse=True)
        for fuzzy_score, right in scored_right[:per_left_limit]:
            records.append(
                {
                    "uid_l": str(left.uid),
                    "uid_r": str(right["uid"]),
                    "source_dataset_l": "A",
                    "source_dataset_r": "B",
                    "name_l": str(left.name),
                    "name_r": str(right["name"]),
                    "address_l": str(left.address),
                    "address_r": str(right["address"]),
                    "city_l": str(left.city),
                    "city_r": str(right["city"]),
                    "state_l": str(left.state),
                    "state_r": str(right["state"]),
                    "zip5_l": str(left.zip5),
                    "zip5_r": str(right["zip5"]),
                    # Convert fuzzy candidate quality into a broad retrieval score band.
                    "match_probability": 0.20 + (0.70 * fuzzy_score),
                    "match_key": 999,
                }
            )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


class UnionFind:
    def __init__(self, items: Sequence[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_clusters(all_uids: Sequence[str], matched_pairs: pd.DataFrame) -> pd.DataFrame:
    uf = UnionFind(all_uids)
    for row in matched_pairs.itertuples(index=False):
        uf.union(str(row.uid_l), str(row.uid_r))

    cluster_root = {uid: uf.find(uid) for uid in all_uids}
    cluster_ids: Dict[str, int] = {}
    next_id = 1
    rows: List[Dict[str, object]] = []

    for uid in all_uids:
        root = cluster_root[uid]
        if root not in cluster_ids:
            cluster_ids[root] = next_id
            next_id += 1
        rows.append({"uid": uid, "cluster_id": cluster_ids[root]})

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    a_raw = read_input_table(args.source_a)
    b_raw = read_input_table(args.source_b)

    if args.max_rows and args.max_rows > 0:
        a_raw = a_raw.head(args.max_rows).copy()
        b_raw = b_raw.head(args.max_rows).copy()

    mapping_a = detect_columns(a_raw)
    mapping_b = detect_columns(b_raw)

    if args.print_column_mapping:
        print("Detected mapping for source A:")
        for key in sorted(mapping_a.keys()):
            print(f"  {key}: {mapping_a[key]}")
        print("Detected mapping for source B:")
        for key in sorted(mapping_b.keys()):
            print(f"  {key}: {mapping_b[key]}")

    a = preprocess(a_raw, "A", mapping_a)
    b = preprocess(b_raw, "B", mapping_b)
    combined = pd.concat([a, b], ignore_index=True)

    has_owner = combined["owner"].fillna("").astype(str).str.strip().ne("").any()
    has_contractor = combined["contractor"].fillna("").astype(str).str.strip().ne("").any()
    has_geo = combined["lat"].notna().any() and combined["lon"].notna().any()

    settings = build_splink_settings(bool(has_owner), bool(has_contractor), bool(has_geo))
    linker = DuckDBLinker(combined, settings)

    training_notes = {
        "u_estimation": "ok",
        "prior_estimation": "not_run",
    }

    linker.estimate_u_using_random_sampling(max_pairs=args.max_training_pairs)
    try:
        linker.estimate_probability_two_random_records_match(
            ["l.state = r.state and l.city = r.city and l.name = r.name"],
            recall=0.7,
        )
        training_notes["prior_estimation"] = "ok"
    except Exception as exc:
        training_notes["prior_estimation"] = f"failed: {exc}"

    # Stage 1: Retrieve all blocked candidates, then threshold/rank in Python for finer control.
    predictions = linker.predict()
    candidate_df = predictions.as_pandas_dataframe()
    if args.retrieval_threshold > 0:
        candidate_df = candidate_df[candidate_df["match_probability"] >= args.retrieval_threshold].copy()

    stage1_candidates = rank_candidates_per_left(candidate_df, "match_probability", args.retrieval_top_k)

    existing_pairs = set(zip(stage1_candidates["uid_l"].astype(str), stage1_candidates["uid_r"].astype(str)))
    supplemental = build_supplemental_fuzzy_candidates(a, b, existing_pairs)
    if not supplemental.empty:
        stage1_candidates = pd.concat([stage1_candidates, supplemental], ignore_index=True, sort=False)
        stage1_candidates = stage1_candidates.sort_values("match_probability", ascending=False)
        stage1_candidates = stage1_candidates.drop_duplicates(subset=["uid_l", "uid_r"], keep="first")
        stage1_candidates = rank_candidates_per_left(stage1_candidates, "match_probability", args.retrieval_top_k)

    # Stage 2: EnsembleLink semantic reranking (or heuristic fallback).
    stage2_scored, rerank_mode = try_ensemblelink_rerank(
        stage1_candidates,
        model_name=args.ensemble_model_name,
        device=args.device,
        disable_ensemblelink=args.disable_ensemblelink,
    )

    stage2_scored["hybrid_score"] = 0.50 * stage2_scored["match_probability"] + 0.50 * stage2_scored["ensemble_score"]

    final_top = rank_candidates_per_left(stage2_scored, "hybrid_score", args.final_top_k)
    final_matches = final_top[final_top["hybrid_score"] >= args.final_threshold].copy()
    final_matches.sort_values("hybrid_score", ascending=False, inplace=True)

    all_uids = combined["uid"].astype(str).tolist()
    cluster_df = build_clusters(all_uids, final_matches)

    stage1_candidates.to_csv(output_dir / "stage1_topk_candidates.csv", index=False)
    stage2_scored.to_csv(output_dir / "stage2_reranked_candidates.csv", index=False)
    final_matches.to_csv(output_dir / "hybrid_final_matches.csv", index=False)
    cluster_df.to_csv(output_dir / "hybrid_clusters.csv", index=False)

    if hasattr(linker, "save_model_to_json"):
        linker.save_model_to_json(output_dir / "splink_retrieval_model.json", overwrite=True)
    else:
        linker.save_settings_to_json(output_dir / "splink_retrieval_settings.json", overwrite=True)

    summary = {
        "source_a_rows": int(len(a)),
        "source_b_rows": int(len(b)),
        "total_rows": int(len(combined)),
        "max_rows_per_source": int(args.max_rows),
        "retrieval_threshold": args.retrieval_threshold,
        "retrieval_top_k": args.retrieval_top_k,
        "final_threshold": args.final_threshold,
        "final_top_k": args.final_top_k,
        "stage1_candidates": int(len(stage1_candidates)),
        "stage2_candidates": int(len(stage2_scored)),
        "final_matches": int(len(final_matches)),
        "unique_clusters": int(cluster_df["cluster_id"].nunique()) if not cluster_df.empty else 0,
        "rerank_mode": rerank_mode,
        "training_notes": training_notes,
        "active_comparison_flags": {
            "owner": bool(has_owner),
            "contractor": bool(has_contractor),
            "geo": bool(has_geo),
        },
    }

    with (output_dir / "hybrid_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("Hybrid pipeline complete")
    print(f"Stage-1 candidates (top-k): {len(stage1_candidates)}")
    print(f"Final matches: {len(final_matches)}")
    print(f"Rerank mode: {rerank_mode}")


if __name__ == "__main__":
    main()
