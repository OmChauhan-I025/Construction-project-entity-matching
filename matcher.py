from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# Canonical names used internally.
CANONICAL_COLUMNS = {
    "id": [
        "ID",
        "PROJECT_ID",
        "PROJECT ID",
        "RECORD_ID",
        "RECORD ID",
        "PROJECTID",
        "RECORDID",
    ],
    "name": ["NAME", "PROJECT_NAME", "PROJECT NAME"],
    "address": [
        "ADDRESS_LINE_1__C",
        "ADDRESS_LINE_1",
        "ADDRESS LINE 1",
        "ADDRESS",
        "ADDRESS1",
    ],
    "city": ["CITY__C", "CITY", "CITY C"],
    "state": ["STATE__C", "STATE", "STATE C"],
    "zip": ["ZIP__C", "ZIP", "ZIP C", "POSTAL_CODE", "POSTCODE", "ZIPCODE"],
    "lat": [
        "LOCATION__LATITUDE__S",
        "LOCATION LATITUDE S",
        "LOCATION_LATITUDE",
        "LATITUDE",
        "LAT",
    ],
    "lon": [
        "LOCATION__LONGITUDE__S",
        "LOCATION LONGITUDE S",
        "LOCATION_LONGITUDE",
        "LONGITUDE",
        "LON",
        "LNG",
    ],
    "estimated_value": [
        "ESTIMATED_VALUE__C",
        "ESTIMATED_VALUE",
        "ESTIMATE",
        "C_ESTIMATE",
        "C ESTIMATE",
        "PROJECT_VALUE",
    ],
    "floors": [
        "FLOORS_ABOVE_GROUND__C",
        "FLOORS",
        "FLOOR",
        "FLOOR_COUNT",
        "AF FLOORS",
        "AB FLOORS",
    ],
    "site_area": ["SITE_AREA__C", "SITE_AREA", "SITE AREA", "C SITE AREA"],
    "work_type": ["WORK_TYPE__C", "WORK_TYPE", "WORK TYPE"],
    "category": ["CATEGORY__C", "CATEGORY", "CATEGORY C"],
    "owner": [
        "OWNER",
        "OWNER C",
        "C OWNER",
        "OWNER_H_PARENT",
        "OWNERHPARENT",
        "OWNER PARENT",
        "PARENT FR",
    ],
    "contractor": [
        "CONTRACTOR",
        "CONTRACTOR C",
        "C CONTRACTOR",
        "GENERAL_CONTRACTOR",
        "GC",
    ],
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


@dataclass
class MatchingConfig:
    w_name: float = 0.35
    w_address: float = 0.25
    w_city: float = 0.15
    w_state: float = 0.05
    w_geo: float = 0.10
    w_value: float = 0.10
    w_floors: float = 0.05
    w_owner: float = 0.10
    w_contractor: float = 0.10
    auto_threshold: float = 0.90
    review_threshold: float = 0.80
    geo_candidate_max_distance_m: float = 5000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-source project deduplication with blocking + fuzzy matching.")
    parser.add_argument("--source-a", required=True, help="Path to source A file (.csv/.xlsx/.xls)")
    parser.add_argument("--source-b", required=True, help="Path to source B file (.csv/.xlsx/.xls)")
    parser.add_argument("--output-dir", default="output", help="Directory for output files")
    parser.add_argument("--id-col-a", default=None, help="Optional explicit ID column in source A")
    parser.add_argument("--id-col-b", default=None, help="Optional explicit ID column in source B")
    parser.add_argument("--auto-threshold", type=float, default=0.90, help="Score threshold for auto_match")
    parser.add_argument("--review-threshold", type=float, default=0.80, help="Score threshold for review")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Optional safety cap for candidate pairs (0 = no cap)",
    )
    parser.add_argument(
        "--print-column-mapping",
        action="store_true",
        help="Print detected source-to-canonical column mapping for each file",
    )
    parser.add_argument(
        "--geo-candidate-max-distance-m",
        type=float,
        default=5000.0,
        help="If both points exist, only keep candidate pairs within this distance (meters)",
    )
    parser.add_argument(
        "--export-llm-review",
        action="store_true",
        help="Export uncertain matches into matches_llm_review.csv for optional LLM adjudication",
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
        key = normalize_column_name(alias)
        if key in normalized_map:
            return normalized_map[key]
    return None


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).upper().strip()
    text = " ".join(text.split())
    return text


def normalize_address(value: object) -> str:
    text = f" {normalize_text(value)} "
    for source, target in STREET_ABBREVIATIONS.items():
        text = text.replace(source, target)
    return " ".join(text.strip().split())


def normalize_state(value: object) -> str:
    return normalize_text(value)


def normalize_zip(value: object) -> str:
    text = normalize_text(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:5]


def coerce_float(value: object) -> float:
    if pd.isna(value):
        return math.nan
    text = str(value).strip().replace(",", "")
    text = text.replace("$", "")
    try:
        return float(text)
    except ValueError:
        return math.nan


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    selected: Dict[str, Optional[str]] = {}
    for canonical, aliases in CANONICAL_COLUMNS.items():
        selected[canonical] = find_column(df, aliases)
    return selected


def preprocess(
    df: pd.DataFrame,
    source_tag: str,
    explicit_id_col: Optional[str],
    selected: Dict[str, Optional[str]],
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    if explicit_id_col:
        if explicit_id_col not in df.columns:
            raise ValueError(f"ID column '{explicit_id_col}' not found in {source_tag}")
        selected["id"] = explicit_id_col

    if selected["id"] is None:
        out["project_id"] = [f"{source_tag}_{i}" for i in range(len(df))]
    else:
        id_values = df[selected["id"]]
        out["project_id"] = id_values.where(id_values.notna(), "").astype(str).str.strip()
        empty_mask = out["project_id"].eq("")
        out.loc[empty_mask, "project_id"] = [f"{source_tag}_{i}" for i in out.index[empty_mask]]

    out["name"] = df[selected["name"]].map(normalize_text) if selected["name"] else ""
    out["address"] = df[selected["address"]].map(normalize_address) if selected["address"] else ""
    out["city"] = df[selected["city"]].map(normalize_text) if selected["city"] else ""
    out["state"] = df[selected["state"]].map(normalize_state) if selected["state"] else ""
    out["zip5"] = df[selected["zip"]].map(normalize_zip) if selected["zip"] else ""

    if selected["lat"]:
        out["lat"] = df[selected["lat"]].map(coerce_float)
    else:
        out["lat"] = math.nan

    if selected["lon"]:
        out["lon"] = df[selected["lon"]].map(coerce_float)
    else:
        out["lon"] = math.nan

    if selected["estimated_value"]:
        out["estimated_value"] = df[selected["estimated_value"]].map(coerce_float)
    else:
        out["estimated_value"] = math.nan

    if selected["floors"]:
        out["floors"] = df[selected["floors"]].map(coerce_float)
    else:
        out["floors"] = math.nan

    if selected["site_area"]:
        out["site_area"] = df[selected["site_area"]].map(coerce_float)
    else:
        out["site_area"] = math.nan

    out["work_type"] = df[selected["work_type"]].map(normalize_text) if selected["work_type"] else ""
    out["category"] = df[selected["category"]].map(normalize_text) if selected["category"] else ""
    out["owner"] = df[selected["owner"]].map(normalize_text) if selected["owner"] else ""
    out["contractor"] = df[selected["contractor"]].map(normalize_text) if selected["contractor"] else ""

    out["name_prefix5"] = out["name"].str[:5]
    out["geo_bucket"] = np.where(
        out["lat"].notna() & out["lon"].notna(),
        out["lat"].round(2).astype(str) + "_" + out["lon"].round(2).astype(str),
        "",
    )

    out["source"] = source_tag
    out = out.reset_index(drop=True)
    out["row_id"] = out.index.astype(int)
    return out


def block_join(
    a: pd.DataFrame,
    b: pd.DataFrame,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    block_name: str,
) -> pd.DataFrame:
    left = a[["row_id", *left_keys]].copy()
    right = b[["row_id", *right_keys]].copy()
    left.columns = ["row_id_a", *[f"k{i}" for i in range(len(left_keys))]]
    right.columns = ["row_id_b", *[f"k{i}" for i in range(len(right_keys))]]

    merge_keys = [f"k{i}" for i in range(len(left_keys))]
    for key in merge_keys:
        left = left[left[key] != ""]
        right = right[right[key] != ""]

    pairs = left.merge(right, on=merge_keys, how="inner")[["row_id_a", "row_id_b"]]
    pairs["block"] = block_name
    return pairs


def generate_candidates(a: pd.DataFrame, b: pd.DataFrame, max_candidates: int) -> pd.DataFrame:
    blocks = [
        (("state", "city"), ("state", "city"), "state_city"),
        (("state", "city", "zip5"), ("state", "city", "zip5"), "state_city_zip"),
        (("state", "city", "name_prefix5"), ("state", "city", "name_prefix5"), "state_city_nameprefix"),
    ]

    candidate_parts: List[pd.DataFrame] = []
    for l_keys, r_keys, name in blocks:
        part = block_join(a, b, l_keys, r_keys, name)
        if not part.empty:
            candidate_parts.append(part)

    if not candidate_parts:
        return pd.DataFrame(columns=["row_id_a", "row_id_b", "blocks"])

    candidates = pd.concat(candidate_parts, ignore_index=True)
    candidates = (
        candidates.groupby(["row_id_a", "row_id_b"], as_index=False)["block"]
        .agg(lambda values: ";".join(sorted(set(values))))
        .rename(columns={"block": "blocks"})
    )

    if max_candidates > 0 and len(candidates) > max_candidates:
        candidates = candidates.head(max_candidates)

    return candidates


def geo_gate_candidates(
    candidates: pd.DataFrame,
    a: pd.DataFrame,
    b: pd.DataFrame,
    max_distance_m: float,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    # Gate only when both sides have coordinates; otherwise keep the pair.
    left = a[["row_id", "lat", "lon"]].rename(columns={"row_id": "row_id_a", "lat": "lat_a", "lon": "lon_a"})
    right = b[["row_id", "lat", "lon"]].rename(columns={"row_id": "row_id_b", "lat": "lat_b", "lon": "lon_b"})
    merged = candidates.merge(left, on="row_id_a", how="left").merge(right, on="row_id_b", how="left")

    keep_mask: List[bool] = []
    for row in merged.itertuples(index=False):
        if any(math.isnan(x) for x in (row.lat_a, row.lon_a, row.lat_b, row.lon_b)):
            keep_mask.append(True)
            continue
        distance = haversine_meters(row.lat_a, row.lon_a, row.lat_b, row.lon_b)
        keep_mask.append(distance <= max_distance_m)

    return merged.loc[keep_mask, ["row_id_a", "row_id_b", "blocks"]].reset_index(drop=True)


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(math.isnan(x) for x in (lat1, lon1, lat2, lon2)):
        return math.nan
    radius = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a_val = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a_val), math.sqrt(1 - a_val))


def ratio_similarity(a_val: float, b_val: float) -> float:
    if math.isnan(a_val) or math.isnan(b_val):
        return 0.0
    denom = max(abs(a_val), abs(b_val), 1.0)
    return max(0.0, 1.0 - abs(a_val - b_val) / denom)


def floor_similarity(a_val: float, b_val: float) -> float:
    if math.isnan(a_val) or math.isnan(b_val):
        return 0.0
    diff = abs(a_val - b_val)
    if diff <= 0:
        return 1.0
    if diff <= 1:
        return 0.6
    if diff <= 2:
        return 0.3
    return 0.0


def geo_similarity(distance_m: float) -> float:
    if math.isnan(distance_m):
        return 0.0
    if distance_m <= 200:
        return 1.0
    if distance_m >= 1200:
        return 0.0
    return 1.0 - ((distance_m - 200) / 1000)


def fuzzy_similarity(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    return fuzz.WRatio(text_a, text_b) / 100.0


def name_similarity(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    # Avoid WRatio over-scoring based on short shared substrings like "CENTER".
    ratio = fuzz.ratio(text_a, text_b) / 100.0
    token_sort = fuzz.token_sort_ratio(text_a, text_b) / 100.0
    token_set = fuzz.token_set_ratio(text_a, text_b) / 100.0
    return (0.45 * ratio) + (0.35 * token_sort) + (0.20 * token_set)


def score_candidates(
    candidates: pd.DataFrame,
    a: pd.DataFrame,
    b: pd.DataFrame,
    config: MatchingConfig,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    left = a.add_suffix("_a")
    right = b.add_suffix("_b")

    merged = candidates.merge(left, left_on="row_id_a", right_on="row_id_a", how="left")
    merged = merged.merge(right, left_on="row_id_b", right_on="row_id_b", how="left")

    name_sim_list: List[float] = []
    address_sim_list: List[float] = []
    city_match_list: List[float] = []
    state_match_list: List[float] = []
    zip_match_list: List[float] = []
    geo_distance_list: List[float] = []
    geo_sim_list: List[float] = []
    value_sim_list: List[float] = []
    floors_sim_list: List[float] = []
    owner_sim_list: List[float] = []
    contractor_sim_list: List[float] = []
    total_score_list: List[float] = []
    decision_list: List[str] = []
    deterministic_rule_list: List[str] = []

    for row in merged.itertuples(index=False):
        name_sim = name_similarity(row.name_a, row.name_b)
        address_sim = fuzzy_similarity(row.address_a, row.address_b)
        city_match = 1.0 if row.city_a and row.city_a == row.city_b else 0.0
        state_match = 1.0 if row.state_a and row.state_a == row.state_b else 0.0
        zip_match = 1.0 if row.zip5_a and row.zip5_a == row.zip5_b else 0.0
        owner_sim = fuzzy_similarity(row.owner_a, row.owner_b)
        contractor_sim = fuzzy_similarity(row.contractor_a, row.contractor_b)

        distance_m = haversine_meters(row.lat_a, row.lon_a, row.lat_b, row.lon_b)
        geo_sim = geo_similarity(distance_m)

        value_sim = ratio_similarity(row.estimated_value_a, row.estimated_value_b)
        floors_sim = floor_similarity(row.floors_a, row.floors_b)

        # Adaptive scoring: ignore missing features rather than penalizing them as zero.
        weighted_total = 0.0
        active_weight = 0.0

        if row.name_a and row.name_b:
            weighted_total += config.w_name * name_sim
            active_weight += config.w_name
        if row.address_a and row.address_b:
            weighted_total += config.w_address * address_sim
            active_weight += config.w_address
        if row.city_a and row.city_b:
            weighted_total += config.w_city * city_match
            active_weight += config.w_city
        if row.state_a and row.state_b:
            weighted_total += config.w_state * state_match
            active_weight += config.w_state
        if not math.isnan(distance_m):
            weighted_total += config.w_geo * geo_sim
            active_weight += config.w_geo
        if not math.isnan(row.estimated_value_a) and not math.isnan(row.estimated_value_b):
            weighted_total += config.w_value * value_sim
            active_weight += config.w_value
        if not math.isnan(row.floors_a) and not math.isnan(row.floors_b):
            weighted_total += config.w_floors * floors_sim
            active_weight += config.w_floors
        if row.owner_a and row.owner_b:
            weighted_total += config.w_owner * owner_sim
            active_weight += config.w_owner
        if row.contractor_a and row.contractor_b:
            weighted_total += config.w_contractor * contractor_sim
            active_weight += config.w_contractor

        score = (weighted_total / active_weight) if active_weight > 0 else 0.0

        geo_close = (not math.isnan(distance_m)) and distance_m <= 300
        geo_near = (not math.isnan(distance_m)) and distance_m <= 1200
        geo_present = not math.isnan(distance_m)
        strong_text = max(name_sim, owner_sim, contractor_sim)
        deterministic_auto = state_match == 1.0 and city_match == 1.0 and geo_close and strong_text >= 0.72
        exact_name_auto = (
            state_match == 1.0
            and city_match == 1.0
            and name_sim >= 0.98
            and (zip_match == 1.0 or value_sim >= 0.95)
        )
        score_based_auto = (
            state_match == 1.0
            and city_match == 1.0
            and (
                (geo_near and strong_text >= 0.78)
                or (
                    (not geo_present)
                    and strong_text >= 0.94
                    and (address_sim >= 0.80 or zip_match == 1.0)
                )
            )
            and score >= config.auto_threshold
        )

        if deterministic_auto:
            decision = "auto_match"
            deterministic_rule = "state_city_geo_text"
        elif exact_name_auto:
            decision = "auto_match"
            deterministic_rule = "exact_name_state_city"
        elif score_based_auto:
            decision = "auto_match"
            deterministic_rule = "weighted_score_guarded"
        elif score >= config.review_threshold or (
            state_match == 1.0
            and city_match == 1.0
            and (math.isnan(distance_m) or distance_m <= 5000)
            and strong_text >= 0.65
        ):
            decision = "review"
            deterministic_rule = "state_city_geo_or_text"
        else:
            decision = "non_match"
            deterministic_rule = ""

        name_sim_list.append(name_sim)
        address_sim_list.append(address_sim)
        city_match_list.append(city_match)
        state_match_list.append(state_match)
        zip_match_list.append(zip_match)
        geo_distance_list.append(distance_m)
        geo_sim_list.append(geo_sim)
        value_sim_list.append(value_sim)
        floors_sim_list.append(floors_sim)
        owner_sim_list.append(owner_sim)
        contractor_sim_list.append(contractor_sim)
        total_score_list.append(score)
        decision_list.append(decision)
        deterministic_rule_list.append(deterministic_rule)

    merged["name_similarity"] = name_sim_list
    merged["address_similarity"] = address_sim_list
    merged["city_match"] = city_match_list
    merged["state_match"] = state_match_list
    merged["zip_match"] = zip_match_list
    merged["geo_distance_m"] = geo_distance_list
    merged["geo_similarity"] = geo_sim_list
    merged["estimated_value_similarity"] = value_sim_list
    merged["floors_similarity"] = floors_sim_list
    merged["owner_similarity"] = owner_sim_list
    merged["contractor_similarity"] = contractor_sim_list
    merged["match_score"] = total_score_list
    merged["decision"] = decision_list
    merged["deterministic_rule"] = deterministic_rule_list

    return merged


def write_outputs(scored: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def safe_to_csv(df: pd.DataFrame, path: Path, columns: Sequence[str]) -> Path:
        try:
            df.to_csv(path, index=False, columns=columns)
            return path
        except PermissionError:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback = path.with_name(f"{path.stem}_{ts}{path.suffix}")
            df.to_csv(fallback, index=False, columns=columns)
            print(f"Warning: file locked '{path.name}', wrote '{fallback.name}' instead")
            return fallback

    cols = [
        "project_id_a",
        "project_id_b",
        "name_a",
        "name_b",
        "owner_a",
        "owner_b",
        "contractor_a",
        "contractor_b",
        "address_a",
        "address_b",
        "city_a",
        "city_b",
        "state_a",
        "state_b",
        "zip5_a",
        "zip5_b",
        "match_score",
        "decision",
        "blocks",
        "name_similarity",
        "address_similarity",
        "city_match",
        "state_match",
        "zip_match",
        "geo_distance_m",
        "geo_similarity",
        "estimated_value_similarity",
        "floors_similarity",
        "owner_similarity",
        "contractor_similarity",
        "deterministic_rule",
    ]
    cols = [c for c in cols if c in scored.columns]

    safe_to_csv(scored.sort_values("match_score", ascending=False), output_dir / "matches_scored.csv", cols)
    safe_to_csv(scored[scored["decision"] == "auto_match"], output_dir / "matches_auto.csv", cols)
    safe_to_csv(scored[scored["decision"] == "review"], output_dir / "matches_review.csv", cols)

    summary = {
        "total_candidates": int(len(scored)),
        "auto_match": int((scored["decision"] == "auto_match").sum()) if "decision" in scored else 0,
        "review": int((scored["decision"] == "review").sum()) if "decision" in scored else 0,
        "non_match": int((scored["decision"] == "non_match").sum()) if "decision" in scored else 0,
    }
    summary_path = output_dir / "summary.json"
    try:
        with summary_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_summary = output_dir / f"summary_{ts}.json"
        with fallback_summary.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        print(f"Warning: file locked '{summary_path.name}', wrote '{fallback_summary.name}' instead")


def write_llm_review_export(scored: pd.DataFrame, output_dir: Path) -> None:
    if scored.empty:
        return

    review = scored[scored["decision"] == "review"].copy()
    if review.empty:
        return

    review = review.sort_values("match_score", ascending=False)
    review["llm_prompt"] = (
        "Decide whether A and B are the same project. Use name, owner, contractor, address, city/state, zip, "
        "and geo distance. Output only MATCH or NON_MATCH. A="
        + review["name_a"].fillna("")
        + " | "
        + review["owner_a"].fillna("")
        + " | "
        + review["contractor_a"].fillna("")
        + " || B="
        + review["name_b"].fillna("")
        + " | "
        + review["owner_b"].fillna("")
        + " | "
        + review["contractor_b"].fillna("")
        + " || city/state="
        + review["city_a"].fillna("")
        + "/"
        + review["state_a"].fillna("")
        + " || geo_distance_m="
        + review["geo_distance_m"].fillna(-1).round(2).astype(str)
    )
    llm_path = output_dir / "matches_llm_review.csv"
    try:
        review.to_csv(llm_path, index=False)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = output_dir / f"matches_llm_review_{ts}.csv"
        review.to_csv(fallback, index=False)
        print(f"Warning: file locked '{llm_path.name}', wrote '{fallback.name}' instead")


def main() -> None:
    args = parse_args()
    cfg = MatchingConfig(
        auto_threshold=args.auto_threshold,
        review_threshold=args.review_threshold,
        geo_candidate_max_distance_m=args.geo_candidate_max_distance_m,
    )

    source_a_raw = read_input_table(args.source_a)
    source_b_raw = read_input_table(args.source_b)

    selected_a = detect_columns(source_a_raw)
    selected_b = detect_columns(source_b_raw)

    if args.print_column_mapping:
        print("Detected column mapping for source A:")
        for key in sorted(selected_a.keys()):
            print(f"  {key}: {selected_a[key]}")
        print("Detected column mapping for source B:")
        for key in sorted(selected_b.keys()):
            print(f"  {key}: {selected_b[key]}")

    strong_fields = ["name", "address", "city", "state", "zip", "lat", "lon"]
    for label, mapping in (("A", selected_a), ("B", selected_b)):
        missing = [field for field in strong_fields if mapping.get(field) is None]
        if missing:
            print(f"Warning: source {label} is missing strong fields: {', '.join(missing)}")

    a = preprocess(source_a_raw, "A", args.id_col_a, selected_a)
    b = preprocess(source_b_raw, "B", args.id_col_b, selected_b)

    candidates = generate_candidates(a, b, args.max_candidates)
    candidates = geo_gate_candidates(candidates, a, b, cfg.geo_candidate_max_distance_m)
    scored = score_candidates(candidates, a, b, cfg)

    write_outputs(scored, Path(args.output_dir))
    if args.export_llm_review:
        write_llm_review_export(scored, Path(args.output_dir))

    print("Run complete")
    print(f"Source A rows: {len(a)}")
    print(f"Source B rows: {len(b)}")
    print(f"Candidates scored: {len(scored)}")
    if not scored.empty:
        print(f"Auto matches: {(scored['decision'] == 'auto_match').sum()}")
        print(f"Review: {(scored['decision'] == 'review').sum()}")


if __name__ == "__main__":
    main()
