from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
	from openai import OpenAI
except ImportError as exc:  # pragma: no cover - import guard for local setup
	raise ImportError(
		"The 'openai' package is required. Install with: pip install openai"
	) from exc


REQUIRED_FIELDS = [
	"name",
	"address",
	"city",
	"state",
	"latitude",
	"longitude",
	"worktype",
	"ownership",
]

FIELD_ALIASES: Dict[str, Sequence[str]] = {
	"record_id": ("id", "record_id", "recordid", "permit_id", "project_id"),
	"name": ("name", "project_name", "job_name", "site_name"),
	"address": (
		"address",
		"address_line_1",
		"address1",
		"street_address",
		"location_address",
	),
	"city": ("city", "town", "municipality"),
	"state": ("state", "province", "region"),
	"latitude": ("lat", "latitude", "y", "coord_y"),
	"longitude": ("lon", "lng", "long", "longitude", "x", "coord_x"),
	"worktype": ("worktype", "work_type", "permit_type", "job_type"),
	"ownership": ("ownership", "owner_type", "owner", "project_owner"),
}

WORKTYPE_NORMALIZATION = {
	"NEW": "NEW",
	"NEW CONSTRUCTION": "NEW",
	"ALETRATION": "ALTERATION",
	"ALTRATION": "ALTERATION",
	"ALTERATION": "ALTERATION",
	"ALTERATIONS": "ALTERATION",
}

OWNERSHIP_NORMALIZATION = {
	"PUBLIC": "PUBLIC",
	"PRIVATE": "PRIVATE",
	"LOCAL": "LOCAL",
	"LOCAL GOVT": "LOCAL",
	"LOCAL GOVERNMENT": "LOCAL",
	"MUNICIPAL": "LOCAL",
	"GOVERNMENT": "GOVERNMENT",
	"FEDERAL": "GOVERNMENT",
	"STATE GOVERNMENT": "GOVERNMENT",
}


@dataclass(frozen=True)
class MatchWeights:
	embedding: float = 4.0
	geo: float = 2.0
	worktype: float = 1.0
	ownership: float = 1.0

	@property
	def total(self) -> float:
		return self.embedding + self.geo + self.worktype + self.ownership


@dataclass
class MatchConfig:
	model: str = "text-embedding-3-small"
	batch_size: int = 100
	retries: int = 3
	right_sep: str = ","
	top_n: int = 300
	max_pairs: Optional[int] = None
	weights: MatchWeights = MatchWeights()


def normalize_key(value: str) -> str:
	return "".join(ch for ch in str(value).lower() if ch.isalnum())


def normalize_text(value: object) -> str:
	if value is None or (isinstance(value, float) and math.isnan(value)):
		return ""
	text = " ".join(str(value).strip().split())
	return text.upper()


def normalize_worktype(value: object) -> str:
	normalized = normalize_text(value)
	return WORKTYPE_NORMALIZATION.get(normalized, normalized)


def normalize_ownership(value: object) -> str:
	normalized = normalize_text(value)
	return OWNERSHIP_NORMALIZATION.get(normalized, normalized)


def coerce_float(value: object) -> float:
	if value is None:
		return float("nan")
	if isinstance(value, (int, float, np.number)):
		return float(value)
	text = str(value).strip()
	if not text:
		return float("nan")
	text = text.replace(",", "")
	try:
		return float(text)
	except ValueError:
		return float("nan")


def detect_field_columns(df: pd.DataFrame, overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
	overrides = overrides or {}
	normalized_columns = {normalize_key(col): col for col in df.columns}
	resolved: Dict[str, str] = {}

	for field, alias_list in FIELD_ALIASES.items():
		if field in overrides:
			override_col = overrides[field]
			if override_col not in df.columns:
				raise ValueError(f"Override column '{override_col}' not found for field '{field}'.")
			resolved[field] = override_col
			continue

		for alias in alias_list:
			alias_key = normalize_key(alias)
			if alias_key in normalized_columns:
				resolved[field] = normalized_columns[alias_key]
				break

	missing_required = [field for field in REQUIRED_FIELDS if field not in resolved]
	if missing_required:
		raise ValueError(
			"Missing required fields after alias detection: "
			+ ", ".join(missing_required)
			+ ". Provide matching columns or update aliases."
		)

	if "record_id" not in resolved:
		resolved["record_id"] = "__generated_id__"
	return resolved


def load_dataset(path: str, txt_sep: str = ",") -> pd.DataFrame:
	ext = os.path.splitext(path)[1].lower()
	if ext in (".xlsx", ".xls"):
		return pd.read_excel(path)
	if ext in (".txt", ".csv"):
		return pd.read_csv(path, sep=txt_sep)
	raise ValueError(f"Unsupported file extension '{ext}' for file: {path}")


def standardize_dataset(
	df: pd.DataFrame,
	source_label: str,
	overrides: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
	mapping = detect_field_columns(df, overrides)
	out = pd.DataFrame(index=df.index)
	out["source"] = source_label

	# Keep the 2nd physical column as requested project id for output traceability.
	if len(df.columns) >= 2:
		second_col = df.columns[1]
		second_values = df.iloc[:, 1]
		out["project_id"] = second_values.astype(str).str.strip()
		empty_project_id = out["project_id"].eq("") | out["project_id"].isna()
		out.loc[empty_project_id, "project_id"] = [
			f"{source_label}_P2_{idx}" for idx in out.index[empty_project_id].tolist()
		]
	else:
		second_col = None
		out["project_id"] = [f"{source_label}_P2_{idx}" for idx in range(len(df))]

	if mapping["record_id"] == "__generated_id__":
		out["record_id"] = [f"{source_label}_{idx}" for idx in range(len(df))]
	else:
		out["record_id"] = df[mapping["record_id"]].astype(str).str.strip()
		empty_ids = out["record_id"].eq("") | out["record_id"].isna()
		out.loc[empty_ids, "record_id"] = [
			f"{source_label}_{idx}" for idx in out.index[empty_ids].tolist()
		]

	out["name"] = df[mapping["name"]].map(normalize_text)
	out["address"] = df[mapping["address"]].map(normalize_text)
	out["city"] = df[mapping["city"]].map(normalize_text)
	out["state"] = df[mapping["state"]].map(normalize_text)
	out["latitude"] = df[mapping["latitude"]].map(coerce_float)
	out["longitude"] = df[mapping["longitude"]].map(coerce_float)
	out["worktype"] = df[mapping["worktype"]].map(normalize_worktype)
	out["ownership"] = df[mapping["ownership"]].map(normalize_ownership)
	out["combined_text"] = (
		out["name"]
		+ ", "
		+ out["address"]
		+ ", "
		+ out["city"]
		+ ", "
		+ out["state"]
	)

	return out.reset_index(drop=True)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	if any(math.isnan(v) for v in (lat1, lon1, lat2, lon2)):
		return float("nan")
	radius_km = 6371.0
	phi1 = math.radians(lat1)
	phi2 = math.radians(lat2)
	d_phi = math.radians(lat2 - lat1)
	d_lambda = math.radians(lon2 - lon1)
	a = (
		math.sin(d_phi / 2.0) ** 2
		+ math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
	)
	c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
	return radius_km * c


def inverse_geo_score(distance_km: float) -> float:
	if math.isnan(distance_km):
		return 0.0
	return 1.0 / (1.0 + max(distance_km, 0.0))


def exact_match_score(left: str, right: str) -> float:
	if not left or not right:
		return 0.0
	return 1.0 if left == right else 0.0


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(vectors, axis=1, keepdims=True)
	norms = np.where(norms == 0.0, 1.0, norms)
	return vectors / norms


class EmbeddingClient:
	def __init__(
		self,
		model: str,
		batch_size: int,
		retries: int,
		api_key: Optional[str] = None,
		base_url: Optional[str] = None,
	):
		self.model = model
		self.batch_size = batch_size
		self.retries = retries
		key = api_key or os.getenv("OPENAI_API_KEY")
		if not key:
			raise ValueError(
				"OPENAI_API_KEY is not set. Provide --api-key or set OPENAI_API_KEY in environment."
			)
		clean_base_url = (base_url or "").strip() or os.getenv("OPENAI_BASE_URL")
		if key.startswith("sk-or-v1") and not clean_base_url:
			raise ValueError(
				"Detected a provider-style key (sk-or-v1...) but OPENAI_BASE_URL is not set. "
				"If using OpenAI Platform, use an OpenAI API key. "
				"If using another OpenAI-compatible provider, set OPENAI_BASE_URL in .env."
			)
		if clean_base_url:
			self.client = OpenAI(api_key=key, base_url=clean_base_url)
		else:
			self.client = OpenAI(api_key=key)

	def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
		cache: Dict[str, np.ndarray] = {}
		unique_texts: List[str] = []
		for text in texts:
			if text not in cache:
				cache[text] = np.array([], dtype=np.float64)
				unique_texts.append(text)

		for start in range(0, len(unique_texts), self.batch_size):
			batch = unique_texts[start : start + self.batch_size]
			response = self._with_retry(batch)
			for item, text in zip(response.data, batch):
				cache[text] = np.array(item.embedding, dtype=np.float64)

		embedded = [cache[text] for text in texts]
		return np.vstack(embedded) if embedded else np.empty((0, 0), dtype=np.float64)

	def _with_retry(self, batch: Sequence[str]):
		wait_s = 1.0
		last_err: Optional[Exception] = None
		for attempt in range(self.retries + 1):
			try:
				return self.client.embeddings.create(model=self.model, input=list(batch))
			except Exception as exc:  # pragma: no cover - API/network path
				if self._is_auth_error(exc):
					raise RuntimeError(
						"Authentication failed for embeddings provider. "
						"Check OPENAI_API_KEY and OPENAI_BASE_URL in .env."
					) from exc
				last_err = exc
				if attempt >= self.retries:
					break
				time.sleep(wait_s)
				wait_s *= 2.0
		raise RuntimeError(f"Embedding request failed after retries: {last_err}")

	@staticmethod
	def _is_auth_error(exc: Exception) -> bool:
		status_code = getattr(exc, "status_code", None)
		message = str(exc).lower()
		if status_code == 401:
			return True
		return "invalid_api_key" in message or "incorrect api key provided" in message


def compute_embedding_similarity_matrix(left_emb: np.ndarray, right_emb: np.ndarray) -> np.ndarray:
	if left_emb.size == 0 or right_emb.size == 0:
		return np.zeros((left_emb.shape[0], right_emb.shape[0]), dtype=np.float64)
	left_unit = l2_normalize(left_emb)
	right_unit = l2_normalize(right_emb)
	cosine = left_unit @ right_unit.T
	return np.clip(cosine, 0.0, 1.0)


def score_all_pairs(
	left_df: pd.DataFrame,
	right_df: pd.DataFrame,
	embedding_sim: np.ndarray,
	weights: MatchWeights,
) -> pd.DataFrame:
	rows: List[Dict[str, object]] = []
	n_left = len(left_df)
	n_right = len(right_df)

	left_lat = left_df["latitude"].to_numpy(dtype=np.float64)
	left_lon = left_df["longitude"].to_numpy(dtype=np.float64)
	right_lat = right_df["latitude"].to_numpy(dtype=np.float64)
	right_lon = right_df["longitude"].to_numpy(dtype=np.float64)

	left_worktype = left_df["worktype"].to_numpy(dtype=object)
	right_worktype = right_df["worktype"].to_numpy(dtype=object)
	left_ownership = left_df["ownership"].to_numpy(dtype=object)
	right_ownership = right_df["ownership"].to_numpy(dtype=object)

	for i in range(n_left):
		for j in range(n_right):
			emb_score = float(embedding_sim[i, j])
			km = haversine_km(left_lat[i], left_lon[i], right_lat[j], right_lon[j])
			geo_score = inverse_geo_score(km)
			worktype_score = exact_match_score(str(left_worktype[i]), str(right_worktype[j]))
			ownership_score = exact_match_score(str(left_ownership[i]), str(right_ownership[j]))

			final_score = (
				weights.embedding * emb_score
				+ weights.geo * geo_score
				+ weights.worktype * worktype_score
				+ weights.ownership * ownership_score
			) / weights.total

			rows.append(
				{
					"project_id_left": left_df.at[i, "project_id"],
					"project_id_right": right_df.at[j, "project_id"],
					"left_name": left_df.at[i, "name"],
					"right_name": right_df.at[j, "name"],
					"embedding_cosine_score": emb_score,
					"geo_inverse_score": geo_score,
					"worktype_score": worktype_score,
					"ownership_score": ownership_score,
					"distance_km": km,
					"final_weighted_score": final_score,
				}
			)

	scored = pd.DataFrame(rows)
	if scored.empty:
		return scored
	return scored.sort_values(
		by=["final_weighted_score", "embedding_cosine_score"], ascending=False
	).reset_index(drop=True)


def parse_overrides(raw_items: Optional[Iterable[str]]) -> Dict[str, str]:
	overrides: Dict[str, str] = {}
	if not raw_items:
		return overrides
	for item in raw_items:
		if "=" not in item:
			raise ValueError(f"Invalid override '{item}'. Expected format: logical_field=column_name")
		key, value = item.split("=", 1)
		logical_key = normalize_key(key)
		if logical_key not in FIELD_ALIASES:
			raise ValueError(f"Unknown logical field in override '{item}'.")
		overrides[logical_key] = value
	return overrides


def parse_int_env(var_name: str) -> Optional[int]:
	raw_value = os.getenv(var_name)
	if raw_value is None or raw_value.strip() == "":
		return None
	try:
		return int(raw_value)
	except ValueError as exc:
		raise ValueError(f"Environment variable {var_name} must be an integer.") from exc


def value_from_arg_or_env(arg_value: Optional[str], env_var: str) -> Optional[str]:
	if arg_value is not None and str(arg_value).strip() != "":
		return arg_value
	return os.getenv(env_var)


def run_matching(
	left_path: str,
	right_path: str,
	output_path: str,
	config: MatchConfig,
	api_key: Optional[str] = None,
	base_url: Optional[str] = None,
	left_overrides: Optional[Dict[str, str]] = None,
	right_overrides: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
	left_raw = load_dataset(left_path, txt_sep=config.right_sep)
	right_raw = load_dataset(right_path, txt_sep=config.right_sep)

	left = standardize_dataset(left_raw, source_label="LEFT", overrides=left_overrides)
	right = standardize_dataset(right_raw, source_label="RIGHT", overrides=right_overrides)
	left = left.head(config.top_n).copy()
	right = right.head(config.top_n).copy()

	pair_count = len(left) * len(right)
	if config.max_pairs is not None and pair_count > config.max_pairs:
		raise ValueError(
			f"Pair count {pair_count} exceeds --max-pairs limit {config.max_pairs}. "
			"Use a larger limit or reduce input sizes."
		)
	if pair_count > 2_000_000:
		print(
			"Warning: scoring all pairs is O(N*M) and can be slow/memory heavy. "
			f"Current pair count: {pair_count}."
		)

	embedding_client = EmbeddingClient(
		model=config.model,
		batch_size=config.batch_size,
		retries=config.retries,
		api_key=api_key,
		base_url=base_url,
	)
	left_emb = embedding_client.embed_texts(left["combined_text"].tolist())
	right_emb = embedding_client.embed_texts(right["combined_text"].tolist())
	emb_sim = compute_embedding_similarity_matrix(left_emb, right_emb)

	scored = score_all_pairs(left, right, emb_sim, config.weights)
	scored.to_csv(output_path, index=False)
	return scored


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Probabilistic all-pairs matcher using embedding cosine, inverse geo distance, worktype, and ownership."
	)
	parser.add_argument("--left", required=False, help="Path to left dataset (Excel/CSV/TXT)")
	parser.add_argument("--right", required=False, help="Path to right dataset (Excel/CSV/TXT)")
	parser.add_argument(
		"--output",
		default=None,
		help="Output CSV path for all pair scores (default: pair_scores.csv)",
	)
	parser.add_argument(
		"--right-sep",
		default=None,
		help="Delimiter for TXT/CSV files (default: comma)",
	)
	parser.add_argument(
		"--model",
		default=None,
		help="OpenAI embedding model name",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=None,
		help="Embedding batch size (default: 100)",
	)
	parser.add_argument(
		"--retries",
		type=int,
		default=None,
		help="Retries for embedding API calls (default: 3)",
	)
	parser.add_argument(
		"--api-key",
		default=None,
		help="Optional OpenAI API key (else uses OPENAI_API_KEY environment variable)",
	)
	parser.add_argument(
		"--base-url",
		default=None,
		help="Optional OpenAI-compatible provider base URL (else uses OPENAI_BASE_URL)",
	)
	parser.add_argument(
		"--top-n",
		type=int,
		default=None,
		help="Use only top N rows from each dataset (default: 300)",
	)
	parser.add_argument(
		"--max-pairs",
		type=int,
		default=None,
		help="Optional safety cap for total pair count",
	)
	parser.add_argument(
		"--left-col",
		action="append",
		default=None,
		help="Left dataset logical column override, format: logical_field=column_name",
	)
	parser.add_argument(
		"--right-col",
		action="append",
		default=None,
		help="Right dataset logical column override, format: logical_field=column_name",
	)
	return parser


def main() -> None:
	load_dotenv()
	parser = build_parser()
	args = parser.parse_args()

	left_path = value_from_arg_or_env(args.left, "LEFT_DATASET_PATH")
	right_path = value_from_arg_or_env(args.right, "RIGHT_DATASET_PATH")
	output_path = value_from_arg_or_env(args.output, "OUTPUT_PATH") or "pair_scores.csv"
	right_sep = value_from_arg_or_env(args.right_sep, "TXT_DELIMITER") or ","
	model = value_from_arg_or_env(args.model, "OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
	api_key = value_from_arg_or_env(args.api_key, "OPENAI_API_KEY")
	base_url = value_from_arg_or_env(args.base_url, "OPENAI_BASE_URL")

	batch_size = args.batch_size
	if batch_size is None:
		batch_size = parse_int_env("EMBEDDING_BATCH_SIZE") or 100

	retries = args.retries
	if retries is None:
		retries = parse_int_env("EMBEDDING_RETRIES") or 3

	top_n = args.top_n
	if top_n is None:
		top_n = parse_int_env("TOP_N") or 300

	max_pairs = args.max_pairs
	if max_pairs is None:
		max_pairs = parse_int_env("MAX_PAIRS")

	if not left_path or not right_path:
		raise ValueError(
			"Dataset paths are missing. Provide --left/--right or set LEFT_DATASET_PATH and RIGHT_DATASET_PATH in .env"
		)

	config = MatchConfig(
		model=model,
		batch_size=batch_size,
		retries=retries,
		right_sep=right_sep,
		top_n=top_n,
		max_pairs=max_pairs,
		weights=MatchWeights(embedding=4.0, geo=2.0, worktype=1.0, ownership=1.0),
	)

	left_overrides = parse_overrides(args.left_col)
	right_overrides = parse_overrides(args.right_col)

	scored = run_matching(
		left_path=left_path,
		right_path=right_path,
		output_path=output_path,
		config=config,
		api_key=api_key,
		base_url=base_url,
		left_overrides=left_overrides,
		right_overrides=right_overrides,
	)
	print(f"Scored pairs: {len(scored)}")
	print(f"Rows used from each dataset (top_n): {top_n}")
	print(f"Output written to: {output_path}")


if __name__ == "__main__":
	main()
