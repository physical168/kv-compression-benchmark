"""Evaluation manager for KV-cache compression experiments (artwork benchmark).

Aligned with CompressionExperiments ``experiment_manager/evaluation/evaluator.py``:
computes precision, recall, and F1 for results under
``results/{dataset}/{model_tag}/{press}/{ratio}/results.csv``
against ``ground_truth/query_NNN.csv`` (paths relative to ``benchmarks/artwork_eval``).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_ARTWORK_EVAL_ROOT = Path(__file__).resolve().parent.parent


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = s.strip("\"'")
    s = s.strip(".,;:!?()[]")
    return s.strip()


def _normalize_bool(answer: str) -> bool | None:
    a = _normalize(answer)
    if a in ("1", "true") or re.search(r"\byes\b", a):
        return True
    if a in ("0", "false") or re.search(r"\bno\b", a):
        return False
    if re.search(r"\b1\b", a) and not re.search(r"\b0\b", a):
        return True
    if re.search(r"\b0\b", a) and not re.search(r"\b1\b", a):
        return False
    return None


def _index_col(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c.startswith("_index_")]
    if not cols:
        raise ValueError(f"No _index_* column found. Available: {df.columns.tolist()}")
    return cols[0]


def _load_filter_gt(gt_csv: Path) -> set[int]:
    df = pd.read_csv(gt_csv)
    return set(df[_index_col(df)].astype(int))


def _answer_to_str(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return _normalize(str(value))


def _load_extract_gt(gt_csv: Path) -> dict[int, str]:
    df = pd.read_csv(gt_csv)
    col = _index_col(df)
    return {
        int(row[col]): _answer_to_str(row["answer"])
        for _, row in df.iterrows()
    }


def _match_categorical(model_answer: str, gt_values: set[str]) -> str | None:
    norm = _normalize(model_answer)
    if norm in gt_values:
        return norm
    hits = [v for v in gt_values if re.search(r"\b" + re.escape(v) + r"\b", norm)]
    return hits[0] if len(hits) == 1 else None


def _match_numeric(model_answer: str, gt_values: set[str]) -> str | None:
    for raw in re.findall(r"\b\d+(?:\.\d+)?\b", model_answer):
        candidate = str(float(raw))
        if candidate in gt_values:
            return candidate
    return None


def _is_numeric(gt_values: set[str]) -> bool:
    return bool(gt_values) and all(re.fullmatch(r"\d+(?:\.\d+)?", v) for v in gt_values)


def _prf1(y_true: list, y_pred: list, classes: list) -> tuple[float, float, float]:
    ps, rs, fs = [], [], []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        ps.append(p)
        rs.append(r)
        fs.append(f)
    n = len(classes) or 1
    return sum(ps) / n, sum(rs) / n, sum(fs) / n


class EvaluationManager:
    """Computes precision, recall, and F1 for KV-cache compression results."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        results_dir: str | Path = "results",
    ) -> None:
        if config_path is None:
            config_path = Path(__file__).parent / "evaluation_config.yaml"
        self.cfg = _load_config(Path(config_path))
        self.results_dir = Path(results_dir)

    def evaluate_all(self) -> pd.DataFrame:
        rows: list[dict] = []
        for csv_path in sorted(self.results_dir.rglob("results.csv")):
            parts = csv_path.parts
            if len(parts) < 5:
                continue
            dataset, model_tag, press, ratio_tag = (
                parts[-5], parts[-4], parts[-3], parts[-2]
            )
            if dataset not in self.cfg:
                logger.debug("No evaluation config for dataset '%s', skipping.", dataset)
                continue
            try:
                rows.extend(
                    self._evaluate_file(csv_path, dataset, model_tag, press, ratio_tag)
                )
            except Exception as exc:
                logger.error("Error evaluating %s: %s", csv_path, exc)
        return pd.DataFrame(rows)

    def _evaluate_file(
        self,
        csv_path: Path,
        dataset: str,
        model_tag: str,
        press: str,
        ratio_tag: str,
    ) -> list[dict]:
        results = pd.read_csv(csv_path)
        dataset_cfg = self.cfg[dataset]
        gt_dir = _ARTWORK_EVAL_ROOT / dataset_cfg["gt_dir"]

        typed_queries: list[tuple[str, str, str]] = []
        for query_text, query_id in dataset_cfg.get("filter_query_mapping", {}).items():
            typed_queries.append((query_text, query_id, "filter"))
        for query_text, query_id in dataset_cfg.get("extract_query_mapping", {}).items():
            typed_queries.append((query_text, query_id, "extract"))

        rows = []
        for query_text, query_id, q_type in typed_queries:
            subset = results[results["query"] == query_text]
            if subset.empty:
                logger.debug("Query '%s' not found in %s.", query_text, csv_path)
                continue

            gt_csv = gt_dir / f"{query_id}.csv"
            if not gt_csv.exists():
                logger.warning("GT file not found: %s", gt_csv)
                continue

            record_ids = subset["record_id"].astype(int).tolist()
            answers = subset["answer"].tolist()

            if q_type == "filter":
                metrics = self._filter_metrics(record_ids, answers, _load_filter_gt(gt_csv))
            else:
                metrics = self._extract_metrics(record_ids, answers, _load_extract_gt(gt_csv))

            rows.append(
                {
                    "dataset": dataset,
                    "model_tag": model_tag,
                    "press": press,
                    "ratio": float(ratio_tag),
                    "query": query_text,
                    "query_id": query_id,
                    "query_type": q_type,
                    **metrics,
                }
            )
            logger.debug(
                "[%s | %s | %s | ratio=%s | %s] P=%.3f R=%.3f F1=%.3f",
                dataset, model_tag, press, ratio_tag, query_id,
                metrics["precision"], metrics["recall"], metrics["f1"],
            )
        return rows

    def _filter_metrics(
        self,
        record_ids: list[int],
        answers: list[str],
        gt_positives: set[int],
    ) -> dict:
        y_true, y_pred = [], []
        unresolved = 0
        for rid, ans in zip(record_ids, answers):
            pred = _normalize_bool(str(ans))
            if pred is None:
                unresolved += 1
                continue
            y_true.append(rid in gt_positives)
            y_pred.append(pred)

        tp = sum(t and p for t, p in zip(y_true, y_pred))
        fp = sum(not t and p for t, p in zip(y_true, y_pred))
        fn = sum(t and not p for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": len(gt_positives),
            "unresolved": unresolved,
        }

    def _extract_metrics(
        self,
        record_ids: list[int],
        answers: list[str],
        gt: dict[int, str],
    ) -> dict:
        gt_values = set(gt.values())
        numeric = _is_numeric(gt_values)
        y_true, y_pred = [], []
        unresolved = 0
        for rid, ans in zip(record_ids, answers):
            if rid not in gt:
                continue
            pred = (
                _match_numeric(str(ans), gt_values)
                if numeric
                else _match_categorical(str(ans), gt_values)
            )
            if pred is None:
                unresolved += 1
                continue
            y_true.append(gt[rid])
            y_pred.append(pred)

        classes = sorted(set(y_true))
        precision, recall, f1 = (
            _prf1(y_true, y_pred, classes) if y_true else (0.0, 0.0, 0.0)
        )
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": len(gt),
            "unresolved": unresolved,
        }
