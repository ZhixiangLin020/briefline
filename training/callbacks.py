"""Checkpoint, evaluation-log, and AdaLoRA callbacks from the original run."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import TrainerCallback


class SaveBestPeftCallback(TrainerCallback):
    """
    Save the top-k best PEFT adapters.

    Recommended setting:
        metric_name="auto"

    Priority order in auto mode:
        1) eval_combo_mover_score
        2) eval_kptime_mover_final_score
        3) eval_cnn_dm_mover_score
        4) eval_valid_mover_score

    If a specific metric_name is provided, only that metric is used.
    """

    def __init__(
        self,
        output_dir: str,
        metric_name: str = "auto",
        tokenizer=None,
        greater_is_better: bool = True,
        top_k: int = 2,
        resume_existing: bool = False,
    ):
        self.output_dir = output_dir
        self.metric_name = metric_name
        self.tokenizer = tokenizer
        self.greater_is_better = bool(greater_is_better)
        self.top_k = max(1, int(top_k))

        self.best_records: List[Dict[str, Any]] = []
        if resume_existing:
            self._load_existing_best_records()

    @property
    def _manifest_path(self) -> Path:
        return Path(self.output_dir) / "best_k_metrics.json"

    def _resolve_record_path(self, raw_path: Any) -> Path:
        if raw_path is None or not str(raw_path).strip():
            raise ValueError("best_k_metrics.json contains an empty adapter path")

        output_root = Path(self.output_dir).resolve()
        path = Path(str(raw_path))
        candidates = [Path(self.output_dir) / path.name, path]

        existing = next((candidate for candidate in candidates if candidate.exists()), None)
        if existing is None:
            raise FileNotFoundError(
                "Cannot restore top-k history because an adapter directory is missing: "
                f"{raw_path}"
            )

        resolved = existing.resolve()
        if resolved != output_root and output_root not in resolved.parents:
            raise ValueError(
                "Refusing to restore a top-k adapter outside best_model_dir: "
                f"{resolved}"
            )
        return existing

    def _load_existing_best_records(self) -> None:
        manifest_path = self._manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(
                "resume_from_checkpoint was requested, but the historical top-k "
                f"manifest is missing: {manifest_path}"
            )

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot restore top-k history from {manifest_path}: {exc}"
            ) from exc

        expected = {
            "metric_name": self.metric_name,
            "greater_is_better": self.greater_is_better,
            "top_k": self.top_k,
        }
        actual = {
            "metric_name": payload.get("metric_name"),
            "greater_is_better": payload.get("greater_is_better"),
            "top_k": payload.get("top_k"),
        }
        if actual != expected:
            raise ValueError(
                "Existing top-k manifest is incompatible with the frozen callback "
                f"configuration: expected={expected}, actual={actual}"
            )

        raw_records = payload.get("best_records")
        if not isinstance(raw_records, list):
            raise ValueError("best_k_metrics.json field 'best_records' must be a list")

        records: List[Dict[str, Any]] = []
        seen_steps = set()
        for index, raw in enumerate(raw_records):
            if not isinstance(raw, dict):
                raise ValueError(
                    f"best_k_metrics.json record {index} must be an object"
                )
            try:
                metric = float(raw["metric"])
                step = int(raw["step"])
                metric_name = str(raw["metric_name"])
                path = self._resolve_record_path(raw["path"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid best_k_metrics.json record {index}: {raw}"
                ) from exc
            if not math.isfinite(metric):
                raise ValueError(
                    f"best_k_metrics.json record {index} has a non-finite metric"
                )
            if step in seen_steps:
                raise ValueError(
                    f"best_k_metrics.json contains duplicate global step {step}"
                )
            seen_steps.add(step)
            records.append(
                {
                    "metric": metric,
                    "metric_name": metric_name,
                    "step": step,
                    "path": str(path),
                }
            )

        self.best_records = records
        self._sort_records()
        if len(self.best_records) > self.top_k:
            raise ValueError(
                f"best_k_metrics.json contains {len(self.best_records)} records, "
                f"more than frozen top_k={self.top_k}"
            )
        print(
            f"[SaveBestPeftCallback] Restored {len(self.best_records)} "
            f"top-{self.top_k} records from {manifest_path}",
            flush=True,
        )

    def _resolve_metric_name(self, metrics: Dict[str, Any]) -> Optional[str]:
        if self.metric_name != "auto":
            return self.metric_name

        for name in ["eval_combo_mover_score", "eval_kptime_mover_final_score", "eval_cnn_dm_mover_score", "eval_valid_mover_score"]:
            if name in metrics:
                return name
        return None

    def _sort_records(self):
        self.best_records = sorted(
            self.best_records,
            key=lambda x: x["metric"],
            reverse=self.greater_is_better,
        )

    def _is_good_enough_to_save(self, value: float) -> bool:
        if len(self.best_records) < self.top_k:
            return True

        self._sort_records()
        worst = self.best_records[-1]["metric"]

        if self.greater_is_better:
            return value > worst

        return value < worst

    def _write_best_k_json(self):
        os.makedirs(self.output_dir, exist_ok=True)

        self._sort_records()

        payload = {
            "metric_name": self.metric_name,
            "greater_is_better": self.greater_is_better,
            "top_k": self.top_k,
            "best_records": [
                {
                    "rank": i + 1,
                    "metric": float(r["metric"]),
                    "metric_name": r["metric_name"],
                    "step": int(r["step"]),
                    "path": r["path"],
                }
                for i, r in enumerate(self.best_records)
            ],
        }

        with open(
            os.path.join(self.output_dir, "best_k_metrics.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if not state.is_world_process_zero:
            return control

        metrics = metrics or {}
        metric_name = self._resolve_metric_name(metrics)

        if metric_name is None or metric_name not in metrics:
            print(
                "[SaveBestPeftCallback] target metric not found. "
                f"metric_name={self.metric_name!r}, available metrics={sorted(metrics.keys())}",
                flush=True,
            )
            return control

        value = metrics[metric_name]

        if value is None:
            return control

        value = float(value)

        if not math.isfinite(value):
            print(
                f"[SaveBestPeftCallback] metric {metric_name} is not finite: {value}",
                flush=True,
            )
            return control

        if not self._is_good_enough_to_save(value):
            self._sort_records()
            worst = self.best_records[-1]

            print(
                f"[SaveBestPeftCallback] not in top-{self.top_k}: "
                f"{metric_name}={value:.6f}, "
                f"current_worst_top{self.top_k}={worst['metric']:.6f} "
                f"at step={worst['step']}",
                flush=True,
            )
            return control

        if model is None:
            model = kwargs.get("model", None)

        if model is None:
            print(
                "[SaveBestPeftCallback] model is None; cannot save adapter.",
                flush=True,
            )
            return control

        step = int(state.global_step)

        os.makedirs(self.output_dir, exist_ok=True)

        safe_metric_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", metric_name)

        save_dir = os.path.join(
            self.output_dir,
            f"step-{step}-{safe_metric_name}-{value:.6f}",
        )

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        print(
            f"\n[SaveBestPeftCallback] New top-{self.top_k} model: "
            f"{metric_name}={value:.6f} at step={step}. "
            f"Saving to {save_dir}",
            flush=True,
        )

        model.save_pretrained(save_dir)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_dir)

        replaced_records = [
            old for old in self.best_records if int(old["step"]) == step
        ]
        self.best_records = [
            old for old in self.best_records if int(old["step"]) != step
        ]

        record = {
            "metric": value,
            "metric_name": metric_name,
            "step": step,
            "path": save_dir,
        }

        self.best_records.append(record)
        self._sort_records()

        removed_records = self.best_records[self.top_k:]
        self.best_records = self.best_records[: self.top_k]

        for r in replaced_records + removed_records:
            path = r.get("path")
            if path and os.path.abspath(path) != os.path.abspath(save_dir) and os.path.exists(path):
                print(
                    f"[SaveBestPeftCallback] Removing non-top-{self.top_k} model: "
                    f"step={r['step']}, metric={r['metric']:.6f}, path={path}",
                    flush=True,
                )
                shutil.rmtree(path)

        self._write_best_k_json()

        print("[SaveBestPeftCallback] Current best records:", flush=True)
        for i, r in enumerate(self.best_records, start=1):
            print(
                f"  rank={i} step={r['step']} "
                f"{r['metric_name']}={r['metric']:.6f} "
                f"path={r['path']}",
                flush=True,
            )

        return control


class EvalTableSaveCallback(TrainerCallback):
    """
    Append Trainer-style metrics to CSV and JSONL after each evaluation.
    CSV is convenient for inspection; JSONL preserves the complete raw metrics.
    """

    def __init__(self, output_dir="autodl-tmp/out/logs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.csv_path = os.path.join(self.output_dir, "eval_table.csv")
        self.jsonl_path = os.path.join(self.output_dir, "eval_metrics.jsonl")

        self.columns = [
            "step",
            "epoch",

            "training_loss",
            "validation_loss",
            "cnn_dm_loss",
            "kptime_loss",
            "combo_loss",

            "cnn_dm_mover_score",
            "kptime_mover_category_score",
            "kptime_mover_keyword_score",
            "kptime_mover_final_score",
            "combo_mover_score",

            "learning_rate",
        ]

    def _safe_float(self, x):
        if x is None:
            return None
        try:
            x = float(x)
            if math.isnan(x) or math.isinf(x):
                return None
            return x
        except Exception:
            return None

    def _latest_train_log(self, state):
        """
        Find the most recent training loss and learning rate.
        Evaluation metrics usually omit training loss, so retrieve the latest
        training entry from state.log_history.
        """
        log_history = getattr(state, "log_history", []) or []

        for item in reversed(log_history):
            if "loss" in item and not any(k.startswith("eval_") for k in item.keys()):
                return item

        return {}

    def _write_csv_row(self, row):
        file_exists = os.path.exists(self.csv_path)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)

            if not file_exists:
                writer.writeheader()

            writer.writerow({k: row.get(k, None) for k in self.columns})

    def _write_jsonl(self, payload):
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero:
            return control

        metrics = metrics or {}
        train_log = self._latest_train_log(state)

        row = {
            "step": int(getattr(state, "global_step", 0) or 0),
            "epoch": self._safe_float(getattr(state, "epoch", None)),

            "training_loss": self._safe_float(train_log.get("loss")),
            "validation_loss": self._safe_float(metrics.get("eval_loss")),
            "cnn_dm_loss": self._safe_float(metrics.get("eval_cnn_dm_loss")),
            "kptime_loss": self._safe_float(metrics.get("eval_kptime_loss")),
            "combo_loss": self._safe_float(metrics.get("eval_combo_loss")),

            "cnn_dm_mover_score": self._safe_float(metrics.get("eval_cnn_dm_mover_score")),
            "kptime_mover_category_score": self._safe_float(metrics.get("eval_kptime_mover_category_score")),
            "kptime_mover_keyword_score": self._safe_float(metrics.get("eval_kptime_mover_keyword_score")),
            "kptime_mover_final_score": self._safe_float(metrics.get("eval_kptime_mover_final_score")),
            "combo_mover_score": self._safe_float(metrics.get("eval_combo_mover_score")),

            "learning_rate": self._safe_float(train_log.get("learning_rate")),
        }

        self._write_csv_row(row)

        self._write_jsonl({
            "step": row["step"],
            "epoch": row["epoch"],
            "row": row,
            "metrics_raw": metrics,
            "latest_train_log": train_log,
        })

        print(f"[EvalTableSaveCallback] saved eval row -> {self.csv_path}", flush=True)

        return control
