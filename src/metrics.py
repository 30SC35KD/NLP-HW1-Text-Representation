"""指标计算与结果呈现。"""
from __future__ import annotations

import json

from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

import config

SUMMARY_PATH = config.RESULTS / "summary.json"


def evaluate(y_true, y_pred, label_names: list[str]) -> dict:
    """Accuracy、Macro-F1、单类明细与混淆矩阵。"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": classification_report(y_true, y_pred, target_names=label_names,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_result(name: str, payload: dict) -> None:
    with open(config.RESULTS / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def update_summary(name: str, section: str, test_metrics: dict, val_metrics: dict | None = None) -> None:
    """把本次实验的指标并进 results/summary.json。"""
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else {}
    entry = {"section": section,
             "test_accuracy": test_metrics["accuracy"],
             "test_macro_f1": test_metrics["macro_f1"]}
    if val_metrics is not None:
        entry["val_accuracy"] = val_metrics["accuracy"]
        entry["val_macro_f1"] = val_metrics["macro_f1"]
    summary[name] = entry
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def report_line(name: str, val: dict | None, test: dict, note: str = "") -> None:
    v = f"final-val  acc={val['accuracy']:.4f} macroF1={val['macro_f1']:.4f} | " if val else ""
    print(f"[{name:34s}] {v}test  acc={test['accuracy']:.4f} macroF1={test['macro_f1']:.4f} {note}")
