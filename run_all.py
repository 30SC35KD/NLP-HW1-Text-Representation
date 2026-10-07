"""依次运行三个任务并汇总结果。

    python run_all.py              # 全部
    python run_all.py --tasks 1 2  # 只跑指定任务
"""
from __future__ import annotations

import argparse
import json
import time

import config
from src import data, metrics, task1_bow, task2_embedding, task3_bert

TASKS = {1: ("Task 1  Bag of Words", task1_bow.main),
         2: ("Task 2  Word2Vec / GloVe", task2_embedding.main),
         3: ("Task 3  BERT fine-tuning", task3_bert.main)}


def reset_summary(full_run: bool) -> None:
    # 跑全部时清空，只跑部分任务就保留其他任务已有的结果
    config.RESULTS.mkdir(exist_ok=True)
    if full_run or not metrics.SUMMARY_PATH.exists():
        metrics.SUMMARY_PATH.write_text("{}", encoding="utf-8")


def print_summary() -> None:
    summary = json.loads(metrics.SUMMARY_PATH.read_text(encoding="utf-8"))
    print("\n" + "=" * 96)
    print(f"{'实验':<28}{'任务':<18}{'Test Acc':>10}{'Test Macro-F1':>15}")
    print("-" * 96)
    for name, e in summary.items():
        print(f"{name:<28}{e['section']:<18}{e['test_accuracy']:>10.4f}{e['test_macro_f1']:>15.4f}")
    print("=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", type=int, default=[1, 2, 3],
                        choices=[1, 2, 3], help="要运行的任务编号")
    args = parser.parse_args()

    reset_summary(full_run=len(args.tasks) == len(TASKS))
    data.write_split_report()
    for t in args.tasks:
        title, fn = TASKS[t]
        print(f"\n{'#' * 96}\n### {title}\n{'#' * 96}")
        t0 = time.time()
        fn()
        print(f"### {title} 用时 {time.time() - t0:.1f}s")

    print_summary()
    print("\n指标明细见 results/*.json，总表见 results/summary.json")


if __name__ == "__main__":
    main()
