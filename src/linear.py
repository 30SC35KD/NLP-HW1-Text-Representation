"""Task 1 / Task 2 共用的 Logistic Regression 训练与评估。"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

import config
from src import metrics


def make_lr(sparse: bool):
    # 稀疏词袋用 OvR + liblinear（sklearn>=1.9 的 liblinear 不再支持多分类，包一层）
    # 稠密词向量直接用多分类 lbfgs
    if sparse:
        return OneVsRestClassifier(LogisticRegression(**config.BOW_LR, random_state=config.SEED))
    return LogisticRegression(**config.EMBED_LR, random_state=config.SEED)


def run_lr(name: str, section: str, X_train, y_train, X_val, y_val, X_test, y_test,
           label_names: list[str], sparse: bool = True, note: str = "") -> dict:
    """训练并在 val/test 上评估，结果落盘。"""
    clf = make_lr(sparse).fit(X_train, y_train)
    val = metrics.evaluate(y_val, clf.predict(X_val), label_names)
    test = metrics.evaluate(y_test, clf.predict(X_test), label_names)
    result = {"name": name, "section": section, "model": "LogisticRegression",
              "feature_shape": list(X_train.shape), "note": note,
              "val": val, "test": test}
    metrics.save_result(name, result)
    metrics.update_summary(name, section, test, val)
    metrics.report_line(name, val, test, note)
    return result
