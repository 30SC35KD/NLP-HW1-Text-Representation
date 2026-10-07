"""Task 1：Bag of Words，Binary 与 Word Frequency 两种表示。"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from src import data, linear


def build_bow(train_tokens, splits, binary: bool):
    # analyzer 直接吃已分好的 token，不用 sklearn 自带分词
    train_idx, val_idx, test_idx = splits
    vectorizer = CountVectorizer(analyzer=lambda toks: toks, binary=binary, dtype=np.float32)
    X_train = vectorizer.fit_transform([train_tokens[i] for i in train_idx])
    X_val = vectorizer.transform([train_tokens[i] for i in val_idx])
    X_test = vectorizer.transform([train_tokens[i] for i in test_idx])
    return vectorizer, X_train, X_val, X_test


def main() -> None:
    texts, y, label_names = data.load_nyt()
    tokens = data.nyt_tokens()
    splits = data.get_splits(len(texts))
    train_idx, val_idx, test_idx = splits
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    for binary, name, desc in [
        (True, "task1_binary_bow", "Binary Bag of Words (0/1 出现指示)"),
        (False, "task1_word_frequency", "Word Frequency (原始词频计数)"),
    ]:
        vectorizer, X_train, X_val, X_test = build_bow(tokens, splits, binary)
        print(f"\n=== {name} | vocab size = {len(vectorizer.vocabulary_)} ===")
        linear.run_lr(name, "Task1-BoW", X_train, y_train, X_val, y_val, X_test, y_test,
                      label_names, sparse=True, note=desc)


if __name__ == "__main__":
    main()
