"""数据读取、数据划分与分词缓存。"""
from __future__ import annotations

import csv
import json
import pickle
import re
from pathlib import Path

import numpy as np

import config


def load_nyt() -> tuple[list[str], np.ndarray, list[str]]:
    """返回 (文本, 标签, 类别名)，类别名按字典序编码。"""
    texts, labels = [], []
    with open(config.NYT_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(row["label"])
    label_names = sorted(set(labels))
    label2id = {name: i for i, name in enumerate(label_names)}
    y = np.array([label2id[t] for t in labels], dtype=np.int64)
    return texts, y, label_names


def load_ag_texts() -> list[str]:
    """AG News 只有 text 一列，用于训练 Word2Vec。"""
    with open(config.AG_CSV, encoding="utf-8", newline="") as f:
        return [row["text"] for row in csv.DictReader(f)]


def get_splits(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 打乱后按比例切分，结果存盘，三个任务共用同一份
    ratios = config.SPLIT_RATIOS
    tag = "-".join(f"{r:g}" for r in ratios)
    cache = config.CACHE / f"splits_n{n}_seed{config.SEED}_{tag}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["train"], d["val"], d["test"]

    rng = np.random.default_rng(config.SEED)
    perm = rng.permutation(n)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = np.sort(perm[:n_train])
    val = np.sort(perm[n_train:n_train + n_val])
    test = np.sort(perm[n_train + n_val:])
    np.savez(cache, train=train, val=val, test=test)
    return train, val, test


_PUNCT_ONLY = re.compile(r"^[^0-9a-zA-Z]+$")


def _tokenize_with_nltk(text: str) -> list[str]:
    from nltk import word_tokenize
    toks = word_tokenize(text, preserve_line=True)
    out = []
    for t in toks:
        if config.TOKENIZE_LOWER:
            t = t.lower()
        if len(t) < config.MIN_TOKEN_CHARS or _PUNCT_ONLY.match(t):
            continue          # 丢掉 "''" "--" "$" 这类纯标点
        out.append(t)
    return out


def tokenize(texts: list[str], cache_name: str, source: Path) -> list[list[str]]:
    # 缓存名带上语料和分词配置的指纹，任一改动都会重新分词
    config.NLTK_DATA.mkdir(parents=True, exist_ok=True)
    import nltk
    if str(config.NLTK_DATA) not in nltk.data.path:
        nltk.data.path.insert(0, str(config.NLTK_DATA))

    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, download_dir=str(config.NLTK_DATA), quiet=True)

    fingerprint = (f"n{len(texts)}_bytes{source.stat().st_size}"
                   f"_low{int(config.TOKENIZE_LOWER)}_min{config.MIN_TOKEN_CHARS}")
    cache = config.CACHE / f"tokens_{cache_name}_{fingerprint}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    tokens = [_tokenize_with_nltk(t) for t in texts]
    with open(cache, "wb") as f:
        pickle.dump(tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
    return tokens


def nyt_tokens() -> list[list[str]]:
    return tokenize(load_nyt()[0], "nyt", config.NYT_CSV)


def ag_tokens() -> list[list[str]]:
    return tokenize(load_ag_texts(), "ag", config.AG_CSV)


def split_indices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return get_splits(len(load_nyt()[0]))


def write_split_report() -> None:
    """把实际划分写到 results/，方便核对。"""
    texts, y, label_names = load_nyt()
    train, val, test = get_splits(len(texts))
    info = {
        "seed": config.SEED,
        "ratios": list(config.SPLIT_RATIOS),
        "n_total": len(texts),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "class_names": label_names,
        "test_label_counts": {label_names[c]: int((y[test] == c).sum()) for c in range(len(label_names))},
        "first_test_indices": test[:10].tolist(),
    }
    with open(config.RESULTS / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
