"""Task 2：GloVe 与 Word2Vec 的文档表示。"""
from __future__ import annotations

import time

import numpy as np
from gensim.models import Word2Vec

import config
from src import data, linear


def load_glove(vocab: set[str]) -> dict[str, np.ndarray]:
    """只读语料用得到的词，不加载全部 40 万词。"""
    if not config.GLOVE_PATH.exists():
        raise FileNotFoundError(f"缺少 GloVe 文件：{config.GLOVE_PATH}，请见 README 的数据准备步骤")
    vectors: dict[str, np.ndarray] = {}
    with open(config.GLOVE_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if parts[0] in vocab:
                vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return vectors


def train_word2vec(token_lists: list[list[str]], tag: str):
    """训练 Word2Vec，返回词向量表。"""
    t0 = time.time()
    model = Word2Vec(sentences=token_lists, **config.W2V_PARAMS)
    kv = model.wv
    print(f"  trained Word2Vec[{tag}]: vocab={len(kv)} dim={kv.vector_size} "
          f"corpus_docs={len(token_lists)} ({time.time() - t0:.1f}s)")
    return kv


def encode_documents(token_lists: list[list[str]], vectors, dim: int) -> np.ndarray:
    """文档向量取文档内所有有效词词向量的平均，OOV 跳过。"""
    X = np.zeros((len(token_lists), dim), dtype=np.float32)
    for i, toks in enumerate(token_lists):
        vecs = [vectors[t] for t in toks if t in vectors]
        if vecs:
            X[i] = np.mean(vecs, axis=0)
    return X


def run_experiment(name: str, vectors, tokens, y, splits, label_names, note: str) -> None:
    """池化成文档向量后交给 LR 训练和评估。"""
    train_idx, val_idx, test_idx = splits
    t0 = time.time()
    X = encode_documents(tokens, vectors, config.EMBED_DIM)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    print(f"  encoded {X.shape} in {time.time() - t0:.1f}s")
    linear.run_lr(name, "Task2-Embedding", X_train, y[train_idx], X_val, y[val_idx],
                  X_test, y[test_idx], label_names, sparse=False, note=note)


def main() -> None:
    texts, y, label_names = data.load_nyt()
    tokens = data.nyt_tokens()
    splits = data.get_splits(len(texts))
    train_idx = splits[0]

    print("\n=== task2_glove_pretrained ===")
    nyt_vocab = {t for toks in tokens for t in toks}
    run_experiment("task2_glove_pretrained", load_glove(nyt_vocab), tokens, y, splits,
                   label_names, note="glove.6B.100d (pre-trained)")

    print("\n=== task2_word2vec_ag ===")
    ag = data.ag_tokens()
    run_experiment("task2_word2vec_ag", train_word2vec(ag, "AG"), tokens, y, splits,
                   label_names, note="Word2Vec-100d trained on AG News")

    print("\n=== task2_word2vec_nyt ===")
    if config.W2V_NYT_CORPUS == "train":
        corpus = [tokens[i] for i in train_idx]
        note = "Word2Vec-100d trained on NYT train split"
    else:
        corpus = tokens
        note = "Word2Vec-100d trained on all NYT text"
    run_experiment("task2_word2vec_nyt", train_word2vec(corpus, "NYT"), tokens, y, splits,
                   label_names, note=note)


if __name__ == "__main__":
    main()
