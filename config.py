"""全局配置。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NYT_CSV = ROOT / "nyt.csv"
AG_CSV = ROOT / "ag.csv"

ASSETS = ROOT / "assets"
CACHE = ASSETS / "cache"
RESULTS = ROOT / "results"
GLOVE_PATH = ASSETS / "glove" / "glove.6B.100d.txt"
HF_CACHE = ASSETS / "hf"
NLTK_DATA = ASSETS / "nltk_data"

for _d in (ASSETS, CACHE, RESULTS, HF_CACHE, NLTK_DATA, CACHE / "joblib"):
    _d.mkdir(parents=True, exist_ok=True)

# 要在 import transformers 之前设置
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(CACHE / "joblib"))

SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)

TOKENIZE_LOWER = True
MIN_TOKEN_CHARS = 1

BOW_LR = dict(solver="liblinear", C=1.0, max_iter=1000)

EMBED_DIM = 100
W2V_PARAMS = dict(vector_size=EMBED_DIM, window=5, min_count=5, sg=1,
                  workers=8, epochs=5, seed=SEED)
W2V_NYT_CORPUS = "train"   # "train" 只用训练集，"all" 用全部 NYT 文本
EMBED_LR = dict(solver="lbfgs", C=1.0, max_iter=2000)

BERT_MODEL = "google-bert/bert-base-uncased"
BERT_MAX_LEN = 64
BERT_EPOCHS = 3
BERT_BATCH_SIZE = 32
BERT_LR = 2e-5
BERT_WEIGHT_DECAY = 0.01
BERT_WARMUP_RATIO = 0.1
BERT_DEVICE = "auto"       # 有 GPU 就用 GPU，否则回退 CPU
BERT_THREADS = 8           # 只在 CPU 训练时生效
BERT_SAVE_MODEL = True
