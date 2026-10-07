# 作业一：文本分类实现

三种文本表示方法（词袋 / Word Embedding / 预训练语言模型）+ Logistic Regression / BERT 微调，
在 NYT 新闻数据集上完成三分类，统一报告 **Accuracy** 与 **Macro-F1**。

---

## 1. 目录结构与代码功能对应表

| 文件 | 对应作业条目 | 功能 |
|---|---|---|
| `config.py` | — | 全局配置：路径、随机种子 `SEED=42`、划分比例、词向量维度、BERT 超参。**所有可调项集中在这一个文件** |
| `src/data.py` | 二、三（数据集与划分） | 读取 `nyt.csv` / `ag.csv`；生成并缓存**统一数据划分**（固定种子随机打乱 → 80/10/10）；nltk 分词与缓存 |
| `src/metrics.py` | 四（评价指标） | 计算 Accuracy、Macro-F1、单类明细与混淆矩阵；结果写入 `results/*.json` 与 `results/summary.json` |
| `src/linear.py` | 五、六（统一分类器） | Task 1 / Task 2 共用的 Logistic Regression 训练 + 评估流程，避免两处重复实现 |
| `src/task1_bow.py` | 五、Task 1（30 分） | **Binary Bag of Words** 与 **Word Frequency** 两种词袋表示 + LR |
| `src/task2_embedding.py` | 六、Task 2（50 分） | **GloVe 预训练** / **Word2Vec 训练于 AG News** / **Word2Vec 训练于 NYT**，词向量均值池化为文档向量 + LR |
| `src/task3_bert.py` | 七、Task 3（20 分） | **BERT-base-uncased 微调**（`max_length=64`、3 epochs），手写训练循环 + 线性 warmup |
| `run_all.py` | 全部 | 一键依次运行三个任务并打印汇总表；`--tasks` 可只跑指定任务 |
| `requirements.txt` | — | 依赖版本 |
| `results/` | — | 各实验指标 JSON、总表 `summary.json`、划分信息 `split_info.json` |

---

## 2. 环境准备

```bash
# 1) 创建虚拟环境（复用已有的 conda 环境作为基础，避免重复下载 torch）
python -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

# 2) 下载 nltk 分词模型（首次运行 src/data.py 时会自动下载到 assets/nltk_data）
# 3) 下载 GloVe 词向量
mkdir -p assets/glove && cd assets/glove
curl -O http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip glove.6B.100d.txt
# 4) BERT 权重首次运行 src/task3_bert.py 时自动从 HuggingFace 下载到 assets/hf
```

## 3. 运行方式

```bash
.venv/bin/python run_all.py              # 跑全部（本机 GPU 下全程约 6 分钟；无 GPU 时 Task 3 回退 CPU，约 3 小时）
.venv/bin/python run_all.py --tasks 1 2  # 只跑词袋与词向量部分（约 4 分钟）
.venv/bin/python -m src.task1_bow        # 单独运行某个任务
.venv/bin/python -m src.task2_embedding
.venv/bin/python -m src.task3_bert
```

BERT 训练设备由 `config.BERT_DEVICE = "auto"` 决定：检测到 CUDA 就用 GPU，否则自动回退 CPU，
并在日志首行打印实际使用的设备。若在容器/沙箱中运行，需确保 `/dev/dxg`（WSL）或 `/dev/nvidia*`
可被**读写**打开——只读打开会让 CUDA 报 `CUDA_ERROR_NO_DEVICE`。

结果写入 `results/summary.json`（总表）与 `results/<实验名>.json`（含单类 P/R/F1、混淆矩阵）。

---

## 4. 实验结果

统一划分：NYT 11,519 条 → train 9,215 / val 1,151 / test 1,153（`SEED=42`，见 `results/split_info.json`）。
类别分布不均（sports 75.0% / politics 12.6% / business 12.4%），故 **Macro-F1 比 Accuracy 更能反映真实能力**。

| 实验 | 对应条目 | Test Accuracy | Test Macro-F1 |
|---|---|---|---|
| `task1_binary_bow` | 五、Task 1（30 分） | **0.9896** | **0.9757** |
| `task1_word_frequency` | 五、Task 1（30 分） | 0.9896 | 0.9741 |
| `task2_glove_pretrained` | 六、Task 2（50 分） | 0.9844 | 0.9646 |
| `task2_word2vec_ag` | 六、Task 2（50 分） | 0.9740 | 0.9426 |
| `task2_word2vec_nyt` | 六、Task 2（50 分） | 0.9801 | 0.9559 |
| `task3_bert` | 七、Task 3（20 分） | 0.9827 | 0.9646 |

几点值得注意的结论（详见 `REPORT.md`）：

- **六组实验都落在很窄的区间内**（Accuracy 0.974–0.990，Macro-F1 0.943–0.976），说明该数据集三个类别在词汇层面区分度很高。
- **词袋击败了 BERT**（Macro-F1 0.9757 vs 0.9646）。直接原因是作业规定的 `max_length=64`：NYT 文档分词后平均 646 个 token，BERT 只能看到每篇约前 10% 的内容，而词袋与词向量方法使用全文。
- **预训练 GloVe 优于自行训练的 Word2Vec**：GloVe 在 60 亿 token 上训练，而 AG News 只有 348 万、NYT 只有 744 万 token。
- **Macro-F1 比 Accuracy 更关键**：sports 占 75% 且所有方法的 sports F1 都 ≥ 0.99，
  真正的区分度全在 business 与 politics 之间——Binary BoW 的 Accuracy 高达 0.9896，但 business 的召回率只有 0.927。


---

## 5. 关键实现说明

1. **统一数据划分**：`src/data.py: get_splits()` 用固定种子生成一次划分并缓存到 `assets/cache/splits_*.npz`，
   三个任务全部通过它取下标。因此「所有实验使用同一划分」由代码结构保证，而非靠人工对齐。
2. **词表只从训练集统计**：`src/task1_bow.py: build_bow()` 用 `fit_transform` 仅在训练集上拟合词表
   （142,222 词），验证/测试集只做 `transform`，避免信息泄漏。
3. **分词**：nltk `word_tokenize` + 转小写 + 丢弃纯标点 token（如 `''`、`--`）。
   分词结果缓存为 pickle，重复实验秒级加载。
4. **OOV 处理**：文档向量只对「在词表中」的词取平均（作业六.1 的公式），
   全 OOV 的文档退化为零向量。GloVe 只把语料用得到的词读入内存，而非加载全部 40 万词。
5. **求解器选择**：高维稀疏词袋用一对多 LR（`liblinear`，sklearn≥1.9 需用 `OneVsRestClassifier` 包装）；
   100 维稠密词向量用 `lbfgs` 多分类。
6. **计算环境**：本机无 GPU，BERT 微调在 CPU（8 线程）上进行；实测 `batch_size=32` 吞吐优于 16。
   99.8% 的文档长度超过 64 token，故 `padding="max_length"` 不产生额外浪费。
