# 代码运行说明

## 1. 环境准备

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

## 2. 数据准备

```bash
mkdir -p assets/glove
curl -L -o assets/glove/glove.6B.zip http://nlp.stanford.edu/data/glove.6B.zip
unzip -o assets/glove/glove.6B.zip glove.6B.100d.txt -d assets/glove
```

## 3. 运行全部实验

```bash
.venv/bin/python run_all.py
```

## 4. Task 1：Bag of Words

Binary Bag of Words、Word Frequency：

```bash
.venv/bin/python -m src.task1_bow
```

## 5. Task 2：Word2Vec / GloVe

Pre-trained GloVe、Word2Vec trained on AG News、Word2Vec trained on NYT：

```bash
.venv/bin/python -m src.task2_embedding
```

## 6. Task 3：Pre-trained Neural Models

BERT-base-uncased 微调：

```bash
.venv/bin/python -m src.task3_bert
```
