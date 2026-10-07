"""Task 3：微调 BERT-base-uncased 做 NYT 三分类。"""
from __future__ import annotations

import time

import config  # 要先于 transformers 导入，这里设置 HF_HOME
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src import data, metrics


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if config.BERT_DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.BERT_DEVICE)


def tokenize_splits(tokenizer, texts, splits, max_len: int):
    """把三个划分都编码好，训练时不用再分词。"""
    out = {}
    for name, idx in zip(("train", "val", "test"), splits):
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_len,
                        padding="max_length", return_tensors="pt")
        out[name] = (enc["input_ids"], enc["attention_mask"])
    return out


def make_loader(ids, mask, labels, batch_size: int, shuffle: bool, pin_memory: bool) -> DataLoader:
    ds = TensorDataset(ids, mask, torch.as_tensor(labels, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, float]:
    """返回预测标签和平均损失。"""
    model.eval()
    preds, losses = [], []
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        losses.append(torch.nn.functional.cross_entropy(logits, labels).item())
        preds.append(logits.argmax(dim=-1))
    return torch.cat(preds).cpu().numpy(), float(np.mean(losses))


def main() -> None:
    set_seed(config.SEED)
    torch.set_num_threads(config.BERT_THREADS)
    device = resolve_device()
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    print(f"  device = {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if use_cuda else " (未检测到 GPU，回退 CPU)"))

    texts, y, label_names = data.load_nyt()
    splits = data.get_splits(len(texts))

    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BERT_MODEL, num_labels=len(label_names)).to(device)
    enc = tokenize_splits(tokenizer, texts, splits, config.BERT_MAX_LEN)
    loaders = {k: make_loader(enc[k][0], enc[k][1], y[idx], config.BERT_BATCH_SIZE,
                              k == "train", pin_memory=use_cuda)
               for k, idx in zip(("train", "val", "test"), splits)}

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.BERT_LR,
                                  weight_decay=config.BERT_WEIGHT_DECAY)
    total_steps = len(loaders["train"]) * config.BERT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config.BERT_WARMUP_RATIO), total_steps)

    history = []
    for epoch in range(1, config.BERT_EPOCHS + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, (ids, mask, labels) in enumerate(loaders["train"], 1):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(loaders['train'])} "
                      f"loss={running / step:.4f} ({time.time() - t0:.0f}s)", flush=True)

        val_pred, val_loss = predict(model, loaders["val"], device)
        val_m = metrics.evaluate(y[splits[1]], val_pred, label_names)
        history.append({"epoch": epoch, "train_loss": running / len(loaders["train"]),
                        "val_loss": val_loss, **{k: val_m[k] for k in ("accuracy", "macro_f1")}})
        print(f"  epoch {epoch} done: train_loss={history[-1]['train_loss']:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_m['accuracy']:.4f} "
              f"val_macroF1={val_m['macro_f1']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    test_pred, _ = predict(model, loaders["test"], device)
    test_m = metrics.evaluate(y[splits[2]], test_pred, label_names)
    # 循环结束后 val_pred 就是最终模型在验证集上的预测
    val_m = metrics.evaluate(y[splits[1]], val_pred, label_names)

    result = {"name": "task3_bert", "section": "Task3-BERT",
              "model": config.BERT_MODEL, "max_length": config.BERT_MAX_LEN,
              "epochs": config.BERT_EPOCHS, "batch_size": config.BERT_BATCH_SIZE,
              "learning_rate": config.BERT_LR, "device": str(device), "history": history,
              "val": val_m, "test": test_m}
    metrics.save_result("task3_bert", result)
    metrics.update_summary("task3_bert", "Task3-BERT", test_m, val_m)
    metrics.report_line("task3_bert", val_m, test_m,
                        f"{config.BERT_MODEL} max_len={config.BERT_MAX_LEN} epochs={config.BERT_EPOCHS}")

    if config.BERT_SAVE_MODEL:
        save_dir = config.ASSETS / "bert_finetuned"
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"  fine-tuned model saved to {save_dir}")


if __name__ == "__main__":
    main()
