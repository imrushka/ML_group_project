"""
Task 5 — Yerlan Berikov

Evaluates the zero-shot baseline (no tweet labels at all)
Runs few-shot training for K = 10, 30, 50, 100, 200 samples per class
Plots a K-shot learning curve — Macro-F1 vs number of labelled examples
"""
from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report


SEED          = 42
NEUTRAL_LABEL = 1
ID2LABEL      = {0: "negative", 1: "neutral", 2: "positive"}

FINETUNED_DIR = "../models/roberta_finetuned"

MAX_LEN       = 128
BATCH_SIZE    = 16
EPOCHS        = 20
LR            = 5e-6

K_SHOTS = [10, 30, 50, 100, 200]

Path("../models").mkdir(exist_ok=True)
Path("../logs").mkdir(exist_ok=True)


class SentimentDataset(TorchDataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "macro_f1": round(f1_score(labels, preds, average="macro", zero_division=0), 4),
    }


def get_logits(model, dataset: TorchDataset, device: str, batch_size: int = 64) -> np.ndarray:
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            out = model(**batch)
            all_logits.append(out.logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def sample_k_per_class(df: pd.DataFrame, k: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    few_shot_idx = []
    for label in df["label"].unique():
        class_idx = df[df["label"] == label].index.tolist()
        chosen = rng.choice(class_idx, size=min(k, len(class_idx)), replace=False)
        few_shot_idx.extend(chosen.tolist())

    few_shot = df.loc[few_shot_idx].reset_index(drop=True)
    held_out = df.drop(index=few_shot_idx).reset_index(drop=True)
    return few_shot, held_out


def run_few_shot(k: int, tweet_df: pd.DataFrame, tokenizer, device: str) -> dict:
    rng = np.random.default_rng(SEED)
    few_shot_df, held_out_df = sample_k_per_class(tweet_df, k, rng)

    print(f"\n{'='*55}")
    print(f"  K = {k} shots per class  ({len(few_shot_df)} train / {len(held_out_df)} test)")
    print(f"{'='*55}")
    print(f"  Few-shot label dist : {few_shot_df['label'].map(ID2LABEL).value_counts().to_dict()}")

    X_train = few_shot_df["text_clean"].fillna("").tolist()
    y_train = few_shot_df["label"].astype(int).tolist()
    X_test  = held_out_df["text_clean"].fillna("").tolist()
    y_test  = held_out_df["label"].astype(int).tolist()

    train_ds = SentimentDataset(X_train, y_train, tokenizer)
    test_ds  = SentimentDataset(X_test,  y_test,  tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_DIR)
    model.to(device)

    output_dir = f"../models/roberta_fewshot_k{k}"
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir="../logs",
        logging_steps=10,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    t0 = time.time()
    trainer.train()
    print(f"  Training done in {(time.time()-t0)/60:.1f} min")

    model.to(device)
    logits = get_logits(model, test_ds, device)
    preds  = np.argmax(logits, axis=1)

    acc      = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    print(f"\n  Held-out  Accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}")
    print(classification_report(y_test, preds,
                                target_names=list(ID2LABEL.values()), zero_division=0))

    return {
        "k":          k,
        "train_size": len(X_train),
        "test_size":  len(X_test),
        "accuracy":   round(acc, 4),
        "macro_f1":   round(macro_f1, 4),
    }


def zero_shot_eval(tweet_df: pd.DataFrame, tokenizer, device: str) -> dict:
    X = tweet_df["text_clean"].fillna("").tolist()
    y = tweet_df["label"].astype(int).tolist()
    ds = SentimentDataset(X, y, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_DIR)
    model.to(device)

    logits = get_logits(model, ds, device)
    preds  = np.argmax(logits, axis=1)

    acc      = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro", zero_division=0)

    print(f"\nZero-shot baseline  Accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}")
    print(classification_report(y, preds,
                                target_names=list(ID2LABEL.values()), zero_division=0))
    return {"k": 0, "train_size": 0, "test_size": len(X),
            "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4)}


def plot_learning_curve(results: list[dict]):
    ks   = [r["k"] for r in results]
    f1s  = [r["macro_f1"] for r in results]
    accs = [r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(ks)), f1s,  marker="o", label="Macro-F1",  color="steelblue",  linewidth=2)
    ax.plot(range(len(ks)), accs, marker="s", label="Accuracy",  color="darkorange", linewidth=2, linestyle="--")

    for i, (k, f1) in enumerate(zip(ks, f1s)):
        ax.annotate(f"{f1:.3f}", (i, f1), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(["Zero-shot"] + [f"K={k}" for k in ks[1:]])
    ax.set_ylabel("Score")
    ax.set_title("Few-Shot Learning Curve on Tweets")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tweet_df = pd.read_csv("../data/processed/tweet_final_test.csv")
    print(f"Tweet test set: {len(tweet_df):,} rows | "
          f"{tweet_df['label'].map(ID2LABEL).value_counts().to_dict()}")

    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)

    print("\n" + "="*55)
    print("  Zero-shot baseline (no tweet labels)")
    print("="*55)
    results = [zero_shot_eval(tweet_df, tokenizer, device)]

    for k in K_SHOTS:
        min_class_size = tweet_df["label"].value_counts().min()
        if k > min_class_size:
            print(f"\nSkipping K={k}: not enough samples per class (min={min_class_size})")
            continue
        result = run_few_shot(k, tweet_df, tokenizer, device)
        results.append(result)
        with open("../logs/few_shot_metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved progress → ../logs/few_shot_metrics.json")

    summary_df = pd.DataFrame(results).set_index("k")
    print("\n\n" + "="*55)
    print("  K-Shot Learning Curve Summary")
    print("="*55)
    print(summary_df[["train_size", "accuracy", "macro_f1"]].to_string())

    plot_learning_curve(results)


if __name__ == "__main__":
    main()
