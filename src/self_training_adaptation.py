"""
Task 5 — Muhammadjon Merzakulov

Evaluates the zero-shot baseline on the target domain (Tweets).
Runs pseudo-labeling on the unlabelled pool using confidence thresholds.
Performs self-training (fine-tuning) on the accepted pseudo-labels.
Compares the final adapted model against the baseline.
"""
from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report


SEED          = 42
ID2LABEL      = {0: "negative", 1: "neutral", 2: "positive"}

FINETUNED_DIR    = "../models/roberta_finetuned"
SELF_TRAINED_DIR = "../models/roberta_self_trained"

MAX_LEN       = 128
BATCH_SIZE    = 16
EPOCHS        = 3
LR            = 2e-5

# Confidence thresholds for pseudo-labeling
THRESHOLD_POS     = 0.90
THRESHOLD_NEG     = 0.90
THRESHOLD_NEUTRAL = 0.75

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


class UnlabelledDataset(TorchDataset):
    def __init__(self, texts: list[str], tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "macro_f1": round(f1_score(labels, preds, average="macro", zero_division=0), 4),
    }


def get_logits(model, dataset: TorchDataset, device: str, batch_size: int = 64) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            out = model(**batch)
            all_logits.append(out.logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def run_pseudo_labeling(unlabelled_texts: list[str], model, tokenizer, device: str) -> tuple[list[str], list[int]]:
    print(f"\n{'='*55}")
    print(f"  Starting Pseudo-Labeling on {len(unlabelled_texts):,} unlabelled tweets")
    print(f"{'='*55}")
    
    ds = UnlabelledDataset(unlabelled_texts, tokenizer)
    logits = get_logits(model, ds, device, batch_size=64)
    
    # Convert logits to probabilities
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    
    pseudo_texts = []
    pseudo_labels = []
    
    for i, prob in enumerate(probs):
        p_neg = prob[0]
        p_pos = prob[2]
        
        label = -1
        # Applying the confidence threshold logic
        if p_pos > THRESHOLD_POS:
            label = 2
        elif p_neg > THRESHOLD_NEG:
            label = 0
        elif p_pos < THRESHOLD_NEUTRAL and p_neg < THRESHOLD_NEUTRAL:
            label = 1
            
        if label != -1:
            pseudo_texts.append(unlabelled_texts[i])
            pseudo_labels.append(label)
            
    print(f"  Pseudo-labeling complete. Accepted: {len(pseudo_texts):,} / {len(unlabelled_texts):,}")
    return pseudo_texts, pseudo_labels


def zero_shot_eval(test_ds: TorchDataset, y_test: list[int], model, device: str) -> dict:
    print(f"\n{'='*55}")
    print("  Evaluating Zero-Shot Baseline (Before Self-Training)")
    print(f"{'='*55}")
    
    logits = get_logits(model, test_ds, device)
    preds  = np.argmax(logits, axis=1)

    acc      = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    print(f"\n  Zero-Shot Baseline  Accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}")
    print(classification_report(y_test, preds, target_names=list(ID2LABEL.values()), zero_division=0))
    
    return {"Model": "Zero-Shot Baseline", "Accuracy": round(acc, 4), "Macro-F1": round(macro_f1, 4)}


def run_self_training(pseudo_texts: list[str], pseudo_labels: list[int], test_ds: TorchDataset, y_test: list[int], tokenizer, device: str) -> dict:
    print(f"\n{'='*55}")
    print(f"  Starting Self-Training (Fine-Tuning on Pseudo-labels)")
    print(f"  Training Set Size: {len(pseudo_texts):,}")
    print(f"{'='*55}")

    train_ds = SentimentDataset(pseudo_texts, pseudo_labels, tokenizer)

    # Re-load the baseline model to fine-tune it
    model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_DIR)
    model.to(device)

    args = TrainingArguments(
        output_dir=SELF_TRAINED_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="no",
        logging_dir="../logs",
        logging_steps=50,
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

    # Save the adapted model
    trainer.save_model(SELF_TRAINED_DIR)
    tokenizer.save_pretrained(SELF_TRAINED_DIR)
    print(f"  Model saved → {SELF_TRAINED_DIR}")

    print(f"\n{'='*55}")
    print("  Evaluating Adapted Model (After Self-Training)")
    print(f"{'='*55}")

    model.to(device)
    logits = get_logits(model, test_ds, device)
    preds  = np.argmax(logits, axis=1)

    acc      = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    print(f"\n  Adapted Model       Accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}")
    print(classification_report(y_test, preds, target_names=list(ID2LABEL.values()), zero_division=0))

    return {"Model": "Self-Trained Model", "Accuracy": round(acc, 4), "Macro-F1": round(macro_f1, 4)}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load datasets
    tweet_df = pd.read_csv("../data/processed/tweet_final_test.csv")
    unlabelled_df = pd.read_csv("../data/processed/tweet_unlabelled_pool.csv")
    
    print(f"Tweet test set: {len(tweet_df):,} rows | "
          f"{tweet_df['label'].map(ID2LABEL).value_counts().to_dict()}")

    # Prepare texts and labels
    test_texts = tweet_df["text_clean"].fillna("").tolist()
    y_test = tweet_df["label"].astype(int).tolist()
    unlabelled_texts = unlabelled_df["text_clean"].fillna("").tolist()

    # Load baseline model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_DIR)
    model.to(device)

    test_ds = SentimentDataset(test_texts, y_test, tokenizer)

    results = []

    # 1. Zero-shot Evaluation
    zero_metrics = zero_shot_eval(test_ds, y_test, model, device)
    results.append(zero_metrics)

    # 2. Pseudo-Labeling
    pseudo_texts, pseudo_labels = run_pseudo_labeling(unlabelled_texts, model, tokenizer, device)
    
    if len(pseudo_texts) == 0:
        print("\n[!] No samples accepted. Try lowering the thresholds.")
        return

    # 3. Self-Training
    adapted_metrics = run_self_training(pseudo_texts, pseudo_labels, test_ds, y_test, tokenizer, device)
    results.append(adapted_metrics)

    # 4. Save Metrics
    final_output = {
        "accepted_samples": len(pseudo_texts),
        "total_unlabelled": len(unlabelled_texts),
        "metrics": results
    }
    
    with open("../logs/self_training_metrics.json", "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n  Saved progress → ../logs/self_training_metrics.json")

    # Display final summary
    summary_df = pd.DataFrame(results).set_index("Model")
    print("\n\n" + "="*55)
    print("  Domain Adaptation Summary")
    print("="*55)
    print(summary_df.to_string())


if __name__ == "__main__":
    main()
