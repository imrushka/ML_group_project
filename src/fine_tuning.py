"""Task 4 — Aigerim Dairanbek
Fine-tuning (Week 12)

Fine-tunes a pretrained RoBERTa transformer on IMDb, then evaluates
zero-shot on tweets with the same threshold sweep used in baseline_models.py.

Run from the src/ folder:
    python finetune_model.py

Saves:
  ../models/roberta_finetuned/   — model + tokenizer checkpoint
  ../logs/finetune_metrics.json  — same format as training_and_testing_metrics.json

New dependencies (to requirements.txt):
    transformers>=4.40
    accelerate>=0.29
    torch>=2.0
"""
from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

SEED          = 42
NEUTRAL_LABEL = 1
THRESHOLDS    = [0.55, 0.65, 0.75]
ID2LABEL      = {0: 'negative', 1: 'neutral', 2: 'positive'}

# Fine tuning hyperparameters 

MAX_LEN    = 128   
BATCH_SIZE = 32
EPOCHS     = 3
LR         = 2e-5

# cardiffnlp/twitter-roberta-base-sentiment-latest is already pretrained on
# tweets AND sentiment — perfect for the domain-shift story.
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


Path('../models').mkdir(exist_ok=True)
Path('../logs').mkdir(exist_ok=True)

# Dataset wrapper

class SentimentDataset(TorchDataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# Metrics passed to trainer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': round(accuracy_score(labels, preds), 4),
        'macro_f1': round(f1_score(labels, preds, average='macro', zero_division=0), 4),
    }


# Threshold prediction

def predict_with_threshold(logits: np.ndarray, threshold: float) -> np.ndarray:
    """
    If max softmax probability < threshold → predict neutral (1).
    Mirrors predict_with_threshold() in baseline_models.py.
    """
    proba     = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    preds     = np.argmax(proba, axis=1)
    max_proba = proba.max(axis=1)
    preds[max_proba < threshold] = NEUTRAL_LABEL
    return preds


def get_logits(model, dataset: TorchDataset, device: str, batch_size: int = 64) -> np.ndarray:
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            out = model(**batch)
            all_logits.append(out.logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')

    # Loading csv files 
    train_df = pd.read_csv('../data/processed/imdb_train.csv')
    val_df   = pd.read_csv('../data/processed/imdb_val.csv')
    test_df  = pd.read_csv('../data/processed/imdb_test.csv')
    tweet_df = pd.read_csv('../data/processed/tweet_final_test.csv')

    print('Split sizes:')
    for name, df in [('train', train_df), ('val', val_df),
                     ('test', test_df),   ('tweet', tweet_df)]:
        print(f'  {name:5s}: {len(df):>6,} rows | '
              f'{df["label_str"].value_counts().to_dict()}')

    X_train = train_df['text_clean'].fillna('').tolist()
    X_val   = val_df['text_clean'].fillna('').tolist()
    X_test  = test_df['text_clean'].fillna('').tolist()
    X_tweet = tweet_df['text_clean'].fillna('').tolist()

    y_train = train_df['label'].astype(int).tolist()
    y_val   = val_df['label'].astype(int).tolist()
    y_test  = test_df['label'].astype(int).tolist()
    y_tweet = tweet_df['label'].astype(int).tolist()

    print(f'\nLoading tokenizer : {MODEL_NAME}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = SentimentDataset(X_train, y_train, tokenizer)
    val_ds   = SentimentDataset(X_val,   y_val,   tokenizer)
    test_ds  = SentimentDataset(X_test,  y_test,  tokenizer)
    tweet_ds = SentimentDataset(X_tweet, y_tweet, tokenizer)

    # Model:
    print(f'Loading model     : {MODEL_NAME}')
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},
        ignore_mismatched_sizes=True,   
    )
    # Training
    training_args = TrainingArguments(
        output_dir='../models/roberta_finetuned',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,
        logging_dir='../logs',
        logging_steps=200,
        seed=SEED,                              
        fp16=torch.cuda.is_available(),        
        report_to='none',                       
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print('\nStarting fine-tuning on IMDb ...')
    t0 = time.time()
    trainer.train()
    print(f'Training finished in {(time.time() - t0) / 60:.1f} min')

    model.save_pretrained('../models/roberta_finetuned')
    tokenizer.save_pretrained('../models/roberta_finetuned')
    print('Model saved → ../models/roberta_finetuned/')

    # IMDb evaluation
    present_labels     = sorted(set(y_train))
    label_names_binary = [ID2LABEL[i] for i in present_labels]  # neg + pos only

    imdb_results = {'RoBERTa': {}}
    model.to(device)

    for split_name, ds, y_true in [('val',  val_ds,  y_val),
                                    ('test', test_ds, y_test)]:
        logits   = get_logits(model, ds, device)
        preds    = np.argmax(logits, axis=1)
        acc      = accuracy_score(y_true, preds)
        macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)

        imdb_results['RoBERTa'][split_name] = {
            'accuracy': round(acc, 4),
            'macro_f1': round(macro_f1, 4),
        }
        print(f'\n── RoBERTa | IMDb {split_name} ──')
        print(classification_report(y_true, preds,
                                    target_names=label_names_binary,
                                    zero_division=0))

    # Zero-shot on Tweets
    tweet_logits = get_logits(model, tweet_ds, device)

    label_names_3class = ['negative', 'neutral', 'positive']
    tweet_results = {'RoBERTa': {}}

    print(f'\n\n\n{"=" * 60}')
    print('  RoBERTa — Zero-Shot on Tweets')
    print(f'{"=" * 60}')

    for threshold in THRESHOLDS:
        y_pred   = predict_with_threshold(tweet_logits, threshold)
        acc      = accuracy_score(y_tweet, y_pred)
        macro_f1 = f1_score(y_tweet, y_pred, average='macro', zero_division=0)

        tweet_results['RoBERTa'][threshold] = {
            'accuracy': round(acc, 4),
            'macro_f1': round(macro_f1, 4),
        }
        print(f'\n\n threshold={threshold} | Accuracy={acc:.4f} | Macro-F1={macro_f1:.4f}')
        print(classification_report(y_tweet, y_pred,
                                    target_names=label_names_3class,
                                    zero_division=0))

    best = {'RoBERTa': max(THRESHOLDS,
                           key=lambda t: tweet_results['RoBERTa'][t]['macro_f1'])}
    print(f'RoBERTa: best threshold = {best["RoBERTa"]} | '
          f'Macro-F1 = {tweet_results["RoBERTa"][best["RoBERTa"]]["macro_f1"]}')
    print('\n\n')

    # Domain-gap summary
    imdb_f1  = imdb_results['RoBERTa']['test']['macro_f1']
    tweet_f1 = tweet_results['RoBERTa'][best['RoBERTa']]['macro_f1']
    drop     = round(imdb_f1 - tweet_f1, 4)

    summary_rows = [{
        'model':                 'RoBERTa',
        'IMDb F1 (binary)':     imdb_f1,
        'Tweet F1 (zero-shot)':  tweet_f1,
        'Drop':                  drop,
        'Best threshold':        best['RoBERTa'],
    }]
    summary_df = pd.DataFrame(summary_rows).set_index('model')
    print(summary_df.to_string())
    print('\n\n')

    # Metrics 
    all_metrics = {
        'imdb':            imdb_results,
        'tweet_zeroshot':  tweet_results,
        'best_thresholds': best,
        'domain_gap':      {'RoBERTa': drop},
    }
    with open('../logs/finetune_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print('Saved → ../logs/finetune_metrics.json')


if __name__ == '__main__':
    main()