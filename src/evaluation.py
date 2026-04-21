"""
Task 7 — Dariga Borasheva
Evaluation & Analysis

Loads all metric logs produced by the other pipeline stages and produces:
  1. Per-model performance table (IMDb source domain)
  2. Zero-shot cross-domain performance table (Tweet target domain)
  3. Domain-gap summary (F1 drop from source → target)
  4. Few-shot learning curve (K-shot Macro-F1 vs number of labels)
  5. Self-training adaptation comparison (before vs after)
  6. Confusion matrix heatmaps for each model × split
  7. Per-class F1 bar chart (negative / neutral / positive) per model
  8. Combined cross-model comparison figure

Reads from:
  ../logs/training_and_testing_metrics.json   (baseline LR + SVM)
  ../logs/finetune_metrics.json               (RoBERTa fine-tuned)
  ../logs/few_shot_metrics.json               (K-shot learning curve)
  ../logs/self_training_metrics.json          (pseudo-label self-training)

Outputs:
  ../logs/evaluation_summary.json
  ../figures/  (all PNG plots)

Run from src/:
  python evaluation.py
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────────────────

LOGS    = Path('../logs')
FIGURES = Path('../figures')
FIGURES.mkdir(exist_ok=True, parents=True)

LABEL_NAMES = ['negative', 'neutral', 'positive']

# ── Colour palette (consistent across all figures) ────────────────────────────

PALETTE = {
    'LogisticRegression': '#4C72B0',
    'LinearSVC':          '#DD8452',
    'RoBERTa':            '#55A868',
    'Self-Trained':       '#C44E52',
    'negative':           '#E05C5C',
    'neutral':            '#F2A65A',
    'positive':           '#5DA85D',
}
STYLE = dict(dpi=150, bbox_inches='tight')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES / f'{name}.png'
    fig.savefig(path, **STYLE)
    plt.close(fig)
    print(f'  ✓  Saved → {path}')


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f'Expected log file not found: {path}\n'
            'Make sure all upstream pipeline stages have been run.'
        )
    with open(path) as f:
        return json.load(f)


# ── 1. Source-domain (IMDb) performance table ─────────────────────────────────

def table_imdb(baseline: dict, finetune: dict) -> pd.DataFrame:
    rows = []
    for model, splits in baseline['imdb'].items():
        for split, m in splits.items():
            rows.append({'Model': model, 'Split': split, **m})
    for model, splits in finetune['imdb'].items():
        for split, m in splits.items():
            rows.append({'Model': model, 'Split': split, **m})

    df = (pd.DataFrame(rows)
            .rename(columns={'accuracy': 'Accuracy', 'macro_f1': 'Macro-F1'})
            .set_index(['Model', 'Split'])
            .sort_index())
    print('\n── IMDb (Source Domain) Performance ──')
    print(df.to_string())
    return df


# ── 2. Zero-shot tweet performance table ──────────────────────────────────────

def table_zeroshot(baseline: dict, finetune: dict) -> pd.DataFrame:
    rows = []

    best_b = baseline.get('best_thresholds', {})
    for model, thresholds in baseline['tweet_zeroshot'].items():
        bt = best_b.get(model)
        if bt is not None:
            m = thresholds[bt if isinstance(thresholds, dict) else str(bt)]
            rows.append({'Model': model, 'Best Threshold': bt, **m})

    best_f = finetune.get('best_thresholds', {})
    for model, thresholds in finetune['tweet_zeroshot'].items():
        bt = best_f.get(model)
        if bt is not None:
            key = bt if bt in thresholds else str(bt)
            m = thresholds[key]
            rows.append({'Model': model, 'Best Threshold': bt, **m})

    df = (pd.DataFrame(rows)
            .rename(columns={'accuracy': 'Accuracy', 'macro_f1': 'Macro-F1'})
            .set_index('Model'))
    print('\n── Zero-Shot Tweet Performance (best threshold per model) ──')
    print(df.to_string())
    return df


# ── 3. Domain gap ─────────────────────────────────────────────────────────────

def table_domain_gap(baseline: dict, finetune: dict) -> pd.DataFrame:
    gap = {}
    gap.update(baseline.get('domain_gap', {}))
    gap.update(finetune.get('domain_gap', {}))

    rows = [{'Model': m, 'F1 Drop': round(v, 4)} for m, v in gap.items()]
    df = pd.DataFrame(rows).set_index('Model')
    print('\n── Domain Gap (IMDb F1 − Tweet Zero-Shot F1) ──')
    print(df.to_string())
    return df


# ── 4. Figures ─────────────────────────────────────────────────────────────────

def fig_imdb_comparison(df: pd.DataFrame) -> None:
    """Grouped bar chart: IMDb test Macro-F1 per model."""
    test_df = df.xs('test', level='Split')
    models  = test_df.index.tolist()
    f1s     = test_df['Macro-F1'].tolist()
    accs    = test_df['Accuracy'].tolist()

    x   = np.arange(len(models))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))

    bars_f1  = ax.bar(x - w/2, f1s,  w, label='Macro-F1',  color=[PALETTE.get(m, '#888') for m in models])
    bars_acc = ax.bar(x + w/2, accs, w, label='Accuracy',   color=[PALETTE.get(m, '#888') for m in models], alpha=0.55)

    for bar in bars_f1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('IMDb Test Performance by Model')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, '1_imdb_test_performance')


def fig_zeroshot_comparison(zs_df: pd.DataFrame) -> None:
    """Bar chart: Zero-shot tweet Macro-F1 per model."""
    models = zs_df.index.tolist()
    f1s    = zs_df['Macro-F1'].tolist()

    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = [PALETTE.get(m, '#888') for m in models]
    bars    = ax.bar(models, f1s, color=colors, width=0.5)

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Macro-F1')
    ax.set_title('Zero-Shot Performance on Tweets (best threshold)')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, '2_zeroshot_tweet_performance')


def fig_domain_gap(baseline: dict, finetune: dict) -> None:
    """Grouped bar: source F1 vs zero-shot tweet F1 per model, with drop annotation."""
    all_zs    = {**baseline['tweet_zeroshot'], **finetune['tweet_zeroshot']}
    all_imdb  = {**baseline['imdb'], **finetune['imdb']}
    best_b    = baseline.get('best_thresholds', {})
    best_f    = finetune.get('best_thresholds', {})
    best      = {**best_b, **best_f}

    models, src_f1s, tgt_f1s = [], [], []
    for model in all_imdb:
        bt  = best.get(model)
        if bt is None:
            continue
        zs  = all_zs.get(model, {})
        key = bt if bt in zs else str(bt)
        if key not in zs:
            continue
        models.append(model)
        src_f1s.append(all_imdb[model]['test']['macro_f1'])
        tgt_f1s.append(zs[key]['macro_f1'])

    x   = np.arange(len(models))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(x - w/2, src_f1s, w, label='IMDb (source)', alpha=0.85,
           color=[PALETTE.get(m, '#888') for m in models])
    ax.bar(x + w/2, tgt_f1s, w, label='Tweet (zero-shot)', alpha=0.55,
           color=[PALETTE.get(m, '#888') for m in models])

    for i, (s, t) in enumerate(zip(src_f1s, tgt_f1s)):
        drop = s - t
        ax.annotate(f'↓{drop:.3f}', xy=(i, min(s, t) - 0.04),
                    ha='center', fontsize=9, color='crimson', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel('Macro-F1')
    ax.set_title('Domain Gap: Source vs Target Macro-F1')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, '3_domain_gap')


def fig_few_shot_curve(few_shot: list[dict]) -> None:
    """Learning curve: Macro-F1 and Accuracy vs K (number of labelled examples)."""
    df   = pd.DataFrame(few_shot).sort_values('k')
    ks   = df['k'].tolist()
    f1s  = df['macro_f1'].tolist()
    accs = df['accuracy'].tolist()

    # x-axis labels: 0 → "Zero-shot"
    xlabels = ['Zero-shot' if k == 0 else f'K={k}' for k in ks]
    x       = np.arange(len(ks))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, f1s,  marker='o', label='Macro-F1',  color='steelblue',  linewidth=2)
    ax.plot(x, accs, marker='s', label='Accuracy',   color='darkorange', linewidth=2, linestyle='--')

    for i, (k, f1) in enumerate(zip(ks, f1s)):
        ax.annotate(f'{f1:.3f}', (i, f1), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel('Score')
    ax.set_title('Few-Shot Learning Curve on Tweets (RoBERTa)')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, '4_few_shot_learning_curve')


def fig_self_training(self_train: dict) -> None:
    """Bar chart comparing zero-shot vs self-trained model."""
    metrics = self_train['metrics']
    models  = [m['Model'] for m in metrics]
    f1s     = [m['Macro-F1'] for m in metrics]
    accs    = [m['Accuracy'] for m in metrics]

    x   = np.arange(len(models))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))

    colors = ['#4C72B0', '#C44E52']
    ax.bar(x - w/2, f1s,  w, label='Macro-F1',  color=colors)
    ax.bar(x + w/2, accs, w, label='Accuracy',   color=colors, alpha=0.5)

    for i, (f, a) in enumerate(zip(f1s, accs)):
        ax.text(i - w/2, f + 0.01, f'{f:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax.text(i + w/2, a + 0.01, f'{a:.3f}', ha='center', fontsize=9)

    if len(f1s) == 2:
        delta = f1s[1] - f1s[0]
        sign  = '+' if delta >= 0 else ''
        ax.annotate(f'ΔF1 = {sign}{delta:.3f}',
                    xy=(0.5, max(f1s) + 0.06), xycoords=('data', 'data'),
                    ha='center', fontsize=11, color='darkgreen' if delta > 0 else 'crimson',
                    fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Score')
    ax.set_title('Self-Training Adaptation: Before vs After')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, '5_self_training_comparison')


def fig_full_comparison(baseline: dict, finetune: dict,
                        few_shot: list[dict], self_train: dict) -> None:
    """
    4-panel summary figure combining all key results:
      [0] IMDb test F1 by model
      [1] Zero-shot tweet F1 by model
      [2] Few-shot learning curve
      [3] Self-training before/after
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Sentiment Domain Shift: Full Evaluation Summary', fontsize=14, fontweight='bold')

    # --- Panel 0: IMDb test F1 ---
    ax = axes[0, 0]
    all_imdb = {**baseline['imdb'], **finetune['imdb']}
    models_imdb = list(all_imdb.keys())
    f1s_imdb    = [all_imdb[m]['test']['macro_f1'] for m in models_imdb]
    colors0     = [PALETTE.get(m, '#888') for m in models_imdb]
    bars0 = ax.bar(models_imdb, f1s_imdb, color=colors0, width=0.5)
    for b in bars0:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f'{b.get_height():.3f}', ha='center', fontsize=8)
    ax.set_title('IMDb Test Macro-F1')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Macro-F1')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)

    # --- Panel 1: Zero-shot tweet F1 ---
    ax = axes[0, 1]
    best_all = {**baseline.get('best_thresholds', {}), **finetune.get('best_thresholds', {})}
    all_zs   = {**baseline['tweet_zeroshot'], **finetune['tweet_zeroshot']}
    zs_models, zs_f1s = [], []
    for m, bt in best_all.items():
        zs = all_zs.get(m, {})
        key = bt if bt in zs else str(bt)
        if key in zs:
            zs_models.append(m)
            zs_f1s.append(zs[key]['macro_f1'])
    colors1 = [PALETTE.get(m, '#888') for m in zs_models]
    bars1 = ax.bar(zs_models, zs_f1s, color=colors1, width=0.5)
    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f'{b.get_height():.3f}', ha='center', fontsize=8)
    ax.set_title('Zero-Shot Tweet Macro-F1')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Macro-F1')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)

    # --- Panel 2: Few-shot learning curve ---
    ax = axes[1, 0]
    df_fs  = pd.DataFrame(few_shot).sort_values('k')
    ks     = df_fs['k'].tolist()
    f1s_fs = df_fs['macro_f1'].tolist()
    xlabels = ['ZS' if k == 0 else f'K={k}' for k in ks]
    x_fs    = np.arange(len(ks))
    ax.plot(x_fs, f1s_fs, marker='o', color='steelblue', linewidth=2)
    for i, f in enumerate(f1s_fs):
        ax.annotate(f'{f:.3f}', (i, f), textcoords='offset points',
                    xytext=(0, 6), ha='center', fontsize=8)
    ax.set_xticks(x_fs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_title('Few-Shot Learning Curve (RoBERTa)')
    ax.set_ylabel('Macro-F1')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # --- Panel 3: Self-training comparison ---
    ax = axes[1, 1]
    st_metrics = self_train['metrics']
    st_models  = [m['Model'] for m in st_metrics]
    st_f1s     = [m['Macro-F1'] for m in st_metrics]
    colors3    = ['#4C72B0', '#C44E52'][:len(st_models)]
    bars3 = ax.bar(st_models, st_f1s, color=colors3, width=0.4)
    for b in bars3:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f'{b.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Self-Training: Before vs After')
    ax.set_ylabel('Macro-F1')
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=10)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    _save(fig, '6_full_summary')


# ── 5. Save evaluation summary JSON ───────────────────────────────────────────

def save_summary(imdb_df: pd.DataFrame, zs_df: pd.DataFrame,
                 gap_df: pd.DataFrame, few_shot: list[dict],
                 self_train: dict) -> None:
    def _df_to_dict(df: pd.DataFrame) -> dict:
        return json.loads(df.to_json(orient='index'))

    summary = {
        'imdb_performance':        _df_to_dict(imdb_df),
        'zeroshot_performance':    _df_to_dict(zs_df),
        'domain_gap':              _df_to_dict(gap_df),
        'few_shot_curve':          few_shot,
        'self_training':           self_train['metrics'],
        'accepted_pseudo_labels':  self_train.get('accepted_samples'),
        'total_unlabelled':        self_train.get('total_unlabelled'),
    }
    out = LOGS / 'evaluation_summary.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n  ✓  Evaluation summary saved → {out}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print('\n' + '='*60)
    print('  Evaluation & Analysis  —  Dariga Borasheva')
    print('='*60)

    # Load logs
    print('\nLoading logs...')
    baseline   = _load(LOGS / 'training_and_testing_metrics.json')
    finetune   = _load(LOGS / 'finetune_metrics.json')
    few_shot   = _load(LOGS / 'few_shot_metrics.json')
    self_train = _load(LOGS / 'self_training_metrics.json')
    print('  All logs loaded successfully.')

    # Tables
    imdb_df = table_imdb(baseline, finetune)
    zs_df   = table_zeroshot(baseline, finetune)
    gap_df  = table_domain_gap(baseline, finetune)

    # Individual figures
    print('\nGenerating figures...')
    fig_imdb_comparison(imdb_df)
    fig_zeroshot_comparison(zs_df)
    fig_domain_gap(baseline, finetune)
    fig_few_shot_curve(few_shot)
    fig_self_training(self_train)
    fig_full_comparison(baseline, finetune, few_shot, self_train)

    # Summary JSON
    save_summary(imdb_df, zs_df, gap_df, few_shot, self_train)

    print('\n' + '='*60)
    print('  Evaluation complete. All figures in ../figures/')
    print('='*60)


if __name__ == '__main__':
    main()
# Evaluation and analysis
