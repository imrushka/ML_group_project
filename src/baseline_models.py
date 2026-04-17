import time
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, accuracy_score, ConfusionMatrixDisplay

NEUTRAL_LABEL = 1
THRESHOLDS = [0.55, 0.65, 0.75]

def predict_with_threshold(model, X, threshold: float) -> np.ndarray:
    """
    If max class probability < threshold → predict neutral (1).
    Otherwise predict the most likely class.
    """
    proba = model.predict_proba(X)
    preds = np.argmax(proba, axis=1)
    max_proba = proba.max(axis=1)
    classes = model.classes_
    preds_mapped = np.array([classes[p] for p in preds])
    preds_mapped[max_proba < threshold] = NEUTRAL_LABEL
    return preds_mapped

def main():
    train_df = pd.read_csv('../data/processed/imdb_train.csv')
    val_df   = pd.read_csv('../data/processed/imdb_val.csv')
    test_df  = pd.read_csv('../data/processed/imdb_test.csv')

    print('Split sizes:')
    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f'  {name:5s}: {len(df):>6,} rows | {df["label_str"].value_counts().to_dict()}')

    train_df['word_count'] = train_df['text_clean'].str.split().str.len()


    #===TfidVectorizer===

    SEED = 42
    Path('../models').mkdir(exist_ok=True)
    Path('../logs').mkdir(exist_ok=True)

    ID2LABEL = {0: 'negative', 1: 'neutral', 2: 'positive'}

    X_train_raw = train_df['text_clean'].fillna('')
    X_val_raw   = val_df['text_clean'].fillna('')
    X_test_raw  = test_df['text_clean'].fillna('')
    y_train = train_df['label'].astype(int)
    y_val   = val_df['label'].astype(int)
    y_test  = test_df['label'].astype(int)

    vec = TfidfVectorizer(
        ngram_range=(1, 2), max_features=50_000,
        sublinear_tf=True, min_df=3, strip_accents='unicode'
    )
    X_train = vec.fit_transform(X_train_raw)  # fit only on train — no leakage
    X_val   = vec.transform(X_val_raw)
    X_test  = vec.transform(X_test_raw)

    joblib.dump(vec, '../models/tfidf_vectorizer.joblib')
    print(f'Vocabulary size : {len(vec.vocabulary_):,}')
    print(f'X_train shape   : {X_train.shape}')



    #===Logistic Regression===

    present_labels = sorted(y_train.unique())
    label_names_binary = [ID2LABEL[i] for i in present_labels]

    lr = LogisticRegression(
        C=1.0, max_iter=1000, 
        solver='lbfgs',
        random_state=SEED, 
        n_jobs=-1
    )
    lr.fit(X_train, y_train)
    joblib.dump(lr, '../models/lr_model.joblib')

    for split_name, X, y in [('val', X_val, y_val), ('test', X_test, y_test)]:
        print(f'\n── LR | {split_name} ──')
        print(classification_report(y, lr.predict(X), target_names=label_names_binary, zero_division=0))


    #===Linear SVM===

    svm = LinearSVC(C=1.0, max_iter=2000, random_state=SEED, dual='auto')
    t0 = time.time()
    svm.fit(X_train, y_train)
    joblib.dump(svm, '../models/svm_model.joblib')

    for split_name, X, y in [('val', X_val, y_val), ('test', X_test, y_test)]:
        print(f'\n── SVM | {split_name} ──')
        print(classification_report(y, svm.predict(X), target_names=label_names_binary, zero_division=0))


    #===IMDB results summary===

    imdb_results = {}
    for model_name, model in [('LogisticRegression', lr), ('LinearSVC', svm)]:
        imdb_results[model_name] = {}
        for split_name, X, y in [('val', X_val, y_val), ('test', X_test, y_test)]:
            y_pred = model.predict(X)
            imdb_results[model_name][split_name] = {
                'accuracy': round(accuracy_score(y, y_pred), 4),
                'macro_f1': round(f1_score(y, y_pred, average='macro', zero_division=0), 4),
            }

    rows = []
    for m, splits in imdb_results.items():
        for s, metrics in splits.items():
            rows.append({'model': m, 'split': s, **metrics})
    print(pd.DataFrame(rows).set_index(['model','split']).to_string())


    #===Zero Shot Evaluation on Tweets===

    tweet_df = pd.read_csv('../data/processed/tweet_final_test.csv')
    print(f'\n\n\nTweet test rows: {len(tweet_df):,}')
    print(tweet_df['label_str'].value_counts())

    X_tweet_raw = tweet_df['text_clean'].fillna('')
    y_tweet = tweet_df['label'].astype(int)   # 0=neg, 1=neutral, 2=pos

    X_tweet = vec.transform(X_tweet_raw) # transform with the same vectrozier
    print(f'X_tweet shape: {X_tweet.shape}')

    svm_calibrated = CalibratedClassifierCV(svm, cv=5)
    svm_calibrated.fit(X_val, y_val)

    print('\n\n\nSVM calibrated. Both LR and SVM now output probabilities.')

    label_names_3class = ['negative', 'neutral', 'positive']
    tweet_results = {}

    for model_name, model in [('LogisticRegression', lr), ('LinearSVC', svm_calibrated)]:
        tweet_results[model_name] = {}
        print(f'\n{"="*60}')
        print(f'  {model_name} — Zero-Shot on Tweets')
        print(f'{"="*60}')

        for threshold in THRESHOLDS:
            y_pred = predict_with_threshold(model, X_tweet, threshold)
            acc      = accuracy_score(y_tweet, y_pred)
            macro_f1 = f1_score(y_tweet, y_pred, average='macro', zero_division=0)

            tweet_results[model_name][threshold] = {
                'accuracy': round(acc, 4),
                'macro_f1': round(macro_f1, 4),
            }

            print(f'\n\n threshold={threshold} | Accuracy={acc:.4f} | Macro-F1={macro_f1:.4f}')
            print(classification_report(
                y_tweet, y_pred,
                target_names=label_names_3class,
                zero_division=0
            ))

    best = {}
    for model_name, thresholds in tweet_results.items():
        best_t = max(thresholds, key=lambda t: thresholds[t]['macro_f1'])
        best[model_name] = best_t
        print(f'{model_name}: best threshold = {best_t} | Macro-F1 = {thresholds[best_t]["macro_f1"]}')
    print('\n\n')


    summary_rows = []
    for model_name in ['LogisticRegression', 'LinearSVC']:
        imdb_f1  = imdb_results[model_name]['test']['macro_f1']
        best_t   = best[model_name]
        tweet_f1 = tweet_results[model_name][best_t]['macro_f1']
        drop     = round(imdb_f1 - tweet_f1, 4)
        summary_rows.append({
            'model': model_name,
            'IMDb F1 (binary)': imdb_f1,
            'Tweet F1 (zero-shot)': tweet_f1,
            'Drop': drop,
            'Best threshold': best_t,
        })

    summary_df = pd.DataFrame(summary_rows).set_index('model')
    print(summary_df.to_string())
    print('\n\n')


    #===Saving all metrics===

    all_metrics = {
        'imdb': imdb_results,
        'tweet_zeroshot': tweet_results,
        'best_thresholds': best,
        'domain_gap': {row['model']: row['Drop'] for row in summary_rows},
    }
    with open('../logs/training_and_testing_metrics.json', 'w') as file:
        json.dump(all_metrics, file, indent=2)
    print('Saved → logs/training_and_testing_metrics.json')


if __name__ == '__main__':    main()