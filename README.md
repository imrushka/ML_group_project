# Sentiment Classifier with Domain Shift: Reviews → Tweets

This project explores domain adaptation for sentiment analysis, moving from formal movie reviews (IMDb) to informal social media posts

Week 9 deliverable — data collection + cleaning.
Week 10-11 delivarable — training and testing baseline models (Log Regressioand and Linear SVM for now) and comparison in 2 different domains

## Layout
```
src/
  config.py             shared paths, seed, label schema
  data_collection.py    Task 1 (Imran): pulls IMDb + TweetEval, writes data/raw/
  data_cleaning.py      Task 2 (Ivan): cleans, balances, splits → data/processed/
  baseline_models.py    Task 3 (Angsar): LR + SVM training, threshold-based zero-shot evaluation
data/
  raw/                
    imdb_raw.csv
    tweeteval_raw.csv
    manifest.json
  processed/          
    imdb_{train,val,test}.csv
    tweet_final_test.csv,
    tweet_unlabelled_pool.csv, 
    _tweet_pool_labels_DIAGNOSTIC_ONLY.csv
logs/                 
  collect.log 
  clean.log
  training_and_testing_metric.json
/notebooks            
  baseline_models.ipynb
/models               
  lr_model.joblib
  svm_model.joblib
  tfidf_vectorizer.joblib
```

## Reproduce
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/data_collection.py
.venv/bin/python src/data_cleaning.py
.venv/bin/python src/baseline_models.py
```
Fixed seed (42). Label schema: `{negative:0, neutral:1, positive:2}`.

## Row counts produced
- IMDb train/val/test: 35 000 / 7 500 / 7 500  (binary, balanced)
- Tweet final_test:    1 740  (3-class, balanced from ~12 k test split)
- Tweet unlabelled pool: 9 283  (labels stripped, for adaptation)

## Known caveat (after week 9)
The released IMDb dataset has no neutral class (reviews 5–6 stars were
excluded by the original authors). Source-domain training is therefore
binary; the shared 3-class schema is still used so the target-domain
tweet eval and adaptation stages can report neutral-class F1.
Options for Week 10+: (a) keep binary IMDb and predict 3-class on
tweets with a neutral-threshold head; (b) mix in a small neutral
source (e.g., SST-3 or Amazon 3-star reviews). Decision lives with the
model-training owner.

## Known caveat (after week 10-11)
The IMDb dataset lacks a "neutral" class (reviews with 5–6 stars were excluded)
Week 10 Strategy: Maintain binary training on IMDb
The Week 11: Zero-Shot stage: implement a confidence threshold to predict the "neutral" class on tweets, providing a more realistic baseline for the upcoming adaptation tasks
