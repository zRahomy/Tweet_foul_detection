
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

RANDOM_STATE = 42


def clean_text(s):
    import re
    s = str(s).lower()
    s = re.sub(r'http\S+|www\.\S+', '', s)
    s = re.sub(r'@\w+', '', s)
    s = re.sub(r'#', '', s)
    s = re.sub(r'[^a-z\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def load_and_prep(path):
    df = pd.read_csv(path)
    if 'class' in df.columns:
        df['label'] = df['class'].map(lambda x: 1 if x in [0, 1] else 0)
    elif {'hate_speech', 'offensive_language'}.issubset(df.columns):
        df['label'] = df[['hate_speech', 'offensive_language']].max(axis=1)
    else:
        raise ValueError("No valid label column found")

    df['text'] = df.get('text', df.get('tweet', df.iloc[:, 0])).astype(str)
    df['clean_text'] = df['text'].apply(clean_text)
    return df[['clean_text', 'label']]


def choose_threshold(model, X_val_vec, y_val, recall_target=0.8):
    probs = model.predict_proba(X_val_vec)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probs)

    thresholds_ext = list(thresholds) + [1.0]
    candidates = []
    for p, r, t in zip(precision, recall, thresholds_ext):
        if r >= recall_target:
            f1 = (2 * p * r) / (p + r + 1e-9)
            candidates.append((t, p, r, f1))
    if candidates:
        best = max(candidates, key=lambda x: (x[1], x[3]))
        thr = float(best[0])
    else:
        f1s = (2 * precision * recall) / (precision + recall + 1e-9)
        idx = np.nanargmax(f1s)
        thr = float(np.append(thresholds, 1.0)[idx])

    
    thr = float(np.clip(thr, 0.4, 0.95))
    return thr


def train_and_save_model(df, recall_target, outpath):
    X = df['clean_text']
    y = df['label']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = {0: weights[0] * 0.7, 1: weights[1] * 0.7}

    clf = LogisticRegression(
        solver='saga',
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight=class_weight
    )

    gs = GridSearchCV(clf, {'C': [0.1, 1.0, 5.0]}, cv=3, scoring='f1', n_jobs=-1)
    gs.fit(X_train_vec, y_train)
    best = gs.best_estimator_

    threshold = choose_threshold(best, X_val_vec, y_val, recall_target)
    probs_test = best.predict_proba(X_test_vec)[:, 1]
    preds_test = (probs_test >= threshold).astype(int)

    print(f"\n=== Model (Recall target={recall_target}) ===")
    print(classification_report(y_test, preds_test))
    print("ROC AUC:", roc_auc_score(y_test, probs_test))
    print("PR AUC:", average_precision_score(y_test, probs_test))
    print("Chosen threshold:", threshold)

    artifact = {
        'vectorizer': vectorizer,
        'model': best,
        'threshold': threshold
    }

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'wb') as f:
        pickle.dump(artifact, f)
    print(" Saved:", outpath)


if __name__ == '__main__':
    df = load_and_prep("data/labeled_data.csv")

    train_and_save_model(df, 0.85, "model_artifacts/foul_detector_low.pkl")
    train_and_save_model(df, 0.9, "model_artifacts/foul_detector_medium.pkl")
    train_and_save_model(df, 0.95, "model_artifacts/foul_detector_high.pkl")
