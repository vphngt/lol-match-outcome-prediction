import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib


def run_baseline_eval(X_train, y_train, SEED=42, cv=TimeSeriesSplit(n_splits=3)):
    
    # dumb baseline
    print(f"Majority class baseline: {round(y_train.value_counts(normalize=True).max(), 3)}")

    # Elo-only baseline
    elo_only_auc = cross_val_score(LogisticRegression(), X_train[['elo_delta']], y_train, cv=cv, scoring='roc_auc').mean()
    print(f"Elo-only AUC: {elo_only_auc:.2f}")

    # comparing baseline models
    baseline_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        'XGBoost':            XGBClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)}

    baseline_scores = {}
    print(f"\n{'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10} {'F1-Score':>10}")
    print('─' * 58)

    for name, model in baseline_models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=["accuracy", "roc_auc", "f1"])
        acc, auc, f1 = scores["test_accuracy"].mean(), scores["test_roc_auc"].mean(), scores["test_f1"].mean()
        baseline_scores[name] = {"acc": acc, "auc": auc, "f1": f1}
        print(f"{name:<25} {acc:>10.3f} {auc:>10.3f} {f1:>10.3f}")

    best_model_name = max(baseline_scores, key=lambda x: baseline_scores[x]["auc"])
    print(f"\nBest baseline model: {best_model_name} (ROC-AUC = {baseline_scores[best_model_name]['auc']:.3f})\n")