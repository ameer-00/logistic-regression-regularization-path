"""
Stretch 5A-S2 — Learning Curves Diagnostic
Telecom Churn Dataset | Logistic Regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, StratifiedKFold

# ─────────────────────────────────────────────────────────
# 1.  Load & preprocess  (same pipeline as Stretch 5A-S1)
# ─────────────────────────────────────────────────────────
df = pd.read_csv('telecom_churn.csv')

X = df.drop(['customer_id', 'churned'], axis=1)
y = df['churned']

X = pd.get_dummies(X, drop_first=True)   # encode categoricals

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────────────────
# 2.  Scoring metric: ROC-AUC
#
#     Justification: the dataset is imbalanced (~84 % no-churn,
#     ~16 % churn). Accuracy is misleading here — a trivial
#     "always predict no-churn" classifier would score ~84 %.
#     ROC-AUC measures the model's ability to *rank* churners
#     above non-churners across all decision thresholds and is
#     insensitive to class prevalence, making it the appropriate
#     metric for this task.
# ─────────────────────────────────────────────────────────
SCORING     = 'roc_auc'
CV          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
TRAIN_SIZES = np.linspace(0.10, 1.0, 10)   # 10 sizes: 10 % → 100 %

# ─────────────────────────────────────────────────────────
# 3.  Three models with different regularization strengths
#     to make the bias-variance tradeoff concrete
# ─────────────────────────────────────────────────────────
models = {
    "LR  C=0.01\n(strong regularization)": LogisticRegression(
        C=0.01, penalty='l2', solver='saga', max_iter=10000, tol=1e-3
    ),
    "LR  C=1.0\n(default regularization)": LogisticRegression(
        C=1.0,  penalty='l2', solver='saga', max_iter=10000, tol=1e-3
    ),
    "LR  C=100\n(weak regularization)": LogisticRegression(
        C=100,  penalty='l2', solver='saga', max_iter=10000, tol=1e-3
    ),
}

colors = ["#DC267F", "#785EF0", "#FE6100"]

# ─────────────────────────────────────────────────────────
# 4.  Plot learning curves
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle(
    "Learning Curves — Logistic Regression on Telecom Churn\n"
    "Scoring: ROC-AUC  |  5-fold Stratified CV  |  Shaded region = ±1 SD",
    fontsize=13
)

for ax, (name, model), color in zip(axes, models.items(), colors):
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator   = model,
        X           = X_scaled,
        y           = y,
        train_sizes = TRAIN_SIZES,
        cv          = CV,
        scoring     = SCORING,
        n_jobs      = -1,
    )

    tm = train_scores.mean(axis=1);  ts = train_scores.std(axis=1)
    vm = val_scores.mean(axis=1);    vs = val_scores.std(axis=1)

    # Training curve + band
    ax.plot(train_sizes_abs, tm,
            color=color, lw=2, marker='o', label='Training score')
    ax.fill_between(train_sizes_abs, tm - ts, tm + ts,
                    alpha=0.15, color=color)

    # Validation curve + band
    ax.plot(train_sizes_abs, vm,
            color=color, lw=2, marker='s', linestyle='--',
            label='Validation score')
    ax.fill_between(train_sizes_abs, vm - vs, vm + vs,
                    alpha=0.30, color=color)

    gap = tm[-1] - vm[-1]
    ax.set_title(f"{name}\nFinal gap (train−val): {gap:.3f}", fontsize=10)
    ax.set_xlabel("Training set size", fontsize=10)
    ax.set_ylabel("ROC-AUC", fontsize=10)
    ax.set_ylim(0.50, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('learning_curve_plot.png', dpi=150, bbox_inches='tight')
print("Plot saved → learning_curve_plot.png")
plt.show()