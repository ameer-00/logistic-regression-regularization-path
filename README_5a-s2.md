# Stretch 5A-S2 — Learning Curves Diagnostic

## Setup

This stretch builds directly on Stretch 5A-S1 (regularization path explorer).
The same preprocessing pipeline is reused: one-hot encoding of categoricals,
`StandardScaler`, and logistic regression with `saga` solver.

**Scoring metric chosen: ROC-AUC**
The dataset is imbalanced (~84 % no-churn, ~16 % churn). Accuracy is misleading
in this setting — a trivial classifier that predicts "no churn" for every customer
would already score ~84 % accuracy while learning nothing useful. ROC-AUC measures
the model's ability to *rank* churners above non-churners across all decision
thresholds and is unaffected by class prevalence, making it the appropriate metric
for this task.

---

## Learning Curve Results (full training set, 5-fold stratified CV)

| Model | Train ROC-AUC | Val ROC-AUC | Gap |
|---|---|---|---|
| LR C=0.01 (strong reg.) | 0.709 ± 0.008 | 0.669 ± 0.035 | 0.040 |
| LR C=1.0  (default)     | 0.710 ± 0.009 | 0.670 ± 0.030 | 0.040 |
| LR C=100  (weak reg.)   | 0.710 ± 0.009 | 0.670 ± 0.030 | 0.040 |

---

## Written Analysis

**Bias-Variance Diagnosis**

The learning curves reveal a clear **high-bias (underfitting)** pattern across all
three regularization settings. Both training and validation ROC-AUC curves plateau
at a low value — approximately 0.71 and 0.67 respectively — and converge closely
together as more data is added. The gap between training and validation performance
is small and nearly identical (~0.04) regardless of whether regularization is strong
(C=0.01) or weak (C=100). This is the textbook signature of high bias: the model
lacks the capacity to capture the underlying patterns in the data, and
regularization strength has almost no effect because the dominant problem is not
overfitting but underfitting. Notably, the validation confidence bands are moderately
wide (±0.030–0.035), indicating some instability in generalization across folds —
a secondary signal that the model is not robustly capturing the signal in the data.

**Actionable Answers to the Diagnostic Questions**

1. **High bias or high variance?**
   High bias. Both curves converge at a low plateau with a small gap between them.
   The model is too simple to fit the data regardless of how much regularization
   is applied.

2. **Would collecting more data help?**
   No. When a model suffers from high bias, adding more training examples does
   not help — the curves have already flattened well before the full dataset size
   is reached. The validation curve stops improving around 60–70 % of the data,
   confirming that the bottleneck is model capacity, not data quantity.

3. **Would increasing model complexity help?**
   Yes — this is the right lever here. Since the problem is high bias, switching
   to a more flexible model (e.g., gradient boosted trees, Random Forest, or
   logistic regression with polynomial/interaction features) is likely to raise
   both training and validation scores meaningfully. Adding polynomial features
   to logistic regression is a lower-risk first step that would test whether the
   relationships in the data are nonlinear without abandoning the interpretable
   linear framework.

4. **Recommended next step**
   Engineer interaction and polynomial features (e.g., `tenure × contract_type`,
   `monthly_charges²`) and re-run the learning curve. If the gap widens
   significantly (training score rises but validation does not follow), the
   problem will have shifted toward high variance and regularization or a larger
   dataset would then become the right tool. Alternatively, try a gradient boosted
   classifier (e.g., `HistGradientBoostingClassifier`) as a direct comparison —
   its learning curve will confirm whether a more flexible model can extract
   signal that logistic regression cannot.