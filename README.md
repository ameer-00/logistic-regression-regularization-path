# Regularization Explorer - Telecom Churn

This project visualizes the effect of L1 (Lasso) and L2 (Ridge) regularization on model coefficients using the Telecom Churn dataset.

## Interpretation
The visualization reveals a distinct difference between L1 and L2 regularization. In the **L1 (Lasso)** path, we observe that as the regularization strength increases (smaller C), coefficients for features like `has_dependents` and `senior_citizen` are driven exactly to zero, effectively performing feature selection. In contrast, the **L2 (Ridge)** path shrinks all coefficients toward zero but maintains them in the model.

**Recommendation:** For this dataset, if model interpretability and simplicity are prioritized, **L1 regularization** is preferred as it highlights the most robust predictors (like `tenure` and `contract_type`) while eliminating noise.

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the script: `python regularization_explorer.py`