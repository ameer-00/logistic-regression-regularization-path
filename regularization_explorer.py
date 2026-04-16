import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    df = pd.read_csv('telecom_churn.csv')
except FileNotFoundError:
    print("Error: telecom_churn.csv not found!")
    exit()

X = df.drop(['customer_id', 'churned'], axis=1)
y = df['churned']

X = pd.get_dummies(X, drop_first=True)
feature_names = X.columns

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cs = np.logspace(-3, 2, 20)

coefs_l1 = []
coefs_l2 = []

for c in cs:
    # L1 (Lasso)
    model_l1 = LogisticRegression(penalty='l1', C=c, solver='saga', max_iter=10000, tol=1e-3)
    model_l1.fit(X_scaled, y)
    coefs_l1.append(model_l1.coef_[0])
    
    # L2 (Ridge)
    model_l2 = LogisticRegression(penalty='l2', C=c, solver='saga', max_iter=10000, tol=1e-3)
    model_l2.fit(X_scaled, y)
    coefs_l2.append(model_l2.coef_[0])

coefs_l1 = np.array(coefs_l1)
coefs_l2 = np.array(coefs_l2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

for i in range(len(feature_names)):
    ax1.plot(cs, coefs_l1[:, i], label=feature_names[i])
    ax2.plot(cs, coefs_l2[:, i], label=feature_names[i])

ax1.set_xscale('log')
ax1.set_title('L1 Regularization Path (Lasso)\nFeatures eliminate to zero')
ax1.set_xlabel('C (Weakens Penalty →)')
ax1.set_ylabel('Coefficient Magnitude')
ax1.grid(True, alpha=0.3)

ax1.annotate('First eliminated features', xy=(0.01, 0), xytext=(0.05, 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))

ax2.set_xscale('log')
ax2.set_title('L2 Regularization Path (Ridge)\nFeatures shrink but persist')
ax2.set_xlabel('C (Weakens Penalty →)')
ax2.grid(True, alpha=0.3)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.savefig('regularization_plot.png') # حفظ الرسم كملف
plt.show()