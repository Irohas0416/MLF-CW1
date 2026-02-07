import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

RANDOM_STATE = 42

trn = pd.read_csv("CW1_train.csv")
X = pd.get_dummies(trn.drop(columns=["outcome"]))
y = trn["outcome"]

# 你的最优参数
model = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="median")),
    ("m", HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        min_samples_leaf=80,
        max_leaf_nodes=63,
        max_depth=3,
        max_bins=128,
        learning_rate=0.08,
        l2_regularization=0.1
    ))
])

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# OOF 预测：每个样本用“没见过它”的fold模型预测
y_oof = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
print("OOF R2:", r2_score(y, y_oof))

residual = y - y_oof

# 图1：预测 vs 真实
plt.figure()
plt.scatter(y, y_oof, s=8)
plt.xlabel("True outcome")
plt.ylabel("OOF prediction")
plt.title("OOF Prediction vs True")
plt.tight_layout()
plt.savefig("oof_pred_vs_true.png", dpi=200)

# 图2：残差 vs 预测
plt.figure()
plt.scatter(y_oof, residual, s=8)
plt.axhline(0)
plt.xlabel("OOF prediction")
plt.ylabel("Residual (true - pred)")
plt.title("Residuals vs Prediction (OOF)")
plt.tight_layout()
plt.savefig("oof_residuals.png", dpi=200)

print("Saved: oof_pred_vs_true.png, oof_residuals.png")