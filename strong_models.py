import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

RANDOM_STATE = 42

trn = pd.read_csv("CW1_train.csv")
tst = pd.read_csv("CW1_test.csv")

X = pd.get_dummies(trn.drop(columns=["outcome"]))
y = trn["outcome"]
X_test = pd.get_dummies(tst)

# 对齐列，避免 one-hot 列不一致
X_test = X_test.reindex(columns=X.columns, fill_value=0)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scorer = make_scorer(r2_score)

def eval_model(name, model):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
    print(f"{name:>28s}: {scores.mean():.5f} ± {scores.std():.5f}")
    return scores.mean(), scores.std()

imputer = SimpleImputer(strategy="median")

models = [
    ("Ridge(alpha=100)", Pipeline([("imp", imputer), ("m", Ridge(alpha=100.0, random_state=RANDOM_STATE))])),
    ("RandomForest",     Pipeline([("imp", imputer), ("m", RandomForestRegressor(
                            n_estimators=600,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            max_features="sqrt"
                        ))])),
    ("ExtraTrees",       Pipeline([("imp", imputer), ("m", ExtraTreesRegressor(
                            n_estimators=1200,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            max_features="sqrt"
                        ))])),
    ("HistGradientBoosting", Pipeline([("imp", imputer), ("m", HistGradientBoostingRegressor(
                            random_state=RANDOM_STATE,
                            learning_rate=0.05,
                            max_leaf_nodes=63,
                            min_samples_leaf=30
                        ))])),
]

best = None
print("5-fold CV R2 (mean ± std):")
for name, model in models:
    m, s = eval_model(name, model)
    if best is None or m > best[0]:
        best = (m, s, name, model)

print("\nBest so far:", best[2], "=>", f"{best[0]:.5f} ± {best[1]:.5f}")

best_model = best[3]
best_model.fit(X, y)
pred = best_model.predict(X_test)


out = pd.DataFrame({"yhat": pred})
out.to_csv("CW1_submission_K23154082-1.csv", index=False)
print("Saved submission from:", best[2])