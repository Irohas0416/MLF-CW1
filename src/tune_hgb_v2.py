import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import make_scorer, r2_score

RANDOM_STATE = 42

trn = pd.read_csv("CW1_train.csv")
tst = pd.read_csv("CW1_test.csv")

X = pd.get_dummies(trn.drop(columns=["outcome"]))
y = trn["outcome"]
X_test = pd.get_dummies(tst).reindex(columns=X.columns, fill_value=0)

pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("m", HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        early_stopping=False
    ))
])

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scorer = make_scorer(r2_score)

param_dist = {
    "m__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
    "m__max_iter": [200, 400, 800, 1200, 2000],
    "m__max_depth": [None, 2, 3, 4, 5],
    "m__max_leaf_nodes": [31, 63, 127, 255],
    "m__min_samples_leaf": [10, 20, 40, 80, 120],
    "m__l2_regularization": [0.0, 0.1, 1.0, 5.0, 10.0],
    "m__max_bins": [128, 255],
}

search = RandomizedSearchCV(
    pipe, param_dist, n_iter=60,
    scoring=scorer, cv=cv,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=1
)
search.fit(X, y)

print("\nBest CV R2:", search.best_score_)
print("Best params:", search.best_params_)

best_model = search.best_estimator_
best_model.fit(X, y)
pred = best_model.predict(X_test)

pd.DataFrame({"yhat": pred}).to_csv("CW1_submission_K23154082-5.csv", index=False)
print("Saved submission.")