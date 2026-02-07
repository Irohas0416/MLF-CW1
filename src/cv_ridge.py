import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, r2_score

trn = pd.read_csv('CW1_train.csv')
tst = pd.read_csv('CW1_test.csv')

X_trn = pd.get_dummies(trn.drop(columns=['outcome']))
y_trn = trn['outcome']
X_tst = pd.get_dummies(tst)

X_tst = X_tst.reindex(columns=X_trn.columns, fill_value=0)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(r2_score)

def cv_r2(model):
    scores = cross_val_score(model, X_trn, y_trn, cv=cv, scoring=scorer)
    return scores.mean(), scores.std()

print("LinearRegression CV R2 (mean±std):", cv_r2(LinearRegression()))

alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
best = None

for a in alphas:
    m, s = cv_r2(Ridge(alpha=a, random_state=42))
    print(f"Ridge(alpha={a}) CV R2 (mean±std): {m:.5f} ± {s:.5f}")
    if best is None or m > best[0]:
        best = (m, s, a)

print("\nBest Ridge:", best)


best_alpha = best[2]
final_model = Ridge(alpha=best_alpha, random_state=42)
final_model.fit(X_trn, y_trn)
yhat = final_model.predict(X_tst)

out = pd.DataFrame({'yhat': yhat})
out.to_csv('CW1_submission_K23154082.csv', index=False)
print("Saved submission with best Ridge alpha:", best_alpha)