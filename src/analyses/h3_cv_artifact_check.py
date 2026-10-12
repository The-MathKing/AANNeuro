"""Reference distribution for the age+sex-only 5-fold CV AUC inside the lockbox (Table 'h3controls').
Pooled out-of-fold AUC of an uninformative model is biased below 0.5 (held-out fold has the opposite class balance
to its training folds). This script compares the observed value with (i) 200 CV re-seedings and (ii) 200
label-permuted datasets, so the value is read against its own null rather than against 0.5."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.load import load_gse132903


def run_h3_cv_artifact_check(n=200):
    m, _ = load_gse132903()
    lb = m[m.diagnosis.isin(["CTL", "AD"])]
    X = np.c_[lb.age.values, (lb.sex == "FEMALE").astype(float)]
    y = (lb.diagnosis == "AD").astype(int).values

    def pooled(yy, seed):
        p = np.zeros(len(yy))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yy):
            p[b] = LogisticRegression(max_iter=1000).fit(X[a], yy[a]).predict_proba(X[b])[:, 1]
        return roc_auc_score(yy, p)

    real = np.array([pooled(y, s) for s in range(n)])
    rng = np.random.default_rng(0)
    null = np.array([pooled(rng.permutation(y), s) for s in range(n)])
    # like-for-like null: the observed statistic is a mean over n re-seedings, so average each permuted dataset over 10 seeds
    rng2 = np.random.default_rng(1)
    null_avg = np.array([np.mean([pooled(yp, s) for s in range(10)]) for yp in (rng2.permutation(y) for _ in range(200))])
    out = {"null_avg_mean": float(null_avg.mean()), "null_avg_p5": float(np.percentile(null_avg, 5)),
           "null_avg_p95": float(np.percentile(null_avg, 95)), "frac_null_avg_le_real_mean": float((null_avg <= real.mean()).mean()),
           "seed42_value": float(pooled(y, 42)), "n_repeats": n, "real_mean": float(real.mean()), "real_sd": float(real.std()), "real_max": float(real.max()),
           "null_mean": float(null.mean()), "null_p5": float(np.percentile(null, 5)), "null_p95": float(np.percentile(null, 95)),
           "frac_null_le_real_mean": float((null <= real.mean()).mean())}
    json.dump(out, open("results/h3_cv_artifact_check.json", "w"), indent=2)
    return out


if __name__ == "__main__":
    print(run_h3_cv_artifact_check())
