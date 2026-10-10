"""Platform-vs-task check for the failed GSE118553 -> MSBB transfer (H3 secondary).
The transfer test changes platform AND label (clinical AD -> Braak >= IV) at once. Here the same model class
(top-1000 in-fold variance genes, z-score, L2 logistic regression with inner CV) is trained and tested WITHIN MSBB
temporal cortex (BM22, BM20, BM38; the samples used in the transfer test) for the same label, with donor-grouped
5-fold CV. If this also fails, platform is not the explanation."""
import json
import numpy as np
import pandas as pd
from src.load import load_gse84422
from src.eval.cross_val import get_donor_grouped_splits
from src.eval.metrics import calc_auc
from src.eval.bootstrap import donor_bootstrap_ci
from src.eval.ledger_logger import log_analysis
from src.models.linear_pipeline import InFoldLinearModel


def run_h3_msbb_internal():
    meta, expr = load_gse84422()
    t = meta[meta.region.isin(["BM22_STG", "BM20_ITG", "BM38_TP"])].dropna(subset=["braak"]).copy()
    X = expr[t.index].T
    y = (t.braak >= 4).astype(int).values
    d = t.donor_id.values
    pred = np.full(len(t), np.nan)
    for tr, te in get_donor_grouped_splits(d, y=y, n_splits=5, seed=42):
        m = InFoldLinearModel("ridge_classifier", top_hvg=1000, seed=42).fit_transform_train(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
    auc = calc_auc(y, pred)
    ci = donor_bootstrap_ci(lambda x: calc_auc(x.y_true.values, x.y_pred.values),
                            pd.DataFrame({"y_true": y, "y_pred": pred, "donor_id": d}), n_boot=2000)
    out = {"n_samples": int(len(t)), "n_donors": int(len(set(d))), "n_positive": int(y.sum()),
           "auc": float(auc), "ci_lower": float(ci["ci_lower"]), "ci_upper": float(ci["ci_upper"])}
    json.dump(out, open("results/h3_msbb_internal.json", "w"), indent=2)
    log_analysis(analysis_id="H3_MSBB_INT_001", category="SENSITIVITY",
                 hypothesis_or_idea="Within-MSBB temporal Braak>=IV classifier (platform vs task)",
                 description="Same model class trained and tested inside MSBB temporal cortex, donor-grouped 5-fold CV",
                 cohorts_used="GSE84422", regions_analyzed="BM22/BM20/BM38", model_type="InFold_LogisticCV",
                 covariates_adjusted="None", sample_size_donors=out["n_donors"], primary_metric="AUC",
                 point_estimate=auc, ci_95_lower=ci["ci_lower"], ci_95_upper=ci["ci_upper"], p_value=float("nan"),
                 replicated_in_lockbox=False, status="COMPLETE", notes="Post hoc, added after review.")
    return out


if __name__ == "__main__":
    print(run_h3_msbb_internal())
