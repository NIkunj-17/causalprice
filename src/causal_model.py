import numpy as np
import pandas as pd
import joblib, warnings
warnings.filterwarnings("ignore")

from dowhy import CausalModel
from econml.dml import LinearDML, CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
from src.features import (CONFOUNDER_COLS, EFFECT_MOD_COLS,
                           TREATMENT_COL, OUTCOME_COL)

MODEL_PATH  = "data/cate_model.pkl"
LINEAR_PATH = "data/linear_dml.pkl"
RESULTS_PATH= "data/results.pkl"


def run_causal_pipeline(df: pd.DataFrame, sample_n: int = 40_000) -> dict:

    # ── Held-out category split ────────────────────────────────────────
    all_cats     = df["category"].unique()
    np.random.seed(42)
    n_holdout    = max(1, int(len(all_cats) * 0.12))
    holdout_cats = np.random.choice(all_cats, size=n_holdout, replace=False)
    train_cats   = [c for c in all_cats if c not in holdout_cats]

    df_train = df[df["category"].isin(train_cats)].copy()
    df_hold  = df[df["category"].isin(holdout_cats)].copy()
    print(f"Train categories: {len(train_cats)} | Holdout: {len(holdout_cats)}")
    print(f"Holdout cats: {list(holdout_cats)}")

    if len(df_train) > sample_n:
        df_train = df_train.sample(sample_n, random_state=42).reset_index(drop=True)

    T = df_train[TREATMENT_COL].values
    Y = df_train[OUTCOME_COL].values
    W = df_train[CONFOUNDER_COLS].values
    X = df_train[EFFECT_MOD_COLS].values

    print(f"Y stats — mean: {Y.mean():.3f}  std: {Y.std():.3f}  "
          f"range: [{Y.min():.2f}, {Y.max():.2f}]")
    print(f"T stats — mean: {T.mean():.3f}  std: {T.std():.3f}")

    # ── DoWhy DAG ──────────────────────────────────────────────────────
    edges = " ".join([
        f"{c} -> {TREATMENT_COL}; {c} -> {OUTCOME_COL};"
        for c in CONFOUNDER_COLS
    ])
    graph = f"digraph {{ {TREATMENT_COL} -> {OUTCOME_COL}; {edges} }}"
    dowhy_model = CausalModel(
        data=df_train,
        treatment=TREATMENT_COL,
        outcome=OUTCOME_COL,
        graph=graph,
    )
    estimand = dowhy_model.identify_effect(proceed_when_unidentifiable=True)
    print("Estimand identified.")

    # ── Linear DML (ATE) ───────────────────────────────────────────────
    print("Fitting Linear DML (ATE)...")
    gb = lambda seed=42: GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=seed
    )
    linear_dml = LinearDML(
        model_y=gb(), model_t=gb(),
        cv=5, random_state=42,
    )
    linear_dml.fit(Y, T, X=X, W=W)

    ate      = float(linear_dml.ate(X))
    ate_lb   = float(linear_dml.ate_interval(X, alpha=0.05)[0])
    ate_ub   = float(linear_dml.ate_interval(X, alpha=0.05)[1])
    ate_pval = _approx_pvalue(ate, ate_lb, ate_ub)
    print(f"ATE = {ate:.4f}  CI [{ate_lb:.4f}, {ate_ub:.4f}]  p≈{ate_pval:.4f}")

    # ── Causal Forest DML (CATE) ───────────────────────────────────────
    print("Fitting Causal Forest DML...")
    cf = CausalForestDML(
        model_y=gb(), model_t=gb(),
        n_estimators=600, min_samples_leaf=15,
        max_features="auto",
        cv=5, random_state=42,
    )
    cf.fit(Y, T, X=X, W=W)
    cate_vals = cf.effect(X)
    cate_lb_v, cate_ub_v = cf.effect_interval(X, alpha=0.05)

    print(f"CATE stats — mean: {cate_vals.mean():.4f}  "
          f"std: {cate_vals.std():.4f}  "
          f"range: [{cate_vals.min():.4f}, {cate_vals.max():.4f}]")

    # ── Price elasticity calculation ────────────────────────────────────
    # CATE in our model = dE[log(reviews)] / d(price_pct)
    # Price elasticity ≈ CATE * (price_pct_std / log_reviews_std)
    # This converts to: "1% price increase → X% change in review volume"
    T_std = T.std()
    Y_std = Y.std()
    elasticity_vals = cate_vals * (T_std / (Y_std + 1e-6))
    print(f"Implied elasticity range: [{elasticity_vals.min():.3f}, {elasticity_vals.max():.3f}]")

    # ── Held-out validation ────────────────────────────────────────────
    print("Held-out validation...")
    X_hold   = df_hold[EFFECT_MOD_COLS].values
    T_hold   = df_hold[TREATMENT_COL].values
    Y_hold   = df_hold[OUTCOME_COL].values
    cate_hold= cf.effect(X_hold)

    hi_mask  = T_hold > 0.75
    lo_mask  = T_hold < 0.25
    neg_mask = cate_hold < 0
    obs_hi   = Y_hold[neg_mask & hi_mask].mean() if (neg_mask & hi_mask).sum()>5 else np.nan
    obs_lo   = Y_hold[neg_mask & lo_mask].mean() if (neg_mask & lo_mask).sum()>5 else np.nan
    dir_ok   = bool(obs_hi < obs_lo) if not any(np.isnan([obs_hi, obs_lo])) else None

    holdout = {
        "holdout_categories": list(holdout_cats),
        "n_holdout":          len(df_hold),
        "cate_hold_mean":     float(cate_hold.mean()),
        "cate_hold_std":      float(cate_hold.std()),
        "obs_hi":             obs_hi,
        "obs_lo":             obs_lo,
        "direction_correct":  dir_ok,
    }
    print(f"Holdout CATE mean: {cate_hold.mean():.4f} | direction: {dir_ok}")
    joblib.dump(holdout, "data/holdout_results.pkl")

    # ── Refutation tests ───────────────────────────────────────────────
    print("Refutation tests...")
    estimate = dowhy_model.estimate_effect(
        estimand, method_name="backdoor.linear_regression"
    )
    ref_placebo = dowhy_model.refute_estimate(
        estimand, estimate,
        method_name="placebo_treatment_refuter",
        placebo_type="permute", num_simulations=10,
    )
    ref_random = dowhy_model.refute_estimate(
        estimand, estimate,
        method_name="random_common_cause", num_simulations=10,
    )
    ref_subset = dowhy_model.refute_estimate(
        estimand, estimate,
        method_name="data_subset_refuter",
        subset_fraction=0.8, num_simulations=10,
    )
    print(str(ref_placebo))
    print(str(ref_random))
    print(str(ref_subset))

    # ── Rosenbaum sensitivity ──────────────────────────────────────────
    gamma = _rosenbaum_gamma(ate, ate_lb, ate_ub)
    print(f"Rosenbaum Gamma: {gamma:.2f}")

    # ── Bootstrap CATE curve ───────────────────────────────────────────
    print("Bootstrap CI (200 samples)...")
    boot = _bootstrap_cate_curve(df_train, n_bootstrap=200)
    joblib.dump(boot, "data/bootstrap_curves.pkl")

    # ── Category CATE summary ──────────────────────────────────────────
    df_full = df[df["category"].isin(train_cats)].copy()
    if len(df_full) > sample_n:
        df_full = df_full.sample(sample_n, random_state=42)
    cate_full = cf.effect(df_full[EFFECT_MOD_COLS].values)
    df_full   = df_full.reset_index(drop=True)
    df_full["cate"] = cate_full
    df_full["elasticity"] = cate_full * (T_std / (Y_std + 1e-6))

    cat_cate = (
        df_full.groupby("category")
        .agg(avg_cate=("cate","mean"),
             avg_elasticity=("elasticity","mean"),
             count=("cate","count"))
        .query("count >= 30")
        .reset_index()
    )
    cat_stats = pd.read_csv("data/category_stats.csv")
    cat_stats = cat_stats.merge(
        cat_cate[["category","avg_cate","avg_elasticity"]],
        on="category", how="left"
    )
    cat_stats.to_csv("data/category_stats.csv", index=False)

    results = {
        "ate":              ate,
        "ate_lb":           ate_lb,
        "ate_ub":           ate_ub,
        "ate_pval":         ate_pval,
        "cate_vals":        cate_vals,
        "cate_lb":          cate_lb_v,
        "cate_ub":          cate_ub_v,
        "elasticity_vals":  elasticity_vals,
        "t_std":            T_std,
        "y_std":            Y_std,
        "n_samples":        len(df_train),
        "n_categories":     len(train_cats),
        "refute_placebo":   str(ref_placebo),
        "refute_random":    str(ref_random),
        "refute_subset":    str(ref_subset),
        "rosenbaum_gamma":  gamma,
        "holdout":          holdout,
        "cat_cate":         cat_cate,
        "effect_mod_cols":  EFFECT_MOD_COLS,
        "outcome_col":      OUTCOME_COL,
        "outcome_label":    "log(1 + review count)",
        "outcome_unit":     "log-reviews",
    }
    joblib.dump(cf,          MODEL_PATH)
    joblib.dump(linear_dml,  LINEAR_PATH)
    joblib.dump(results,     RESULTS_PATH)
    print(f"\nDone. ATE={ate:.4f}  Gamma={gamma:.2f}")
    return results


def _approx_pvalue(ate, lb, ub):
    se = (ub - lb) / (2 * 1.96)
    if se < 1e-9:
        return 1.0
    z = abs(ate) / se
    from scipy.stats import norm
    return float(2 * (1 - norm.cdf(z)))


def _rosenbaum_gamma(ate, lb, ub):
    se = (ub - lb) / (2 * 1.96)
    if se < 1e-9:
        return 1.0
    return round(float(np.exp(abs(ate) / (se + 1e-9))), 2)


def _bootstrap_cate_curve(df_train, n_bootstrap=200):
    grid    = np.linspace(
        df_train[EFFECT_MOD_COLS[0]].quantile(0.05),
        df_train[EFFECT_MOD_COLS[0]].quantile(0.95),
        60
    )
    lc_med  = float(df_train[EFFECT_MOD_COLS[1]].median())
    X_grid  = np.column_stack([grid, np.full(60, lc_med)])
    curves  = np.zeros((n_bootstrap, 60))

    for i in range(n_bootstrap):
        df_b = df_train.sample(len(df_train), replace=True, random_state=i)
        cf_b = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                               random_state=i),
            model_t=GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                               random_state=i),
            n_estimators=200, min_samples_leaf=25, cv=3, random_state=i,
        )
        cf_b.fit(
            df_b[OUTCOME_COL].values,
            df_b[TREATMENT_COL].values,
            X=df_b[EFFECT_MOD_COLS].values,
            W=df_b[CONFOUNDER_COLS].values,
        )
        curves[i] = cf_b.effect(X_grid)
        if (i+1) % 25 == 0:
            print(f"  Bootstrap {i+1}/{n_bootstrap}")

    return {"grid": grid, "curves": curves, "lc_median": lc_med,
            "grid_label": EFFECT_MOD_COLS[0]}


def load_model():   return joblib.load(MODEL_PATH)
def load_results(): return joblib.load(RESULTS_PATH)