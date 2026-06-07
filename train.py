"""
Full training pipeline — run once.
Windows: python train.py   (~45-60 min total due to bootstrap)
"""
from src.features import download_data, build_features
from src.causal_model import run_causal_pipeline
import pandas as pd

if __name__ == "__main__":
    download_data()

    df_raw = pd.read_csv("data/products_clean.csv")
    df     = build_features(df_raw)
    df.to_csv("data/features.csv", index=False)
    print("Features saved.\n")

    results = run_causal_pipeline(df, sample_n=40_000)

    sig = results["ate_lb"] > 0 or results["ate_ub"] < 0
    print("\n" + "="*55)
    print("TRAINING COMPLETE")
    print("="*55)
    print(f"Outcome:         log(1 + review_count)")
    print(f"ATE:             {results['ate']:.4f} log-reviews")
    print(f"95% CI:          [{results['ate_lb']:.4f}, {results['ate_ub']:.4f}]")
    print(f"p-value:         {results['ate_pval']:.4f}  {'*** SIGNIFICANT' if sig else '(not significant)'}")
    print(f"CATE range:      [{results['cate_vals'].min():.4f}, {results['cate_vals'].max():.4f}]")
    print(f"Elasticity range:[{results['elasticity_vals'].min():.3f}, {results['elasticity_vals'].max():.3f}]")
    print(f"Rosenbaum Gamma: {results['rosenbaum_gamma']:.2f}")
    print(f"Holdout dir:     {results['holdout']['direction_correct']}")
    print("\nRun: python app.py")