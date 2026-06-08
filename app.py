"""
CausalPrice — Interactive Dashboard v4
Uses Plotly for live browser-side charts + Gradio Blocks layout
python app.py → http://127.0.0.1:7860
"""
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, textwrap, re
from scipy.stats import norm

# ── Load artifacts ────────────────────────────────────────────────────────
model     = joblib.load("data/cate_model.pkl")
results   = joblib.load("data/results.pkl")
df_feat   = pd.read_csv("data/features.csv")
cat_stats = pd.read_csv("data/category_stats.csv")
boot      = joblib.load("data/bootstrap_curves.pkl")

ATE    = results["ate"]
ATE_LB = results["ate_lb"]
ATE_UB = results["ate_ub"]
PVAL   = results["ate_pval"]
GAMMA  = results["rosenbaum_gamma"]
T_STD  = results["t_std"]
Y_STD  = results["y_std"]
SIG    = ATE_LB > 0 or ATE_UB < 0
sig_star = "***" if PVAL<0.001 else ("**" if PVAL<0.01 else ("*" if PVAL<0.05 else "n.s."))

# Pre-compute
X_all     = df_feat[["log_price_level","log_competition"]].values
all_cate  = model.effect(X_all)
cate_p2   = np.percentile(all_cate, 2)
cate_p98  = np.percentile(all_cate, 98)
all_cate  = np.clip(all_cate, cate_p2, cate_p98)
all_elast = all_cate * (T_STD / (Y_STD + 1e-6))
df_feat["cate"]       = all_cate
df_feat["elasticity"] = all_elast

cat_summary = (
    df_feat.groupby("category")
    .agg(avg_cate=("cate","mean"),
         avg_elasticity=("elasticity","mean"),
         count=("cate","count"),
         med_price=("price","median"),
         med_reviews=("review_count","median"))
    .query("count >= 50")
    .sort_values("avg_elasticity")
    .reset_index()
)
CATEGORIES = sorted(df_feat["category"].dropna().unique().tolist())

# ── Color palette ─────────────────────────────────────────────────────────
C_POS  = "#1D9E75"
C_NEG  = "#D85A30"
C_ATE  = "#534AB7"
C_GRAY = "#888780"
C_YEL  = "#FFD166"
PLOTLY_TEMPLATE = "plotly_white"

def parse_refute(s, key):
    m = re.search(rf"{key}[^:]*:\s*([-\d.eE+]+)", s)
    return float(m.group(1)) if m else np.nan

# ── Core estimator ────────────────────────────────────────────────────────
def estimate(category, review_count, price_pct_raw, actual_price):
    row      = cat_stats[cat_stats["category"] == category]
    log_comp = float(row["log_competition"].values[0]) if len(row)>0 \
               else float(cat_stats["log_competition"].median())
    log_pl   = np.log1p(float(actual_price)) if actual_price>0 \
               else (float(row["log_price_level"].values[0]) if len(row)>0
                     else float(cat_stats["log_price_level"].median()))

    X_q    = np.array([[log_pl, log_comp]])
    cate   = float(model.effect(X_q)[0])
    lb, ub = model.effect_interval(X_q, alpha=0.05)
    lb, ub = float(lb[0]), float(ub[0])
    elast  = cate * (T_STD / (Y_STD + 1e-6))
    rv     = int(review_count)
    dr     = np.expm1(np.log1p(rv) + cate) - rv
    pct_r  = dr / rv * 100 if rv > 0 else 0

    return dict(cate=cate, lb=lb, ub=ub, elast=elast,
                pct_rank=(all_cate < cate).mean()*100,
                delta_reviews=dr, pct_change_rev=pct_r,
                log_pl=log_pl, log_comp=log_comp,
                sig=lb>0 or ub<0)


# ── Metric cards HTML ─────────────────────────────────────────────────────
def build_cards(est, category, review_count):
    cate  = est["cate"]
    elast = est["elast"]
    dr    = est["delta_reviews"]
    pct_r = est["pct_change_rev"]
    rank  = est["pct_rank"]
    sig   = est["sig"]
    lb, ub = est["lb"], est["ub"]

    cc = C_POS if cate>0 else C_NEG
    ec = C_POS if elast>0 else C_NEG
    dc = C_POS if dr>0 else C_NEG
    sig_txt = "✓ significant (95%)" if sig else "✗ not significant"
    sig_col = C_POS if sig else C_GRAY

    if   cate >  0.05: story = "Premium pricing <strong>signals quality</strong> — customers trust high prices and engage more."
    elif cate >  0.01: story = "Mild positive — slightly higher engagement at premium prices."
    elif cate < -0.05: story = "Premium pricing <strong>raises the bar without delivering</strong> — suppresses engagement."
    elif cate < -0.01: story = "Mild negative — premium prices slightly suppress review activity."
    else:              story = "Price has <strong>negligible causal impact</strong> here — other factors dominate."

    return f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px">
  <div style="background:var(--color-background-secondary);border-radius:10px;
       padding:14px 16px;border-top:3px solid {cc};transition:all .3s">
    <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;
         letter-spacing:.8px;margin-bottom:4px">Causal Effect</div>
    <div style="font-size:30px;font-weight:700;color:{cc};
         font-variant-numeric:tabular-nums;letter-spacing:-1px">{cate:+.4f}</div>
    <div style="font-size:10px;color:var(--color-text-secondary);margin-top:2px">log-review units</div>
    <div style="font-size:11px;color:{sig_col};margin-top:6px">{sig_txt}</div>
  </div>
  <div style="background:var(--color-background-secondary);border-radius:10px;
       padding:14px 16px;border-top:3px solid {ec}">
    <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;
         letter-spacing:.8px;margin-bottom:4px">Price Elasticity</div>
    <div style="font-size:30px;font-weight:700;color:{ec};
         font-variant-numeric:tabular-nums;letter-spacing:-1px">{elast:+.3f}</div>
    <div style="font-size:10px;color:var(--color-text-secondary);margin-top:2px">Δ log-reviews per Δ price-pct</div>
    <div style="font-size:11px;color:var(--color-text-secondary);margin-top:6px">
      CI [{lb*(T_STD/(Y_STD+1e-6)):+.3f}, {ub*(T_STD/(Y_STD+1e-6)):+.3f}]
    </div>
  </div>
  <div style="background:var(--color-background-secondary);border-radius:10px;
       padding:14px 16px;border-top:3px solid {dc}">
    <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;
         letter-spacing:.8px;margin-bottom:4px">Review Impact</div>
    <div style="font-size:30px;font-weight:700;color:{dc};
         font-variant-numeric:tabular-nums;letter-spacing:-1px">{dr:+,.0f}</div>
    <div style="font-size:10px;color:var(--color-text-secondary);margin-top:2px">
      reviews vs median-priced peer
    </div>
    <div style="font-size:11px;color:{dc};margin-top:6px">{pct_r:+.1f}% volume change</div>
  </div>
  <div style="background:var(--color-background-secondary);border-radius:10px;
       padding:14px 16px;border-top:3px solid {C_ATE}">
    <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;
         letter-spacing:.8px;margin-bottom:4px">Product Percentile</div>
    <div style="font-size:30px;font-weight:700;color:var(--color-text-primary);
         font-variant-numeric:tabular-nums;letter-spacing:-1px">
      {rank:.0f}<span style="font-size:14px;font-weight:400">th</span>
    </div>
    <div style="font-size:10px;color:var(--color-text-secondary);margin-top:2px">
      stronger effect than {rank:.0f}% of products
    </div>
    <div style="font-size:11px;color:var(--color-text-secondary);margin-top:6px">
      Rosenbaum Γ = {GAMMA:.2f}
    </div>
  </div>
</div>
<div style="background:var(--color-background-secondary);border-radius:10px;
     padding:12px 16px;border-left:4px solid {cc};margin-bottom:2px">
  <span style="font-size:13px;color:var(--color-text-primary)">{story}</span>
  <span style="font-size:12px;color:var(--color-text-secondary);display:block;margin-top:5px">
    Moving <strong>50th → 100th price percentile</strong> in <em>{category}</em>
    {'↑ increases' if dr>0 else '↓ decreases'} expected reviews by
    <strong style="color:{dc}">{abs(pct_r):.1f}%</strong>
    ({dr:+,.0f} reviews for a product with {int(review_count):,} current reviews).
  </span>
</div>"""


# ── Plotly: CATE curve with bootstrap CI ─────────────────────────────────
def fig_cate_curve(est, category):
    grid   = boot["grid"]
    curves = boot["curves"]
    lc     = est["log_comp"]
    lpl    = est["log_pl"]
    cate   = est["cate"]

    X_curve = np.column_stack([grid, np.full(len(grid), lc)])
    effects = model.effect(X_curve)
    shift   = effects - curves.mean(axis=0)
    blo     = np.percentile(curves, 5,  axis=0) + shift
    bhi     = np.percentile(curves, 95, axis=0) + shift

    # Natural price labels
    price_vals = [5, 15, 50, 150, 500, 1500]
    price_logs = [np.log1p(p) for p in price_vals]

    fig = go.Figure()

    # Bootstrap CI band
    fig.add_trace(go.Scatter(
        x=np.concatenate([grid, grid[::-1]]),
        y=np.concatenate([bhi, blo[::-1]]),
        fill="toself",
        fillcolor="rgba(83,74,183,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Bootstrap 90% CI",
        hoverinfo="skip",
    ))

    # Main CATE line
    fig.add_trace(go.Scatter(
        x=grid, y=effects,
        mode="lines",
        line=dict(color=C_ATE, width=2.5),
        name="CATE estimate",
        hovertemplate="log(price)=%{x:.2f}<br>CATE=%{y:.4f}<extra></extra>",
    ))

    # Zero line
    fig.add_hline(y=0, line=dict(color=C_GRAY, width=1, dash="dash"))
    # ATE line
    fig.add_hline(y=ATE, line=dict(color=C_ATE, width=1, dash="dot"),
                  annotation_text=f"Population ATE={ATE:.4f}",
                  annotation_position="bottom right",
                  annotation_font_color=C_ATE)

    # Your product point
    fig.add_trace(go.Scatter(
        x=[lpl], y=[cate],
        mode="markers+text",
        marker=dict(color=C_NEG, size=12, symbol="circle",
                    line=dict(color="white", width=2)),
        text=[f"  {cate:+.4f}"],
        textposition="middle right",
        textfont=dict(color=C_NEG, size=11),
        name="Your product",
        hovertemplate=f"Your product<br>CATE={cate:+.4f}<extra></extra>",
    ))

    # Vertical line for your product
    fig.add_vline(x=lpl, line=dict(color=C_NEG, width=1.5, dash="dot"))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=f"Price sensitivity curve — {textwrap.shorten(category, 35)}<br>"
                        f"<sup>Shaded = bootstrap 90% CI across 200 resampled models</sup>",
                   font_size=13),
        xaxis=dict(
            title="Log(1 + price)",
            tickvals=price_logs,
            ticktext=[f"${p}" for p in price_vals],
            tickfont_size=9,
        ),
        yaxis=dict(title="Causal effect on log(reviews)"),
        legend=dict(orientation="h", y=-0.15, font_size=10),
        margin=dict(l=50, r=20, t=70, b=60),
        height=340,
        hovermode="x unified",
    )
    return fig


# ── Plotly: elasticity distribution ──────────────────────────────────────
def fig_elast_dist(est):
    elast = est["elast"]
    ate_e = ATE * (T_STD / (Y_STD + 1e-6))

    # Histogram with colour split at 0
    neg_vals = all_elast[all_elast < 0]
    pos_vals = all_elast[all_elast >= 0]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=neg_vals, nbinsx=50,
        marker_color=C_NEG, opacity=0.7,
        name=f"Price hurts ({(all_elast<0).mean():.0%})",
        hovertemplate="elasticity=%{x:.3f}<br>count=%{y}<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=pos_vals, nbinsx=50,
        marker_color=C_POS, opacity=0.7,
        name=f"Price helps ({(all_elast>=0).mean():.0%})",
        hovertemplate="elasticity=%{x:.3f}<br>count=%{y}<extra></extra>",
    ))

    # Reference lines
    fig.add_vline(x=0,     line=dict(color=C_GRAY, width=1, dash="dash"))
    fig.add_vline(x=ate_e, line=dict(color=C_ATE,  width=1.5, dash="dot"),
                  annotation_text=f"ATE={ate_e:.3f}",
                  annotation_position="top",
                  annotation_font_color=C_ATE)
    fig.add_vline(x=elast, line=dict(color="black", width=2.5),
                  annotation_text=f"Yours={elast:+.3f}",
                  annotation_position="top right",
                  annotation_font_color="black")

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=f"Elasticity distribution — {len(all_elast):,} products<br>"
                        f"<sup>Where does your product sit?</sup>",
                   font_size=13),
        xaxis_title="Price elasticity of review volume",
        yaxis_title="Number of products",
        barmode="overlay",
        legend=dict(orientation="h", y=-0.18, font_size=10),
        margin=dict(l=50, r=20, t=70, b=60),
        height=340,
    )
    return fig


# ── Plotly: competitor scatter ────────────────────────────────────────────
def fig_competitor(est, category, price_pct_raw):
    cat_df = df_feat[df_feat["category"] == category].copy()
    if len(cat_df) < 10:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for this category",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font_size=14, font_color=C_GRAY)
        fig.update_layout(height=340, template=PLOTLY_TEMPLATE)
        return fig

    fig = go.Figure()

    # All competitors — colour by review count
    fig.add_trace(go.Scatter(
        x=cat_df["price_pct"],
        y=cat_df["elasticity"],
        mode="markers",
        marker=dict(
            color=np.log1p(cat_df["review_count"]),
            colorscale="Viridis",
            size=6, opacity=0.4,
            colorbar=dict(title="Log(reviews)", thickness=12, len=0.7),
        ),
        name="Competitors",
        text=cat_df.get("asin", ""),
        hovertemplate="price_pct=%{x:.2f}<br>elasticity=%{y:.3f}<extra></extra>",
    ))

    # Your product
    fig.add_trace(go.Scatter(
        x=[price_pct_raw / 100],
        y=[est["elast"]],
        mode="markers+text",
        marker=dict(color=C_NEG, size=18, symbol="star",
                    line=dict(color="white", width=2)),
        text=["  ← Your product"],
        textposition="middle right",
        textfont=dict(color=C_NEG, size=11, family="monospace"),
        name=f"Your product ({est['elast']:+.3f})",
        hovertemplate=f"Your product<br>elasticity={est['elast']:+.3f}<extra></extra>",
    ))

    ate_e = ATE * (T_STD / (Y_STD + 1e-6))
    fig.add_hline(y=0,     line=dict(color=C_GRAY, width=1, dash="dash"))
    fig.add_hline(y=ate_e, line=dict(color=C_ATE,  width=1, dash="dot"),
                  annotation_text=f"Dataset ATE={ate_e:.3f}",
                  annotation_position="bottom right",
                  annotation_font_color=C_ATE)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=f"Competitor map — {textwrap.shorten(category, 32)}<br>"
                        f"<sup>⭐ = your product  |  colour = review volume</sup>",
                   font_size=13),
        xaxis=dict(title="Price percentile in category",
                   tickformat=".0%", range=[-0.05, 1.05]),
        yaxis_title="Price elasticity of review volume",
        legend=dict(orientation="h", y=-0.18, font_size=10),
        margin=dict(l=50, r=20, t=70, b=60),
        height=340,
        hovermode="closest",
    )
    return fig


# ── Plotly: category bar ──────────────────────────────────────────────────
def fig_category_bar(highlight=None):
    top5 = cat_summary.tail(5)
    bot5 = cat_summary.head(5)
    dfp  = pd.concat([bot5, top5]).reset_index(drop=True)

    colors = []
    for _, r in dfp.iterrows():
        if highlight and r["category"] == highlight:
            colors.append(C_YEL)
        else:
            colors.append(C_POS if r["avg_elasticity"] >= 0 else C_NEG)

    labels = [textwrap.fill(c, 28) for c in dfp["category"]]

    fig = go.Figure(go.Bar(
        x=dfp["avg_elasticity"],
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in dfp["avg_elasticity"]],
        textposition="outside",
        textfont_size=9,
        hovertemplate="%{y}<br>elasticity=%{x:.4f}<extra></extra>",
    ))

    ate_e = ATE * (T_STD / (Y_STD + 1e-6))
    fig.add_vline(x=0,     line=dict(color=C_GRAY, width=1, dash="dash"))
    fig.add_vline(x=ate_e, line=dict(color=C_ATE,  width=1.5, dash="dot"),
                  annotation_text=f"ATE={ate_e:.3f}",
                  annotation_position="top",
                  annotation_font_color=C_ATE)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Price elasticity by category — top 5 vs bottom 5<br>"
                        "<sup>Yellow = your selected category</sup>",
                   font_size=13),
        xaxis_title="Avg price elasticity of review volume",
        margin=dict(l=200, r=60, t=70, b=40),
        height=380,
        showlegend=False,
    )
    return fig


# ── Plotly: CATE heatmap (price level × competition) ─────────────────────
def fig_cate_heatmap(est):
    n = 40
    lpl_range = np.linspace(
        df_feat["log_price_level"].quantile(0.05),
        df_feat["log_price_level"].quantile(0.95), n
    )
    lc_range = np.linspace(
        df_feat["log_competition"].quantile(0.05),
        df_feat["log_competition"].quantile(0.95), n
    )
    grid_pl, grid_lc = np.meshgrid(lpl_range, lc_range)
    X_grid = np.column_stack([grid_pl.ravel(), grid_lc.ravel()])
    Z = model.effect(X_grid).reshape(n, n)
    Z = np.clip(Z, cate_p2, cate_p98)

    price_ticks = [5, 20, 100, 500, 2000]
    comp_ticks  = [50, 200, 500, 1000, 3000]

    fig = go.Figure(go.Heatmap(
        z=Z,
        x=lpl_range,
        y=lc_range,
        colorscale=[
            [0.0, C_NEG], [0.5, "#F5F5F0"], [1.0, C_POS]
        ],
        zmid=0,
        colorbar=dict(title="CATE", thickness=14),
        hovertemplate="log(price)=%{x:.2f}<br>log(competition)=%{y:.2f}<br>CATE=%{z:.4f}<extra></extra>",
    ))

    # Mark your product
    fig.add_trace(go.Scatter(
        x=[est["log_pl"]],
        y=[est["log_comp"]],
        mode="markers",
        marker=dict(color="black", size=14, symbol="star",
                    line=dict(color="white", width=2)),
        name="Your product",
        hovertemplate=f"Your product<br>CATE={est['cate']:+.4f}<extra></extra>",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="CATE heatmap — price level × market competition<br>"
                        "<sup>Green=price helps · Red=price hurts · ⭐=your product</sup>",
                   font_size=13),
        xaxis=dict(
            title="Price level",
            tickvals=[np.log1p(p) for p in price_ticks],
            ticktext=[f"${p}" for p in price_ticks],
        ),
        yaxis=dict(
            title="Market competition",
            tickvals=[np.log1p(c) for c in comp_ticks],
            ticktext=[f"{c} products" for c in comp_ticks],
        ),
        margin=dict(l=100, r=20, t=70, b=60),
        height=340,
    )
    return fig


# ── Matplotlib: refutation panel (static — only computed once) ────────────
def _make_refutation_fig():
    effs   = [parse_refute(results["refute_placebo"],"New effect"),
              parse_refute(results["refute_random"], "New effect"),
              parse_refute(results["refute_subset"], "New effect")]
    pvals  = [parse_refute(results["refute_placebo"],"p value"),
              parse_refute(results["refute_random"], "p value"),
              parse_refute(results["refute_subset"], "p value")]
    labels = ["Placebo\n(expect ~0)",
              "Random cause\n(expect ≈ ATE)",
              "Data subset 80%\n(expect ≈ ATE)"]
    tgts   = [0.0, ATE, ATE]
    cols   = []
    for i, (e, t) in enumerate(zip(effs, tgts)):
        if i == 0:   # placebo — good if near zero
            cols.append(C_POS if abs(e) < abs(ATE) * 0.3 else C_NEG)
        else:        # random/subset — good if near ATE
            cols.append(C_POS if abs(e - t) < abs(ATE) * 0.5 else C_NEG)
            
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,
                         "figure.facecolor":"white","axes.facecolor":"white"})
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh(labels, effs, color=cols, height=0.45)
    dowhy_ate = parse_refute(results["refute_placebo"], "Estimated effect")
    ref_label = dowhy_ate if not (isinstance(dowhy_ate, float) and np.isnan(dowhy_ate)) else ATE
    ax.axvline(ref_label, color=C_ATE, linewidth=1.5, linestyle="--",
               label=f"DoWhy estimate={ref_label:.4f}")
    ax.axvline(0,   color=C_GRAY, linewidth=0.8, linestyle=":")
    for i,(e,p) in enumerate(zip(effs,pvals)):
        if not (isinstance(e, float) and np.isnan(e)):
            ax.text(max(abs(e),abs(ATE))*0.05+e, i, f"  p={p:.2f}",
                    va="center", fontsize=8.5)
    ax.set_title(f"3 refutation tests  |  Rosenbaum Γ={GAMMA:.2f}\n"
                 f"Γ>1.5 = robust to moderate unmeasured confounding", fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig

REFUTE_FIG = _make_refutation_fig()


# ── Main callback ─────────────────────────────────────────────────────────
def on_change(category, review_count, price_pct_raw, actual_price):
    est       = estimate(category, review_count, price_pct_raw, actual_price)
    cards     = build_cards(est, category, review_count)
    f_curve   = fig_cate_curve(est, category)
    f_dist    = fig_elast_dist(est)
    f_comp    = fig_competitor(est, category, price_pct_raw)
    f_cat     = fig_category_bar(highlight=category)
    f_heat    = fig_cate_heatmap(est)
    return cards, f_curve, f_dist, f_comp, f_cat, f_heat


# ── Header HTML ───────────────────────────────────────────────────────────
ate_e    = ATE * (T_STD / (Y_STD + 1e-6))
el_min   = all_elast.min()
el_max   = all_elast.max()
pct_neg  = (all_elast < 0).mean()

HEADER_HTML = f"""
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;padding:4px 0 16px">
  <div style="background:var(--color-background-secondary);border-radius:10px;padding:14px 16px">
    <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;
         letter-spacing:.8px;margin-bottom:4px">Products Analysed</div>
    <div style="font-size:26px;font-weight:700;color:var(--color-text-primary)">{len(df_feat):,}</div>
    <div style="font-size:10px;color:var(--color-text-secondary);margin-top:3px">Amazon Sep 2023 · 62 categories</div>
  </div>
  <div style="background:var(--color-background-secondary);border-radius:10px;padding:14px 16px;
       border-top:3px solid {'#1D9E75' if ATE>0 else '#D85A30'}">
    <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;
         letter-spacing:.8px;margin-bottom:4px">Population ATE {sig_star}</div>
    <div style="font-size:26px;font-weight:700;color:{'#1D9E75' if ATE>0 else '#D85A30'}">{ATE:+.4f}</div>
    <div style="font-size:10px;color:var(--color-text-secondary);margin-top:3px">
      log-reviews · CI [{ATE_LB:.4f}, {ATE_UB:.4f}]
    </div>
  </div>
  <div style="background:var(--color-background-secondary);border-radius:10px;padding:14px 16px">
    <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;
         letter-spacing:.8px;margin-bottom:4px">Elasticity Range</div>
    <div style="font-size:22px;font-weight:700;color:var(--color-text-primary)">{el_min:.2f} → {el_max:.2f}</div>
    <div style="font-size:10px;color:var(--color-text-secondary);margin-top:3px">massive heterogeneity hidden in ATE</div>
  </div>
  <div style="background:var(--color-background-secondary);border-radius:10px;padding:14px 16px;
       border-top:3px solid #D85A30">
    <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;
         letter-spacing:.8px;margin-bottom:4px">Price Hurts Engagement</div>
    <div style="font-size:26px;font-weight:700;color:#D85A30">{pct_neg:.0%}</div>
    <div style="font-size:10px;color:var(--color-text-secondary);margin-top:3px">of all {len(df_feat):,} products</div>
  </div>
  <div style="background:var(--color-background-secondary);border-radius:10px;padding:14px 16px">
    <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;
         letter-spacing:.8px;margin-bottom:4px">Robustness (Γ)</div>
    <div style="font-size:26px;font-weight:700;color:var(--color-text-primary)">{GAMMA:.2f}</div>
    <div style="font-size:10px;color:var(--color-text-secondary);margin-top:3px">Rosenbaum sensitivity bound</div>
  </div>
</div>
<div style="background:var(--color-background-secondary);border-radius:10px;padding:14px 18px;
     border-left:4px solid {C_ATE};margin-bottom:4px">
  <div style="font-size:13px;font-weight:600;color:var(--color-text-primary);margin-bottom:5px">
    Finding: premium pricing causally suppresses engagement for most products — but heterogeneity is large
  </div>
  <div style="font-size:12px;color:var(--color-text-secondary);line-height:1.7">
    Outcome: <code>log(1 + review_count)</code> — proxy for purchase volume and satisfaction engagement.
    Population ATE = <strong>{ATE:+.4f}</strong> ({sig_star}, p={PVAL:.4f}).
    Individual elasticities: <strong style="color:{C_NEG}">{el_min:.2f}</strong> to
    <strong style="color:{C_POS}">{el_max:.2f}</strong>.
    Premium pricing <em>helps</em> in experience goods (headphones, gaming) and
    <em>hurts</em> in commodity categories — consistent with
    <strong>information asymmetry theory</strong> (Akerlof, 1970).
  </div>
</div>
"""


# ── Gradio layout ─────────────────────────────────────────────────────────
with gr.Blocks(title="CausalPrice — Causal Inference Dashboard") as demo:

    gr.Markdown("# CausalPrice\n### Causal effect of price on customer engagement — Amazon 2023")
    gr.HTML(HEADER_HTML)

    gr.Markdown("---")
    gr.Markdown(
        "### Estimate causal effect for your product\n"
        "*All 5 charts update instantly as you move the sliders — "
        "hover over any chart for exact values.*"
    )

    with gr.Row():
        with gr.Column(scale=1):
            category     = gr.Dropdown(choices=CATEGORIES, value=CATEGORIES[0],
                                       label="Product category")
            review_count = gr.Slider(10, 50000, value=500, step=10,
                                     label="Current review count",
                                     info="Used to calculate absolute review impact")
            price_pct    = gr.Slider(0, 100, value=50, step=1,
                                     label="Price percentile in category (%)",
                                     info="50=median · 90=top 10% most expensive")
            actual_price = gr.Slider(1, 3000, value=50, step=1,
                                     label="Actual price ($)",
                                     info="Used as a key effect modifier")

            gr.Markdown("**Try these examples:**")
            gr.Examples(
                examples=[
                    ["Headphones & Earbuds",          8000, 80, 120],
                    ["Smart Home: Security Cameras and Systems", 150, 85, 200],
                    ["Video Games",                   2000, 55, 60 ],
                    ["Computers",                     500,  70, 800],
                    ["Skin Care Products",             300,  45, 35 ],
                    ["Industrial Adhesives, Sealants & Lubricants", 80, 60, 25],
                ],
                inputs=[category, review_count, price_pct, actual_price],
                examples_per_page=6,
            )

        with gr.Column(scale=2):
            result_cards = gr.HTML()

    # Row 1: curve + competitor
    with gr.Row():
        fig_curve = gr.Plot(label="Price sensitivity curve + bootstrap CI")
        fig_comp  = gr.Plot(label="Competitor map in your category")

    # Row 2: distribution + category bar
    with gr.Row():
        fig_dist  = gr.Plot(label="Elasticity distribution — your product highlighted")
        fig_cat   = gr.Plot(label="Category ranking — your category highlighted")

    # Row 3: CATE heatmap (full width)
    with gr.Row():
        fig_heat  = gr.Plot(label="CATE heatmap — price level × market competition")

    # Validation section
    gr.Markdown("---")
    gr.Markdown(
        f"### Model validation\n"
        f"**Outcome:** `log(1+review_count)` — continuous, std≈1.8, 6× more signal than binary outcome.  \n"
        f"**Refutations:** 3 tests — placebo, random cause, data subset.  \n"
        f"**Holdout:** validated on {results['holdout']['n_holdout']:,} products across "
        f"{len(results['holdout']['holdout_categories'])} unseen categories.  \n"
        f"**Rosenbaum Γ={GAMMA:.2f}** — confounder would need odds ratio >{GAMMA:.2f} to flip result."
    )
    refute_plot = gr.Plot(value=REFUTE_FIG, label="Refutation tests")

    gr.Markdown(f"""
---
**Technical:**
`CausalForestDML` (EconML) · `GradientBoostingRegressor` nuisance models · 5-fold cross-fitting ·
600 trees · 7 confounders · bootstrap CI from 200 resampled models ·
trained on {results['n_categories']} categories.

**Why log(reviews) as outcome?**
Binary P(rating≥4.0) has 90% baseline → Bernoulli variance=0.09.
log(review_count) has std≈1.8 → 6× more signal.
Review volume is a published proxy for sales (Chevalier & Mayzlin 2006).

Data: [Amazon Products 2023](https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023-1-4m-products) ·
Method: [EconML](https://econml.azurewebsites.net/) · [DoWhy](https://py-why.github.io/dowhy/)
""")

    # Wire up all inputs to all outputs
    inputs  = [category, review_count, price_pct, actual_price]
    outputs = [result_cards, fig_curve, fig_dist, fig_comp, fig_cat, fig_heat]

    for inp in inputs:
        inp.change(fn=on_change, inputs=inputs, outputs=outputs)

    demo.load(
        fn=lambda: on_change(CATEGORIES[0], 500, 50, 50),
        outputs=outputs,
    )

if __name__ == "__main__":
    demo.launch()