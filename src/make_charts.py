"""Slide charts from results/ (owner: M5). Run after train_price_models:
    python -m src.make_charts
Writes PNGs to results/charts/. Every chart's numbers are in the matching results/*.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("results")
OUT = R / "charts"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
FINAL = "v2"   # results version of the final model (finalize_model.py)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
})


def finish(fig, ax, name, title, subtitle):
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=INK2, va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


def model_comparison():
    t = pd.read_csv(R / "model_comparison_v1.csv").sort_values("cv_MAPE_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    is_bench = t.model.str.startswith("Benchmark")
    labels = t.model.str.replace("Benchmark (median by model line + year)", "Benchmark: median price\nfor model line + year", regex=False)
    bars = ax.barh(labels, t.cv_MAPE_pct, height=0.6, color=np.where(is_bench, GRAY, BLUE))
    for b, v in zip(bars, t.cv_MAPE_pct):
        ax.text(v + 0.1, b.get_y() + b.get_height() / 2, f"{v:.2f}%", va="center", fontsize=10, color=INK)
    ax.set_xlabel("Cross-validated MAPE (%), lower is better")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, t.cv_MAPE_pct.max() * 1.15)
    finish(fig, ax, "01_model_comparison.png", "Every tuned model beats the benchmark",
           "5-fold CV on 12,123 training listings; same folds for every model")


def error_by_band():
    b = pd.read_csv(R / f"error_by_price_band_{FINAL}.csv")
    w, bm = b[b.set == "cv_winner"], b[b.set == "cv_benchmark"]
    x = np.arange(len(w))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for off, d, c, lab in [(-0.17, w, BLUE, "Chosen model"), (0.17, bm, ORANGE, "Benchmark")]:
        bars = ax.bar(x + off, d.MAPE_pct, width=0.32, color=c, label=lab)
        for bar, v in zip(bars, d.MAPE_pct):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [f"{s}\n(n={int(n):,})" for s, n in zip(w.band, w.n)])
    ax.set_ylabel("CV MAPE (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    ax.set_ylim(0, max(w.MAPE_pct.max(), bm.MAPE_pct.max()) * 1.2)
    finish(fig, ax, "02_error_by_price_band.png", "Cheap cars are hardest to price in % terms",
           "Out-of-fold error by actual listing price band, training listings")


def q2_drivers():
    q = pd.read_csv(R / "q2_pct_effects_v1.csv")
    a = q[q.spec.str.startswith("A")].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    err = [a.pct_effect - a.ci_low_pct, a.ci_high_pct - a.pct_effect]
    ax.barh(a.effect, a.pct_effect, height=0.5, color=BLUE, xerr=err,
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
    for y, (v, lo_ci, hi_ci) in enumerate(zip(a.pct_effect, a.ci_low_pct, a.ci_high_pct)):
        ax.text(lo_ci - 0.25 if v < 0 else hi_ci + 0.25, y, f"{v:+.1f}%", va="center",
                ha="right" if v < 0 else "left", fontsize=10, color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlabel("Change in price (%), same trim, other factors held fixed")
    ax.grid(axis="y", visible=False)
    lo = min(a.ci_low_pct.min(), 0) * 1.35
    ax.set_xlim(lo, max(a.ci_high_pct.max(), 0) + abs(lo) * 0.25)
    finish(fig, ax, "03_q2_value_drivers.png", "What moves a used EV's price",
           "OLS on log price with trim fixed effects; bars show 95% robust CI")


def q2_brands(min_n=30):
    q = pd.read_csv(R / "q2_pct_effects_v1.csv")
    b = q[q.spec.str.startswith("B") & (q.n_listings >= min_n)].sort_values("pct_effect")
    b = b.assign(brand=b.effect.str.replace("Brand: ", "").str.replace(" vs Nissan", ""))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(b.brand, b.pct_effect, height=0.6, color=BLUE)
    for y, (v, n) in enumerate(zip(b.pct_effect, b.n_listings)):
        ax.text(v + 4 if v >= 0 else v - 4, y, f"{v:+.0f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlabel("Price premium vs a Nissan of the same age, mileage and history (%)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(b.pct_effect.min() * 1.3, -10), b.pct_effect.max() * 1.15)
    finish(fig, ax, "04_q2_brand_premiums.png", "Brand premiums include model mix",
           f"OLS on log price; brands with at least {min_n} training listings")


def ablation():
    """Gain in CV MAPE from each feature block: condition (M1's and M5's rules) and complaints."""
    c = pd.read_csv(R / "condition_test_v1.csv")
    a = pd.read_csv(R / "feature_ablation_v1.csv")
    models = ["Ridge", "XGBoost"]
    def pick(df, variant, col="MAPE_gain_pts", folds="folds_helped_MAPE"):
        d = df[df.variant == variant].set_index("model").loc[models]
        return d[col].to_numpy(), d[folds].to_numpy()
    series = [("Condition features, M1's rules", *pick(c, "M1 rules (all 7)"), BLUE),
              ("Condition features, M5's rules", *pick(c, "M5 rules"), ORANGE),
              ("Complaint features (M4)", *pick(a, "gain from complaints (pts)", "cv_MAPE_pct", "folds_improved"), AQUA)]
    x = np.arange(len(models))
    w = 0.24
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (lab, vals, folds, col) in enumerate(series):
        bars = ax.bar(x + (i - 1) * (w + 0.02), vals, width=w, color=col, label=lab)
        for bar, v, k in zip(bars, vals, folds):
            ax.text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 0.008,
                    f"{v:+.2f}\n{int(k)}/5", ha="center", fontsize=9, color=INK)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(x, models)
    ax.set_ylabel("MAPE improvement (percentage points)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(-0.06, 0.33)
    finish(fig, ax, "05_feature_ablation.png", "Feature blocks help the linear model, barely the final one",
           "CV MAPE without the block minus with it; labels show the gain and folds improved (of 5)")


def predicted_vs_actual():
    p = pd.read_csv(R / f"test_predictions_winner_{FINAL}.csv")
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.scatter(p.price, p.predicted, s=10, color=BLUE, alpha=0.35, linewidths=0)
    lim = [p[["price", "predicted"]].min().min() * 0.9, p[["price", "predicted"]].max().max() * 1.1]
    ax.plot(lim, lim, color=INK2, linewidth=1)
    xs = np.array(lim)
    ax.fill_between(xs, xs * 0.9, xs * 1.1, color=BLUE, alpha=0.08, linewidth=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ticks = [5000, 10000, 20000, 50000, 100000, 200000]
    ax.set_xticks(ticks, [f"${t//1000}K" for t in ticks]); ax.set_yticks(ticks, [f"${t//1000}K" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual listing price"); ax.set_ylabel("Predicted fair price")
    finish(fig, ax, "06_predicted_vs_actual_test.png", "Test set: predicted vs actual",
           "Final model, 3,031 held-out listings; shaded band is within ±10%")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in [model_comparison, error_by_band, q2_drivers, q2_brands, ablation, predicted_vs_actual]:
        f()
    print("charts in", OUT)
