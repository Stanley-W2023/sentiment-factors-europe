"""
plots.py — All visualisations for the RSF research pipeline.

Generates publication-quality figures and saves them to plots/.
Call generateAllPlots() from main.py with GENERATE_PLOTS = True.

Plots produced:
  1.  return_heatmap_vw.png     — 5×5 mean excess return heatmap (VW)
  2.  return_heatmap_ew.png     — 5×5 mean excess return heatmap (EW)
  3.  sat_spread_ladder.png     — delta_SAT by PE quintile with NW error bars
  4.  pe_spread_ladder.png      — delta_PE by SAT quintile with NW error bars
  5.  cumulative_ls.png         — Cumulative RSF long-short return
  6.  rolling_sharpe.png        — 36-month rolling Sharpe of L/S
  7.  rolling_delta_cross.png   — 36-month rolling delta_Cross over time
  8.  alpha_heatmap.png         — 5×5 FF6 alpha heatmap
  9.  fmb_coefficients.png      — FMB lambda estimates with t-stat annotations
  10. smallmidcap_comparison.png— Full sample vs small/mid-cap spread comparison
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for server/headless runs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from config.constants import (
    PE_QUINTILE_COL,
    SAT_QUINTILE_COL,
    REGIME_BREAK_DATE,
    N_QUINTILES,
)

_PLOTS_DIR = "plots"
_FIGSIZE_SQUARE  = (7, 6)
_FIGSIZE_WIDE    = (10, 5)
_FIGSIZE_TALL    = (8, 7)
_DPI             = 150
_REGIME_COLOR    = "#d62728"
_PALETTE_MAIN    = "#1f77b4"
_PALETTE_ROBUST  = "#ff7f0e"


def generateAllPlots(
    portfolioReturns:      pd.DataFrame,
    portfolioReturnsEW:    pd.DataFrame,
    spreads:               pd.DataFrame,
    spreadsSmallMid:       pd.DataFrame,
    rollingDeltaCross:     pd.DataFrame,
    rsfLongShort:          pd.DataFrame,
    alphaResults:          pd.DataFrame,
    fmbResults:            pd.DataFrame,
    parametricResults:     dict | None = None,
    dailySpreads:          pd.DataFrame | None = None,
    dailyFMBResults:       pd.DataFrame | None = None,
    dailyRollingDC:        pd.DataFrame | None = None,
    outDir:                str = _PLOTS_DIR,
):
    """
    Generate and save all RSF visualisations.

    Args:
        portfolioReturns   : VW 5×25 portfolio returns from constructFF25Portfolios.
        portfolioReturnsEW : EW 5×25 portfolio returns from constructFF25PortfoliosEW.
        spreads            : Full-sample spreads with NW t-stats from computeSpreads.
        spreadsSmallMid    : Small/mid-cap spreads from computeSpreads.
        rollingDeltaCross  : Rolling delta_Cross from computeRollingDeltaCross.
        rsfLongShort       : RSF L/S monthly returns from buildRSFLongShort.
        alphaResults       : FF6 alphas from estimateTimeSeriesAlpha.
        fmbResults         : FMB coefficients from estimateFamaMacBeth.
        outDir             : Directory to save plots (created if missing).
    """
    os.makedirs(outDir, exist_ok=True)
    warnings.filterwarnings("ignore")

    print(f"\n  Generating plots → {outDir}/")

    _plotReturnHeatmap(portfolioReturns,   outDir, suffix="vw",
                       title="Mean Monthly Excess Return — Value Weighted")
    _plotReturnHeatmap(portfolioReturnsEW, outDir, suffix="ew",
                       title="Mean Monthly Excess Return — Equal Weighted",
                       ret_col="ret_ew")
    _plotSpreadLadder(spreads, spreadsSmallMid, outDir)
    _plotCumulativeLS(rsfLongShort, outDir)
    _plotRollingSharpe(rsfLongShort, outDir)
    _plotRollingDeltaCross(rollingDeltaCross, outDir)
    _plotAlphaHeatmap(alphaResults, outDir)
    _plotFMBCoefficients(fmbResults, outDir)
    _plotSmallMidCapComparison(spreads, spreadsSmallMid, outDir)

    if parametricResults is not None:
        _plotParametricSATAmplification(parametricResults, outDir)

    if dailySpreads is not None:
        _plotDailySpreadComparison(dailySpreads, outDir)

    if dailyFMBResults is not None:
        _plotDailyFMBComparison(dailyFMBResults, outDir)

    if dailyRollingDC is not None:
        _plotDailyRollingDeltaCross(dailyRollingDC, outDir)

    print(f"  Done — {len(os.listdir(outDir))} plots saved.")


# ── 1 & 2. Return heatmaps ────────────────────────────────────────────────────

def _plotReturnHeatmap(
    portfolioReturns: pd.DataFrame,
    outDir: str,
    suffix: str,
    title: str,
    ret_col: str = "excess_ret",
):
    mean_ret = (
        portfolioReturns
        .groupby([PE_QUINTILE_COL, SAT_QUINTILE_COL])[ret_col]
        .mean()
        .unstack(SAT_QUINTILE_COL)
        * 100  # convert to percent
    )

    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)
    vmax = max(abs(mean_ret.values.max()), abs(mean_ret.values.min()))
    im = ax.imshow(
        mean_ret.values,
        cmap="RdYlGn",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="Mean Monthly Excess Return (%)")

    ax.set_xticks(range(N_QUINTILES))
    ax.set_xticklabels([f"SAT Q{i}" for i in range(1, N_QUINTILES + 1)])
    ax.set_yticks(range(N_QUINTILES))
    ax.set_yticklabels([f"PE Q{i}" for i in range(1, N_QUINTILES + 1)])

    for i in range(N_QUINTILES):
        for j in range(N_QUINTILES):
            val = mean_ret.values[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("SAT Quintile (Q1 = Most Bearish → Q5 = Most Bullish)")
    ax.set_ylabel("PE Quintile (Q1 = Cheapest → Q5 = Most Expensive)")

    plt.tight_layout()
    _save(fig, outDir, f"return_heatmap_{suffix}.png")


# ── 3 & 4. Spread ladder plots ────────────────────────────────────────────────

def _plotSpreadLadder(
    spreads: pd.DataFrame,
    spreadsSmallMid: pd.DataFrame,
    outDir: str,
):
    fig, axes = plt.subplots(1, 2, figsize=_FIGSIZE_WIDE)

    # SAT spreads: delta_SAT_k1..k5
    _drawLadder(
        ax=axes[0],
        spreads_full=spreads,
        spreads_sub=spreadsSmallMid,
        prefix="delta_sat_k",
        xlabels=[f"PE Q{i}" for i in range(1, N_QUINTILES + 1)],
        title="SAT Spread (Q5−Q1) by PE Quintile",
        ylabel="Mean Monthly Excess Return",
    )

    # PE spreads: delta_PE_s1..s5
    _drawLadder(
        ax=axes[1],
        spreads_full=spreads,
        spreads_sub=spreadsSmallMid,
        prefix="delta_pe_s",
        xlabels=[f"SAT Q{i}" for i in range(1, N_QUINTILES + 1)],
        title="PE Spread (Q5−Q1) by SAT Quintile",
        ylabel="",
    )

    fig.suptitle("Spread Ladders: Full Sample vs Small/Mid-Cap", fontsize=12)
    plt.tight_layout()
    _save(fig, outDir, "spread_ladders.png")


def _drawLadder(ax, spreads_full, spreads_sub, prefix, xlabels, title, ylabel):
    keys  = [f"{prefix}{i}" for i in range(1, N_QUINTILES + 1)]
    x     = np.arange(N_QUINTILES)
    width = 0.35

    def _get(df, key):
        row = df[df["stat"] == key]
        if row.empty:
            return np.nan, np.nan
        return float(row["mean"].iloc[0]), float(row["t_stat"].iloc[0])

    means_full = [_get(spreads_full, k)[0] for k in keys]
    tstats_full = [_get(spreads_full, k)[1] for k in keys]
    means_sub   = [_get(spreads_sub,  k)[0] for k in keys]
    tstats_sub  = [_get(spreads_sub,  k)[1] for k in keys]

    bars1 = ax.bar(x - width/2, means_full, width, label="Full sample",
                   color=_PALETTE_MAIN, alpha=0.85)
    bars2 = ax.bar(x + width/2, means_sub,  width, label="Small/mid-cap",
                   color=_PALETTE_ROBUST, alpha=0.85)

    # Annotate t-stats
    for bar, t in zip(bars1, tstats_full):
        if not np.isnan(t):
            stars = "***" if abs(t) > 3.0 else ("**" if abs(t) > 2.5 else ("*" if abs(t) > 2.0 else ""))
            if stars:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.001, stars,
                        ha="center", va="bottom", fontsize=8, color=_PALETTE_MAIN)
    for bar, t in zip(bars2, tstats_sub):
        if not np.isnan(t):
            stars = "***" if abs(t) > 3.0 else ("**" if abs(t) > 2.5 else ("*" if abs(t) > 2.0 else ""))
            if stars:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.001, stars,
                        ha="center", va="bottom", fontsize=8, color=_PALETTE_ROBUST)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


# ── 5. Cumulative L/S return ──────────────────────────────────────────────────

def _plotCumulativeLS(rsf: pd.DataFrame, outDir: str):
    rsf = rsf.copy().sort_values("year_month")
    rsf["cum_ret"] = (1 + rsf["rsf_return"]).cumprod() - 1

    dates = [str(p) for p in rsf["year_month"]]
    regime_str = str(pd.Period(str(REGIME_BREAK_DATE)[:7], freq="M"))

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.plot(dates, rsf["cum_ret"] * 100, color=_PALETTE_MAIN, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")

    if regime_str in dates:
        regime_idx = dates.index(regime_str)
        ax.axvline(regime_idx, color=_REGIME_COLOR, linewidth=1.2,
                   linestyle=":", label="Regime break (Dec-2019)")
        ax.legend(fontsize=9)

    _setDateTicks(ax, dates)
    ax.set_title("RSF Long-Short Cumulative Return\n(Long PE=5/SAT=5, Short PE=1/SAT=1)",
                 fontsize=11)
    ax.set_ylabel("Cumulative Excess Return (%)")
    ax.fill_between(range(len(dates)), rsf["cum_ret"] * 100,
                    0, where=rsf["cum_ret"] >= 0,
                    alpha=0.15, color=_PALETTE_MAIN)
    ax.fill_between(range(len(dates)), rsf["cum_ret"] * 100,
                    0, where=rsf["cum_ret"] < 0,
                    alpha=0.15, color=_REGIME_COLOR)
    plt.tight_layout()
    _save(fig, outDir, "cumulative_ls.png")


# ── 6. Rolling Sharpe ─────────────────────────────────────────────────────────

def _plotRollingSharpe(rsf: pd.DataFrame, outDir: str, window: int = 36):
    rsf = rsf.copy().sort_values("year_month")
    roll_mean = rsf["rsf_return"].rolling(window).mean()
    roll_std  = rsf["rsf_return"].rolling(window).std()
    roll_sharpe = (roll_mean / roll_std) * np.sqrt(12)

    dates = [str(p) for p in rsf["year_month"]]
    regime_str = str(pd.Period(str(REGIME_BREAK_DATE)[:7], freq="M"))

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.plot(dates, roll_sharpe, color=_PALETTE_MAIN, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")

    if regime_str in dates:
        ax.axvline(dates.index(regime_str), color=_REGIME_COLOR,
                   linewidth=1.2, linestyle=":", label="Regime break (Dec-2019)")
        ax.legend(fontsize=9)

    _setDateTicks(ax, dates)
    ax.set_title(f"RSF Long-Short Rolling {window}-Month Sharpe Ratio (Annualised)",
                 fontsize=11)
    ax.set_ylabel("Sharpe Ratio")
    plt.tight_layout()
    _save(fig, outDir, "rolling_sharpe.png")


# ── 7. Rolling delta_Cross ────────────────────────────────────────────────────

def _plotRollingDeltaCross(rolling: pd.DataFrame, outDir: str):
    rolling = rolling.copy().sort_values("year_month")
    dates = [str(p) for p in rolling["year_month"]]
    regime_str = str(pd.Period(str(REGIME_BREAK_DATE)[:7], freq="M"))

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.plot(dates, rolling["delta_cross_rolling"] * 100,
            color=_PALETTE_MAIN, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")

    if regime_str in dates:
        ax.axvline(dates.index(regime_str), color=_REGIME_COLOR,
                   linewidth=1.2, linestyle=":",
                   label="Regime break (Dec-2019)")
        ax.legend(fontsize=9)

    _setDateTicks(ax, dates)
    ax.set_title("36-Month Rolling ΔCross\n(SAT Amplification by PE Valuation Over Time)",
                 fontsize=11)
    ax.set_ylabel("ΔCross (%)")
    ax.fill_between(range(len(dates)),
                    rolling["delta_cross_rolling"] * 100,
                    0, alpha=0.15, color=_PALETTE_MAIN)
    plt.tight_layout()
    _save(fig, outDir, "rolling_delta_cross.png")


# ── 8. FF6 alpha heatmap ──────────────────────────────────────────────────────

def _plotAlphaHeatmap(alphas: pd.DataFrame, outDir: str):
    alpha_matrix = alphas.pivot(
        index=PE_QUINTILE_COL, columns=SAT_QUINTILE_COL, values="alpha"
    ) * 100

    tstat_matrix = alphas.pivot(
        index=PE_QUINTILE_COL, columns=SAT_QUINTILE_COL, values="t_stat"
    )

    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)
    vmax = max(abs(alpha_matrix.values.max()), abs(alpha_matrix.values.min()))
    im = ax.imshow(alpha_matrix.values, cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Monthly FF6 Alpha (%)")

    ax.set_xticks(range(N_QUINTILES))
    ax.set_xticklabels([f"SAT Q{i}" for i in range(1, N_QUINTILES + 1)])
    ax.set_yticks(range(N_QUINTILES))
    ax.set_yticklabels([f"PE Q{i}" for i in range(1, N_QUINTILES + 1)])

    for i in range(N_QUINTILES):
        for j in range(N_QUINTILES):
            alpha_val = alpha_matrix.values[i, j]
            t_val     = tstat_matrix.values[i, j]
            stars = "***" if abs(t_val) > 3 else ("**" if abs(t_val) > 2.5 else ("*" if abs(t_val) > 2 else ""))
            color = "white" if abs(alpha_val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{alpha_val:.2f}%{stars}",
                    ha="center", va="center", fontsize=8,
                    color=color, fontweight="bold")

    ax.set_title("FF6 Monthly Alpha by Portfolio Cell\n(* p<0.10, ** p<0.05, *** p<0.01)",
                 fontsize=11)
    ax.set_xlabel("SAT Quintile")
    ax.set_ylabel("PE Quintile")
    plt.tight_layout()
    _save(fig, outDir, "alpha_heatmap.png")


# ── 9. FMB coefficients ───────────────────────────────────────────────────────

def _plotFMBCoefficients(fmb: pd.DataFrame, outDir: str):
    fmb = fmb[fmb["variable"] != "const"].copy()
    if fmb.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(fmb) * 2), 5))

    colors = [_PALETTE_MAIN if abs(t) >= 2.0 else "#aec7e8"
              for t in fmb["t_stat"].fillna(0)]
    bars = ax.bar(fmb["variable"], fmb["lambda_mean"], color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # Pad y-axis so annotations don't fall outside the plot
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.25
    ax.set_ylim(ymin - pad, ymax + pad)

    for bar, (_, row) in zip(bars, fmb.iterrows()):
        t = row["t_stat"]
        if not np.isnan(t):
            stars = "***" if abs(t) > 3 else ("**" if abs(t) > 2.5 else ("*" if abs(t) > 2 else ""))
            label = f"t={t:.2f}{stars}"
            # Place label inside bar if bar is tall enough, otherwise just above/below zero
            h = bar.get_height()
            ypos = h + pad * 0.15 if h >= 0 else h - pad * 0.15
            va   = "bottom" if h >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, ypos, label,
                    ha="center", va=va, fontsize=8)

    ax.set_title("Fama-MacBeth λ Estimates\n(dark blue = |t| ≥ 2.0)", fontsize=11)
    ax.set_ylabel("λ (time-series mean coefficient)")
    ax.set_xlabel("")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    _save(fig, outDir, "fmb_coefficients.png")


# ── 10. Small/mid-cap comparison ──────────────────────────────────────────────

def _plotSmallMidCapComparison(
    spreads_full: pd.DataFrame,
    spreads_sub: pd.DataFrame,
    outDir: str,
):
    fig, axes = plt.subplots(1, 2, figsize=_FIGSIZE_WIDE)

    for ax, prefix, title, xlabels in [
        (axes[0], "delta_sat_k",
         "SAT Spread by PE Quintile", [f"PE Q{i}" for i in range(1, 6)]),
        (axes[1], "delta_pe_s",
         "PE Spread by SAT Quintile", [f"SAT Q{i}" for i in range(1, 6)]),
    ]:
        keys = [f"{prefix}{i}" for i in range(1, N_QUINTILES + 1)]
        x = np.arange(N_QUINTILES)
        width = 0.35

        def _m(df, k):
            row = df[df["stat"] == k]
            return float(row["mean"].iloc[0]) if not row.empty else np.nan

        full_vals = [_m(spreads_full, k) for k in keys]
        sub_vals  = [_m(spreads_sub,  k) for k in keys]

        ax.bar(x - width/2, full_vals, width, label="Full sample",
               color=_PALETTE_MAIN, alpha=0.85)
        ax.bar(x + width/2, sub_vals,  width, label="Small/mid-cap",
               color=_PALETTE_ROBUST, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Full Sample vs Small/Mid-Cap Robustness", fontsize=12)
    plt.tight_layout()
    _save(fig, outDir, "smallmidcap_comparison.png")


# ── 11. Parametric SAT amplification by PE regime ────────────────────────────

def _plotParametricSATAmplification(results: dict, outDir: str):
    """
    Two-panel plot:
      Left  — lambda_SAT by PE quintile for full / pre / post-2020
      Right — fitted f(PE) Weibull curves for pre vs post-2020
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: lambda_SAT bar chart by PE quintile, three regimes ─────────────
    ax = axes[0]
    x = np.arange(5)
    width = 0.25
    palette = {
        "full":      (_PALETTE_MAIN,    "Full sample"),
        "pre_2020":  ("#2ca02c",        "Pre-2020"),
        "post_2020": (_REGIME_COLOR,    "Post-2020"),
    }
    offsets = {"full": -width, "pre_2020": 0, "post_2020": width}

    for regime, (color, label) in palette.items():
        if regime not in results:
            continue
        lam = results[regime]["lambda_sat_by_pe"]
        vals = lam["lambda_sat"].values
        bars = ax.bar(x + offsets[regime], vals, width,
                      label=label, color=color, alpha=0.80)

        # Significance stars above bars
        for bar, (_, row) in zip(bars, lam.iterrows()):
            t = row["t_stat"]
            if not np.isnan(t) and abs(t) >= 2.0:
                stars = "***" if abs(t) > 3 else ("**" if abs(t) > 2.5 else "*")
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + abs(h) * 0.04,
                        stars, ha="center", va="bottom",
                        fontsize=7, color=color)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"PE Q{i}" for i in range(1, 6)])
    ax.set_ylabel("λ_SAT (within-PE-quintile FMB coefficient)")
    ax.set_title("SAT Premium Conditional on PE Quintile\n"
                 "(* p<0.10, ** p<0.05, *** p<0.01)", fontsize=10)
    ax.legend(fontsize=8)

    # ── Right: fitted f(PE) curves pre vs post-2020 ───────────────────────────
    ax2 = axes[1]
    curve_palette = {
        "full":      (_PALETTE_MAIN, "Full sample", "-"),
        "pre_2020":  ("#2ca02c",     "Pre-2020",    "--"),
        "post_2020": (_REGIME_COLOR, "Post-2020",   ":"),
    }

    any_curve = False
    for regime, (color, label, ls) in curve_palette.items():
        if regime not in results:
            continue
        fit = results[regime]["fpe_fit"]
        lam = results[regime]["lambda_sat_by_pe"]

        # Scatter the observed lambda_SAT points
        valid = lam.dropna(subset=["pe_median", "lambda_sat"])
        ax2.scatter(valid["pe_median"], valid["lambda_sat"],
                    color=color, zorder=5, s=40, alpha=0.8)

        if fit.get("converged"):
            ax2.plot(fit["pe_grid"], fit["fpe_curve"],
                     color=color, linestyle=ls, linewidth=1.8,
                     label=f"{label} (r²={fit['r_squared']:.2f})")
            any_curve = True
        else:
            # Fallback: connect dots
            ax2.plot(valid["pe_median"], valid["lambda_sat"],
                     color=color, linestyle=ls, linewidth=1.5,
                     label=f"{label} (no fit)")

    ax2.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax2.set_xlabel("Median Trailing PE (within quintile)")
    ax2.set_ylabel("λ_SAT")
    ax2.set_title("Parametric f(PE): SAT Amplification Shape\n"
                  "by Valuation Regime", fontsize=10)
    ax2.legend(fontsize=8)

    fig.suptitle("Parametric PE–SAT Interaction: Non-linear Amplification by Regime\n"
                 "(Regime break: Mar-2020 COVID crash)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    _save(fig, outDir, "parametric_sat_amplification.png")


# ── 12. Daily SAT spread comparison (raw / 5d / 10d) ─────────────────────────

def _plotDailySpreadComparison(dailySpreads: pd.DataFrame, outDir: str):
    """
    Two-panel plot comparing daily SAT spreads across variants.
    Left:  delta_SAT (Q5-Q1) by PE quintile for each variant.
    Right: delta_PE  (Q5-Q1) by SAT quintile for each variant.
    """
    fig, axes = plt.subplots(1, 2, figsize=_FIGSIZE_WIDE)
    palette = {"raw": _PALETTE_MAIN, "5d": "#2ca02c", "10d": _REGIME_COLOR}
    x = np.arange(N_QUINTILES)
    width = 0.25
    offsets = {"raw": -width, "5d": 0, "10d": width}

    for ax, prefix, xlabel_fn, title in [
        (axes[0], "delta_sat_k",
         lambda i: f"PE Q{i}", "Daily SAT Spread (Q5-Q1) by PE Quintile"),
        (axes[1], "delta_pe_s",
         lambda i: f"SAT Q{i}", "Daily PE Spread (Q5-Q1) by SAT Quintile"),
    ]:
        for variant, color in palette.items():
            sub = dailySpreads[dailySpreads["variant"] == variant]
            vals = []
            for i in range(1, N_QUINTILES + 1):
                row = sub[sub["stat"] == f"{prefix}{i}"]
                vals.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)

            bars = ax.bar(x + offsets[variant], vals, width,
                          label=variant, color=color, alpha=0.82)

            # Significance stars
            for bar, i in zip(bars, range(1, N_QUINTILES + 1)):
                row = sub[sub["stat"] == f"{prefix}{i}"]
                if row.empty:
                    continue
                t = row["t_stat"].iloc[0]
                if not np.isnan(t) and abs(t) >= 2.0:
                    stars = "***" if abs(t) > 3 else "**" if abs(t) > 2.5 else "*"
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            h + abs(h) * 0.04,
                            stars, ha="center", va="bottom",
                            fontsize=7, color=color)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([xlabel_fn(i) for i in range(1, 6)], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Mean Daily Excess Return")
        ax.legend(title="SAT variant", fontsize=8)

    fig.suptitle("Daily FF-25 Spread Statistics: Raw vs 5-Day vs 10-Day SAT\n"
                 "(* p<0.10  ** p<0.05  *** p<0.01, NW(22) s.e.)", fontsize=11)
    plt.tight_layout()
    _save(fig, outDir, "daily_spread_comparison.png")


# ── 13. Daily FMB coefficient comparison ─────────────────────────────────────

def _plotDailyFMBComparison(dailyFMB: pd.DataFrame, outDir: str):
    """
    Three-panel bar chart: one panel per SAT variant (raw, 5d, 10d).
    Each panel shows lambda estimates with t-stat annotations, mirroring
    the monthly fmb_coefficients.png but for daily cross-sections.
    Also overlays the monthly FMB result as a reference dashed line if
    the magnitudes are comparable.
    """
    variants = ["raw", "5d", "10d"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    variables = ["pe_quintile", "sat_quintile", "pe_sat_interaction"]

    for ax, variant in zip(axes, variants):
        sub = dailyFMB[
            (dailyFMB["variant"] == variant) &
            (dailyFMB["regime"]  == "full") &
            (dailyFMB["variable"].isin(variables))
        ].copy()

        if sub.empty:
            ax.set_title(f"SAT {variant} — no data")
            continue

        colors = [_PALETTE_MAIN if abs(t) >= 2.0 else "#aec7e8"
                  for t in sub["t_stat"].fillna(0)]
        bars = ax.bar(sub["variable"], sub["lambda_mean"],
                      color=colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

        # Pad y-axis so annotations don't clip
        ymin, ymax = ax.get_ylim()
        pad = max((ymax - ymin) * 0.25, 1e-6)
        ax.set_ylim(ymin - pad, ymax + pad)

        for bar, (_, row) in zip(bars, sub.iterrows()):
            t = row["t_stat"]
            if not np.isnan(t):
                stars = ("***" if abs(t) > 3 else "**" if abs(t) > 2.5
                         else "*" if abs(t) > 2 else "")
                h = bar.get_height()
                ypos = h + pad * 0.15 if h >= 0 else h - pad * 0.15
                va = "bottom" if h >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                        f"t={t:.2f}{stars}",
                        ha="center", va=va, fontsize=7.5)

        n_days = int(sub["n_days"].iloc[0]) if "n_days" in sub.columns else 0
        ax.set_title(f"SAT {variant}\n({n_days:,} daily cross-sections)",
                     fontsize=10)
        ax.set_ylabel("λ (daily mean coefficient)" if variant == "raw" else "")
        plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=8)

    fig.suptitle("Daily Fama-MacBeth λ Estimates by SAT Variant\n"
                 "(dark blue = |t| ≥ 2.0,  NW(22) standard errors)", fontsize=11)
    plt.tight_layout()
    _save(fig, outDir, "daily_fmb_comparison.png")


# ── 14. Daily rolling delta_Cross ─────────────────────────────────────────────

def _plotDailyRollingDeltaCross(dailyRollingDC: pd.DataFrame, outDir: str):
    """
    Rolling 60-day delta_Cross for each SAT variant — daily analogue of
    rolling_delta_cross.png. Three lines on one axes with regime break.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)

    palette = {
        "delta_cross_rolling_raw": (_PALETTE_MAIN, "raw SAT",  "-"),
        "delta_cross_rolling_5d":  ("#2ca02c",     "5d SAT",   "--"),
        "delta_cross_rolling_10d": (_REGIME_COLOR, "10d SAT",  ":"),
    }

    dates = pd.to_datetime(dailyRollingDC["date"])
    x = np.arange(len(dates))

    for col, (color, label, ls) in palette.items():
        if col not in dailyRollingDC.columns:
            continue
        vals = dailyRollingDC[col].values * 100
        ax.plot(x, vals, color=color, linestyle=ls, linewidth=1.5,
                label=label, alpha=0.85)

    # Regime break line
    regime_date = pd.Timestamp("2020-03-01")
    regime_idx  = np.searchsorted(dates.values, np.datetime64(regime_date))
    if 0 < regime_idx < len(dates):
        ax.axvline(regime_idx, color=_REGIME_COLOR, linewidth=1.2,
                   linestyle="dotted", alpha=0.8,
                   label="Regime break (Mar-2020)")

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    _setDateTicks(ax, [str(d)[:7] for d in dates], max_ticks=10)
    ax.set_title("60-Day Rolling ΔCross (Daily) by SAT Variant\n"
                 "(SAT amplification of PE premium over time)", fontsize=11)
    ax.set_ylabel("ΔCross (%)")
    ax.legend(fontsize=9)
    ax.fill_between(x,
                    dailyRollingDC.get("delta_cross_rolling_10d",
                                       pd.Series(0, index=range(len(x)))).values * 100,
                    0, alpha=0.08, color=_REGIME_COLOR)
    plt.tight_layout()
    _save(fig, outDir, "daily_rolling_delta_cross.png")


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fig, outDir: str, filename: str):
    path = os.path.join(outDir, filename)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {filename}")


def _setDateTicks(ax, dates: list, max_ticks: int = 8):
    """Show every N-th date label to avoid overcrowding."""
    step = max(1, len(dates) // max_ticks)
    ax.set_xticks(range(0, len(dates), step))
    ax.set_xticklabels(dates[::step], rotation=30, ha="right", fontsize=8)