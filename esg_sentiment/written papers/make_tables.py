"""
make_tables.py — Generate LaTeX table fragments for both ESG papers from DuckDB.

Reads the corrected pipeline outputs (portfolio_returns, ff6_alphas,
fmb_results, chow_results, esf_factor, esf_spanning, esf_correlations,
alpha_comparison_*, daily_spreads, daily_fmb) and writes one .tex fragment
per table into paper/tables/. The papers \input these fragments, so
recompiling after a pipeline rerun updates every number.

Run from the repo root:  python paper/make_tables.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import Database
from config.constants import DB_PATH, N_QUINTILES, FF_FACTORS
from portfolios.construct_ff25 import computeSpreads
from estimation.parametric_sat import estimateParametricSATAmplification

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")

# Old (contaminated) headline numbers captured before the corrected rerun
_OLD_ESF_LS_MEAN_PCT = 7.76
_OLD_MEAN_ABS_ALPHA_PCT = 2.67
_OLD_N_SIG_ALPHAS = 25


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def pct(x: float, dp: int = 2) -> str:
    return f"{x * 100:.{dp}f}\\%"


def write(name: str, content: str):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  wrote {name}")


def cellMeans(portfolioReturns: pd.DataFrame) -> pd.DataFrame:
    return (
        portfolioReturns
        .groupby(["esg_quintile", "sat_quintile"])["excess_ret"]
        .mean()
        .unstack("sat_quintile")
    )


def table_ff25_returns(db: Database):
    """Table: 5x5 mean monthly excess returns with spread t-stats."""
    pr = db.readTable("portfolio_returns")
    pr["year_month"] = pd.PeriodIndex(pr["year_month"], freq="M")
    means = cellMeans(pr)
    spreads = computeSpreads(pr).set_index("stat")

    def sp(stat):
        row = spreads.loc[stat]
        return row["mean"], row["t_stat"], stars(row["p_value"])

    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"& \multicolumn{5}{c}{\textbf{SAT Quintile}} & \\",
        r"\cmidrule(lr){2-6}",
        r"\textbf{ESG Quintile} & Q1 (Bear) & Q2 & Q3 & Q4 & Q5 (Bull) & \textbf{Q5$-$Q1} ($t$) \\",
        r"\midrule",
    ]
    row_labels = {1: "Q1 (Low ESG)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (High ESG)"}
    for k in range(1, N_QUINTILES + 1):
        cells = " & ".join(pct(means.loc[k, s]) for s in range(1, N_QUINTILES + 1))
        m, t, st = sp(f"delta_sat_k{k}")
        lines.append(f"{row_labels[k]} & {cells} & {pct(m)} ({t:.2f}){st} \\\\")
    lines.append(r"\midrule")
    esg_means = " & ".join(
        pct(sp(f"delta_esg_s{s}")[0]) for s in range(1, N_QUINTILES + 1)
    )
    m, t, st = sp("delta_cross")
    lines.append(rf"\textbf{{Q5$-$Q1}} & {esg_means} & {pct(m)} ({t:.2f}){st} \\")
    esg_ts = " & ".join(
        f"({sp(f'delta_esg_s{s}')[1]:.2f}){sp(f'delta_esg_s{s}')[2]}"
        for s in range(1, N_QUINTILES + 1)
    )
    lines.append(rf"($t$) & {esg_ts} & \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ff25_returns.tex", "\n".join(lines))

    write("val_n_months.tex", str(pr["year_month"].nunique()))
    write("val_delta_cross.tex", f"{sp('delta_cross')[0]*100:.2f}")
    write("val_delta_cross_t.tex", f"{sp('delta_cross')[1]:.2f}")
    sat_rows = [spreads.loc[f"delta_sat_k{k}"] for k in range(1, 6)]
    write("val_sat_lo.tex", f"{min(r['mean'] for r in sat_rows)*100:.2f}")
    write("val_sat_hi.tex", f"{max(r['mean'] for r in sat_rows)*100:.2f}")
    write("val_sat_tmax.tex", f"{max(abs(r['t_stat']) for r in sat_rows):.2f}")


def table_ff25_alphas(db: Database):
    al = db.readTable("ff6_alphas").set_index(["esg_quintile", "sat_quintile"])
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"& \multicolumn{5}{c}{\textbf{SAT Quintile}} \\",
        r"\cmidrule(lr){2-6}",
        r"\textbf{ESG Quintile} & Q1 (Bear) & Q2 & Q3 & Q4 & Q5 (Bull) \\",
        r"\midrule",
    ]
    row_labels = {1: "Q1 (Low ESG)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (High ESG)"}
    for k in range(1, N_QUINTILES + 1):
        cells = []
        for s in range(1, N_QUINTILES + 1):
            row = al.loc[(k, s)]
            cells.append(f"{pct(row['alpha'])}{stars(row['p_value'])}")
        lines.append(f"{row_labels[k]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ff25_alphas.tex", "\n".join(lines))

    write("val_alpha_min.tex", f"{al['alpha'].min()*100:.2f}")
    write("val_alpha_max.tex", f"{al['alpha'].max()*100:.2f}")
    write("val_alpha_nsig.tex", str(int((al["t_stat"].abs() > 2).sum())))


def table_fmb(db: Database):
    fmb = db.readTable("fmb_results")
    name_map = {
        "const": r"const",
        "esg_quintile": r"ESGQ",
        "sat_quintile": r"SATQ",
        "esg_sat_interaction": r"ESGQ$\times$SATQ",
    }
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Variable & $\hat{\lambda}$ & $t$-stat & $p$-value & $N$ (months) \\",
        r"\midrule",
    ]
    order = ["const", "esg_quintile", "sat_quintile", "esg_sat_interaction"]
    fmb = fmb.set_index("variable").loc[order].reset_index()
    for _, row in fmb.iterrows():
        lines.append(
            f"{name_map[row['variable']]} & {row['lambda_mean']:.6f} & "
            f"{row['t_stat']:.2f} & {row['p_value']:.3f} & "
            f"{int(row['n_months'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_fmb.tex", "\n".join(lines))


def table_chow(db: Database):
    chow = db.readTable("chow_results")
    name_map = {"const": "const",
                "post_2018": "Post-2018 indicator (EU Action Plan)",
                "post_2020": "Post-2020 indicator (retail surge + SFDR)"}
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Variable & Coefficient & $t$-stat & $p$-value \\",
        r"\midrule",
    ]
    for _, row in chow.iterrows():
        name = name_map.get(row["variable"], row["variable"])
        lines.append(
            f"{name} & {row['coefficient']:.6f} & {row['t_stat']:.2f} & "
            f"{row['p_value']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_chow.tex", "\n".join(lines))


def table_parametric(db: Database):
    """Table: ESG-quintile-conditional SAT premium across the four regimes."""
    panel = db.readTable("monthly_panel")
    panel["year_month"] = pd.PeriodIndex(panel["year_month"], freq="M")
    results = estimateParametricSATAmplification(panel)

    regimes = ["full", "pre_2018", "between_2018_2020", "post_2020"]
    heads = ["Full Sample", "Pre-2018", "2018--2019", "Post-2020"]
    lam = {r: results[r]["lambda_sat_by_esg"].set_index("esg_quintile")
           for r in regimes}
    fits = {r: results[r]["fesg_fit"] for r in regimes}

    lines = [
        r"\begin{tabular}{lr" + "rr" * len(regimes) + "}",
        r"\toprule",
        "& & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{h}}}" for h in heads) + r" \\",
        "".join(rf"\cmidrule(lr){{{3+2*i}-{4+2*i}}}" for i in range(len(regimes))),
        r"ESG Quintile & Med.\ ESG & " +
        " & ".join([r"$\hat{\lambda}$ & $t$"] * len(regimes)) + r" \\",
        r"\midrule",
    ]
    row_labels = {1: "Q1 (Low)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (High)"}

    def fmt(v, spec):
        return "--" if pd.isna(v) else spec.format(v)

    for k in range(1, N_QUINTILES + 1):
        cells = [f"{lam['full'].loc[k, 'esg_median']:.1f}"]
        for r in regimes:
            row = lam[r].loc[k]
            cells += [fmt(row["lambda_sat"], "{:.5f}"),
                      fmt(row["t_stat"], "{:.2f}")]
        lines.append(f"{row_labels[k]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    ncol = 2 + 2 * len(regimes)
    lines.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{Weibull $f(\mathrm{{ESG}})$ fit diagnostics}}}} \\")

    def fitrow(label, key, fmt="{:.3f}"):
        cells = []
        for r in regimes:
            fit = fits[r]
            if fit.get("converged"):
                v = fit["r_squared"] if key == "r_squared" else fit["params"][key]
                cells.append(rf"\multicolumn{{2}}{{c}}{{{fmt.format(v)}}}")
            else:
                cells.append(r"\multicolumn{2}{c}{n/a}")
        return f"{label} & & " + " & ".join(cells) + r" \\"

    lines.append(fitrow(r"$R^2$", "r_squared"))
    lines.append(fitrow(r"peak ESG", "peak_esg", "{:.1f}"))
    lines.append(fitrow(r"scale", "scale", "{:.4f}"))
    lines.append(fitrow(r"asymmetry", "asymmetry", "{:.2f}"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_parametric.tex", "\n".join(lines))


def table_lookahead_comparison(db: Database):
    """Table: contaminated vs corrected headline statistics."""
    pr = db.readTable("portfolio_returns")
    pr["year_month"] = pd.PeriodIndex(pr["year_month"], freq="M")
    spreads = computeSpreads(pr).set_index("stat")
    al = db.readTable("ff6_alphas")
    ls = db.readTable("esf_longshort")
    dc = spreads.loc["delta_cross"]

    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Statistic & Contaminated (sort $t$, return $t$) & Corrected (sort $t$, return $t{+}1$) \\",
        r"\midrule",
        rf"Corner long--short mean (\%/month) & {_OLD_ESF_LS_MEAN_PCT:.2f} & "
        rf"{ls['rsf_return'].mean()*100:.2f} \\",
        rf"Mean $|$FF6 alpha$|$ (\%/month) & {_OLD_MEAN_ABS_ALPHA_PCT:.2f} & "
        rf"{al['alpha'].abs().mean()*100:.2f} \\",
        rf"Alphas with $|t| > 2$ & {_OLD_N_SIG_ALPHAS} of 25 & "
        rf"{int((al['t_stat'].abs() > 2).sum())} of 25 \\",
        rf"$\Delta_\text{{Cross}}$ (\%/month) & --- & "
        rf"${dc['mean']*100:+.2f}$ ($t = {dc['t_stat']:.2f}$) \\",
        r"FMB $\lambda_\text{Int}$ & insignificant & insignificant \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write("tab_lookahead.tex", "\n".join(lines))
    write("val_ls_mean.tex", f"{ls['rsf_return'].mean()*100:.2f}")


def table_daily(db: Database):
    if not db.hasTable("daily_fmb"):
        print("  [skip] daily_fmb not in DB yet")
        return
    fmb = db.readTable("daily_fmb")
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{SAT raw} & \multicolumn{2}{c}{SAT 5-day} & \multicolumn{2}{c}{SAT 10-day} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Variable & $\hat{\lambda}$ & $t$ & $\hat{\lambda}$ & $t$ & $\hat{\lambda}$ & $t$ \\",
        r"\midrule",
    ]
    name_map = {
        "esg_quintile": "ESGQ",
        "sat_quintile": "SATQ",
        "esg_sat_interaction": r"ESGQ$\times$SATQ",
        "pe_quintile": "ESGQ",
        "pe_sat_interaction": r"ESGQ$\times$SATQ",
        "const": "const",
    }
    present = fmb["variable"].unique().tolist()
    order = [v for v in ["const", "esg_quintile", "pe_quintile",
                         "sat_quintile", "esg_sat_interaction",
                         "pe_sat_interaction"] if v in present]
    for var in order:
        cells = []
        for variant in ["raw", "5d", "10d"]:
            sub = fmb[(fmb["variant"] == variant) & (fmb["variable"] == var)]
            if "regime" in fmb.columns:
                sub = sub[sub["regime"] == "full"]
            if sub.empty:
                cells += ["n/a", ""]
            else:
                r = sub.iloc[0]
                cells += [f"{r['lambda_mean']:.6f}",
                          f"{r['t_stat']:.2f}{stars(r['p_value'])}"]
        lines.append(f"{name_map[var]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_daily_fmb.tex", "\n".join(lines))

    spreads = db.readTable("daily_spreads")
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"SAT variant & $\Delta_\text{Cross}$ (\%/day) & $t$ (NW22) & $N$ days \\",
        r"\midrule",
    ]
    for variant in ["raw", "5d", "10d"]:
        sub = spreads[(spreads["variant"] == variant) &
                      (spreads["stat"] == "delta_cross")]
        if sub.empty:
            continue
        r = sub.iloc[0]
        label = {"raw": "Raw daily", "5d": "5-day rolling",
                 "10d": "10-day rolling"}[variant]
        lines.append(
            f"{label} & {r['mean']*100:.4f} & "
            f"{r['t_stat']:.2f}{stars(r['p_value'])} & {int(r['n_days'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_daily_spreads.tex", "\n".join(lines))


def tables_factor_paper(db: Database):
    esf = db.readTable("esf_factor")
    r = esf["ESF"]
    write("val_esf_mean.tex", f"{r.mean()*100:.3f}")
    write("val_esf_sd.tex", f"{r.std()*100:.2f}")
    write("val_esf_sharpe.tex", f"{r.mean()/r.std()*np.sqrt(12):.2f}")
    write("val_esf_nmonths.tex", str(len(esf)))

    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Statistic & Value \\",
        r"\midrule",
        rf"Mean (\%/month) & {r.mean()*100:.3f} \\",
        rf"Std.\ dev.\ (\%/month) & {r.std()*100:.2f} \\",
        rf"Annualised Sharpe ratio & {r.mean()/r.std()*np.sqrt(12):.2f} \\",
        rf"Skewness & {r.skew():.2f} \\",
        rf"Excess kurtosis & {r.kurtosis():.2f} \\",
        rf"Months & {len(esf)} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write("tab_esf_summary.tex", "\n".join(lines))

    sp = db.readTable("esf_spanning").set_index("variable")
    name_map = {"alpha": r"$\alpha$", "Mkt_RF": "Mkt-RF", "SMB": "SMB",
                "HML": "HML", "RMW": "RMW", "CMA": "CMA", "UMD": "UMD"}
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Variable & Coefficient & $t$-stat & $p$-value \\",
        r"\midrule",
    ]
    for var in ["alpha"] + FF_FACTORS:
        row = sp.loc[var]
        coef = row["coefficient"]
        coef_str = f"{coef*100:.3f}\\%" if var == "alpha" else f"{coef:.3f}"
        lines.append(
            f"{name_map[var]} & {coef_str}{stars(row['p_value'])} & "
            f"{row['t_stat']:.2f} & {row['p_value']:.4f} \\\\"
        )
    lines.append(r"\midrule")
    lines.append(rf"$R^2$ & {sp.loc['r_squared','coefficient']:.3f} & & \\")
    lines.append(rf"$N$ (months) & {int(sp.loc['n_months','coefficient'])} & & \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_spanning.tex", "\n".join(lines))
    write("val_span_alpha.tex", f"{sp.loc['alpha','coefficient']*100:.2f}")
    write("val_span_alpha_t.tex", f"{sp.loc['alpha','t_stat']:.2f}")
    write("val_span_r2.tex", f"{sp.loc['r_squared','coefficient']:.3f}")

    corr = db.readTable("esf_correlations")
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"FF factor & Corr.\ with ESF \\",
        r"\midrule",
    ]
    for _, row in corr.iterrows():
        lines.append(f"{name_map.get(row['factor'], row['factor'])} & "
                     f"{row['correlation']:+.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_correlations.tex", "\n".join(lines))

    summ = db.readTable("alpha_comparison_summary")
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Mean $|\alpha|$ (\%/month) & $\#|t|>2$ & Mean $R^2$ \\",
        r"\midrule",
    ]
    for _, row in summ.iterrows():
        model = row["model"].replace("RSF", "ESF")
        lines.append(
            f"{model} & {row['mean_abs_alpha']*100:.3f} & "
            f"{int(row['n_significant'])} & {row['mean_r_squared']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_alpha_comparison.tex", "\n".join(lines))

    comp = db.readTable("alpha_comparison_esf").set_index(
        ["esg_quintile", "sat_quintile"]
    )
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"& \multicolumn{5}{c}{\textbf{SAT Quintile}} \\",
        r"\cmidrule(lr){2-6}",
        r"\textbf{ESG Quintile} & Q1 (Bear) & Q2 & Q3 & Q4 & Q5 (Bull) \\",
        r"\midrule",
    ]
    row_labels = {1: "Q1 (Low ESG)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (High ESG)"}
    for k in range(1, N_QUINTILES + 1):
        cells = [f"{comp.loc[(k, s), 'beta_ESF']:.2f}"
                 for s in range(1, N_QUINTILES + 1)]
        lines.append(f"{row_labels[k]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_esf_betas.tex", "\n".join(lines))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with Database(DB_PATH) as db:
        print("Generating monthly tables...")
        table_ff25_returns(db)
        table_ff25_alphas(db)
        table_fmb(db)
        table_chow(db)
        table_lookahead_comparison(db)
        print("Generating parametric table (re-runs quintile FMB, ~1 min)...")
        table_parametric(db)
        print("Generating daily tables...")
        table_daily(db)
        print("Generating factor-paper tables...")
        tables_factor_paper(db)
    print("Done.")


if __name__ == "__main__":
    main()
