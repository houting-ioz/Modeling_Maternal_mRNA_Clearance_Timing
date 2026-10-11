from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "xgboost_classify_results"
OUT = ROOT / "outputs/xgboost_classify_results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)
sns.set_theme(style="white", context="paper")

COMBO_ORDER = [
    "base_only",
    "only_full_kmer123",
    "base_full_kmer123",
    "base_m6a_micro_transposon",
    "base_f1",
    "base_polyA_f1",
    "base_polyA_f1_codon",
    "all_scalar",
]
DISPLAY = {
    "base_only": "Base only",
    "only_full_kmer123": "k-mer only",
    "base_full_kmer123": "Base + k-mer",
    "base_m6a_micro_transposon": "Base + m6A + TE",
    "base_f1": "Base + F1",
    "base_polyA_f1": "Base + poly(A) + F1",
    "base_polyA_f1_codon": "Base + poly(A) + F1 + codon",
    "all_scalar": "All scalar features",
}
COLORS = {
    "base_only": "#A9B6C6",
    "only_full_kmer123": "#C7CDD4",
    "base_full_kmer123": "#7E9AB7",
    "base_m6a_micro_transposon": "#91AFC9",
    "base_f1": "#547B9E",
    "base_polyA_f1": "#C44E52",
    "base_polyA_f1_codon": "#8C2D3A",
    "all_scalar": "#B8860B",
}
BEST = "base_polyA_f1"


def load_data():
    summary = pd.read_csv(DATA / "all_combos_summary.csv")
    summary["combo"] = summary["combo"].astype(str).str.strip()
    summary["combo"] = pd.Categorical(summary["combo"], COMBO_ORDER, ordered=True)
    summary = summary.sort_values("combo").reset_index(drop=True)
    folds = []
    for combo in COMBO_ORDER:
        path = DATA / f"{combo}_fold_metrics.csv"
        df = pd.read_csv(path)
        df["combo"] = combo
        folds.append(df)
    folds = pd.concat(folds, ignore_index=True)
    preds = {}
    for combo in COMBO_ORDER:
        preds[combo] = pd.read_csv(DATA / f"{combo}_predictions.csv")
    return summary, folds, preds


def bh_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def add_panel_label(ax, label):
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def save_pub(fig, stem):
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_tests(folds):
    wide = folds.pivot(index="fold", columns="combo", values="auc")[COMBO_ORDER]
    fr = friedmanchisquare(*[wide[c].values for c in COMBO_ORDER])
    raw = []
    for combo in COMBO_ORDER:
        if combo == BEST:
            continue
        try:
            stat = wilcoxon(wide[BEST], wide[combo], alternative="two-sided", method="auto")
            raw.append((combo, float(stat.statistic), float(stat.pvalue)))
        except ValueError:
            raw.append((combo, np.nan, np.nan))
    qvals = bh_adjust([r[2] for r in raw]) if raw else []
    records = [{"test": "Friedman", "comparison": "all combinations", "statistic": fr.statistic, "p_value": fr.pvalue, "q_value": np.nan}]
    for (combo, stat, p), q in zip(raw, qvals):
        records.append({"test": "Wilcoxon signed-rank", "comparison": f"{BEST} vs {combo}", "statistic": stat, "p_value": p, "q_value": q})
    pd.DataFrame(records).to_csv(OUT / "statistical_tests.csv", index=False)
    return wide, fr, dict((c, q) for (c, _, _), q in zip(raw, qvals))


def p_label(q):
    if not np.isfinite(q):
        return "n.s."
    if q < 1e-3:
        return "q < 0.001"
    if q < 0.01:
        return "q < 0.01"
    if q < 0.05:
        return "q < 0.05"
    return "n.s."


def make_table(summary, preds):
    """单独导出二分类性能表，不再将表格嵌入组合图。"""
    best_pred = preds[BEST]
    y = best_pred["y_true"].astype(int).to_numpy()
    score = best_pred["y_proba"].to_numpy()
    best_fpr, best_tpr, _ = roc_curve(y, score)
    best_roc_auc = auc(best_fpr, best_tpr)

    tab = summary.copy().sort_values("oof_auc", ascending=False)
    table_data = pd.DataFrame({
        "Feature set": tab["combo"].map(DISPLAY),
        "OOF AUC": tab["oof_auc"].map(lambda x: f"{x:.3f}"),
        "OOF F1": tab["oof_f1"].map(lambda x: f"{x:.3f}"),
        "10-fold AUPR (mean +/- s.d.)": tab.apply(
            lambda r: f"{r.aupr_mean:.3f} +/- {r.aupr_std:.3f}", axis=1
        ),
        "Features": tab["n_features"].astype(int),
    })
    table_data.to_csv(OUT / "Table_1_XGBoost_degradation_classification.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    ax.axis("off")
    ax.set_title(
        "XGBoost classification performance across degradation-related feature sets",
        loc="left", fontsize=8, pad=5, fontweight="bold",
    )
    rows = table_data.astype(str).values.tolist()
    table = ax.table(
        cellText=rows, colLabels=table_data.columns.tolist(),
        cellLoc="center", colLoc="center", bbox=[0, 0.13, 1, 0.77],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.4)
    widths = [0.34, 0.13, 0.13, 0.25, 0.12]
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D8DDE3")
        cell.set_linewidth(0.45)
        if r == 0:
            cell.set_facecolor("#EEF1F4")
            cell.set_text_props(weight="bold")
        elif rows[r - 1][0] == DISPLAY[BEST]:
            cell.set_facecolor("#F8E6E7")
        else:
            cell.set_facecolor("white")
        if c < len(widths):
            cell.set_width(widths[c])
    ax.text(
        0, 0.04,
        f"Best configuration: {DISPLAY[BEST]}; OOF AUC = {best_roc_auc:.3f} "
        f"(n = {len(y)} samples)",
        transform=ax.transAxes, fontsize=6.2, color="#40474F",
    )
    save_pub(fig, "Table_1_XGBoost_degradation_classification")


def make_main(summary, folds, preds, wide, friedman, qvals):
    best_pred = preds[BEST]
    y = best_pred["y_true"].astype(int).to_numpy()
    score = best_pred["y_proba"].to_numpy()
    cm = confusion_matrix(y, (score >= 0.5).astype(int), normalize="true") * 100

    fig = plt.figure(figsize=(8.4, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 0.90], hspace=0.47)
    top_gs = gs[0].subgridspec(1, 3, width_ratios=[1.68, 1.0, 1.0], wspace=0.30)
    bottom_gs = gs[1].subgridspec(1, 4, wspace=0.42)
    # Give the boxplot, ROC and PR panels equal top-row widths. ROC and PR
    # retain square axes without being compressed into half-width columns.
    ax_box = fig.add_subplot(top_gs[0, 0])
    ax_roc = fig.add_subplot(top_gs[0, 1])
    ax_pr = fig.add_subplot(top_gs[0, 2])
    # 2×2 混淆矩阵只占一列，避免单元格被拉得过大；说明区使用其余三列。
    ax_cm = fig.add_subplot(bottom_gs[0, 0])
    ax_notes = fig.add_subplot(bottom_gs[0, 1:])
    add_panel_label(ax_box, "a")
    add_panel_label(ax_roc, "b")
    add_panel_label(ax_pr, "c")
    add_panel_label(ax_cm, "d")

    # Fold distribution and paired significance versus the best model.
    long = folds.copy()
    long["display"] = long["combo"].map(DISPLAY)
    sns.boxplot(data=long, x="display", hue="display", y="auc", order=[DISPLAY[c] for c in COMBO_ORDER], palette=[COLORS[c] for c in COMBO_ORDER], width=0.62, fliersize=0, linewidth=0.65, ax=ax_box, legend=False, dodge=False)
    sns.stripplot(data=long, x="display", y="auc", order=[DISPLAY[c] for c in COMBO_ORDER], color="#30343B", size=2.2, alpha=0.65, jitter=0.08, ax=ax_box)
    ax_box.set_ylabel("10-fold validation AUC")
    ax_box.set_xlabel("")
    box_labels = ["Base", "k-mer", "Base+k", "Base+m6A", "Base+F1", "Base+polyA+F1", "...+codon", "All scalar"]
    ax_box.set_xticks(np.arange(len(COMBO_ORDER)))
    ax_box.set_xticklabels(box_labels, rotation=45, rotation_mode="anchor", ha="right", fontsize=5.5)
    ax_box.set_ylim(0.58, 0.91)
    # Use explicit, evenly spaced AUC ticks so readers can locate values such
    # as 0.60 and 0.80 without estimating their positions from the boxplots.
    auc_ticks = np.arange(0.60, 0.901, 0.05)
    ax_box.set_yticks(auc_ticks)
    ax_box.set_yticklabels([f"{value:.2f}" for value in auc_ticks], fontsize=6.0)
    ax_box.set_axisbelow(True)
    ax_box.grid(axis="y", color="#DDE2E7", linewidth=0.45, alpha=0.75)
    ax_box.text(0.02, 0.98, f"Friedman P = {friedman.pvalue:.2e}", transform=ax_box.transAxes, ha="left", va="top", fontsize=5.8)
    for i, c in enumerate(COMBO_ORDER):
        if c == BEST:
            continue
        q = qvals.get(c, np.nan)
        ax_box.text(i, 0.595, p_label(q), ha="center", va="bottom", rotation=90, rotation_mode="anchor", fontsize=5.0, color="#5A6068")

    roc_handles, roc_labels = [], []
    for combo in COMBO_ORDER:
        pred = preds[combo]
        yy = pred["y_true"].astype(int).to_numpy()
        ss = pred["y_proba"].to_numpy()
        fpr, tpr, _ = roc_curve(yy, ss)
        score_auc = auc(fpr, tpr)
        (line,) = ax_roc.plot(fpr, tpr, color=COLORS[combo], lw=1.4 if combo != BEST else 2.2, alpha=0.9, label=f"{DISPLAY[combo]} (AUC = {score_auc:.3f})")
        roc_handles.append(line)
        roc_labels.append(f"{DISPLAY[combo]} (AUC = {score_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="#B8BEC6", lw=0.8, ls="--")
    ax_roc.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
    ax_roc.set_title("OOF ROC curves", loc="left", fontsize=7.5, fontweight="bold")
    ax_roc.set_box_aspect(1)

    for combo in COMBO_ORDER:
        pred = preds[combo]
        yy = pred["y_true"].astype(int).to_numpy()
        ss = pred["y_proba"].to_numpy()
        prec, rec, _ = precision_recall_curve(yy, ss)
        ap = average_precision_score(yy, ss)
        ax_pr.plot(rec, prec, color=COLORS[combo], lw=1.4 if combo != BEST else 2.2, alpha=0.9)
    ax_pr.axhline(y.mean(), color="#B8BEC6", lw=0.8, ls="--")
    # 二分类随机基线等于正类比例；裁掉基线以下没有实际信息的空白区域。
    ax_pr.set(xlim=(0, 1), ylim=(y.mean(), 1), xlabel="Recall", ylabel="Precision")
    ax_pr.set_title("OOF PR curves", loc="left", fontsize=7.5, fontweight="bold")
    ax_pr.set_box_aspect(1)

    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100,
                cbar=False, square=True, linewidths=0.65, linecolor="white",
                annot_kws={"fontsize": 7}, ax=ax_cm)
    ax_cm.set_anchor("N")
    ax_cm.set_xticklabels(["Pred. 0", "Pred. 1"], fontsize=5.8)
    ax_cm.set_yticklabels(["True 0", "True 1"], fontsize=5.8)
    ax_cm.tick_params(axis="both", length=0, pad=1.5)
    ax_cm.set_xlabel("Predicted class", fontsize=6.5, labelpad=3)
    ax_cm.set_ylabel("True class", fontsize=6.5, labelpad=3)
    ax_cm.set_title("Confusion matrix (%)", loc="left", fontsize=7.5,
                    pad=4, fontweight="bold")

    ax_notes.axis("off")
    ax_notes.set_title("Model comparison", loc="left", fontsize=7.5, fontweight="bold")
    ax_notes.legend(roc_handles, roc_labels, loc="upper left", fontsize=5.2, ncol=1, handlelength=2.2, labelspacing=0.65, frameon=False)
    ax_notes.text(0, 0.14, "Curves use out-of-fold predictions from the same 10-fold runs.\nConfusion matrix values are row-normalized percentages at a 0.5 probability threshold.\nBoxplots show median, interquartile range and individual folds; q values are BH-adjusted.", transform=ax_notes.transAxes, fontsize=5.8, va="bottom", color="#4D535A")

    fig.text(0.5, 0.015, "All ROC and PR curves are based on OOF predictions; the highlighted red curve denotes the best OOF-AUC feature set.", ha="center", fontsize=5.8, color="#4D535A")
    save_pub(fig, "Figure_1_XGBoost_degradation_classification")


def make_extended(summary, folds, preds):
    best_pred = preds[BEST]
    y = best_pred["y_true"].astype(int).to_numpy()
    score = best_pred["y_proba"].to_numpy()
    fi_path = DATA / f"{BEST}_feature_importance.csv"
    fi = pd.read_csv(fi_path)
    name_col = fi.columns[0]
    val_col = [c for c in fi.columns if c != name_col][0]
    fi = fi.sort_values(val_col, ascending=False).head(10).sort_values(val_col)

    fig = plt.figure(figsize=(7.2, 6.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.25], hspace=0.42, wspace=0.36)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_metrics = fig.add_subplot(gs[1, :])
    add_panel_label(ax_bar, "a")
    add_panel_label(ax_heat, "b")
    add_panel_label(ax_metrics, "c")

    feature_display = {
        "F1_TPM_median": "F1 TPM median",
        "polyA_GV_median": "poly(A) GV median",
        "polyA_MI_median": "poly(A) MI median",
        "polyA_MII_median": "poly(A) MII median",
        "cds_len": "CDS length",
        "cds_gc": "CDS GC",
        "cds_au": "CDS AU",
        "cds_gcp": "CDS GC proportion",
        "utr3_len": "3′ UTR length",
        "3utr_gc": "3′ UTR GC",
        "3utr_au": "3′ UTR AU",
        "3utr_gcp": "3′ UTR GC proportion",
    }
    feature_names = [feature_display.get(str(x), str(x).replace("_", " ")) for x in fi[name_col]]
    values = fi[val_col].astype(float).to_numpy()
    bars = ax_bar.barh(
        feature_names, values,
        color="#5D7F9E", edgecolor="white", linewidth=0.3,
    )
    ax_bar.set_xlabel("Mean gain importance")
    ax_bar.set_ylabel("")
    ax_bar.tick_params(axis="y", labelsize=5.2, length=0, pad=2)
    ax_bar.set_title(f"Top features: {DISPLAY[BEST]}", loc="left", fontsize=7.5, fontweight="bold")
    ax_bar.set_axisbelow(True)
    ax_bar.grid(axis="x", color="#E5E9EC", linewidth=0.45)

    # Label each bar with its model-reported gain importance. The offset and
    # right-hand margin scale with the largest value to prevent clipping.
    max_value = float(values.max())
    label_offset = max(max_value * 0.018, 0.001)
    for bar, value in zip(bars, values):
        ax_bar.text(
            value + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left", va="center", fontsize=5.2, color="#30363B",
            clip_on=False,
        )
    ax_bar.set_xlim(0, max_value * 1.18)

    auc_wide = folds.pivot(index="fold", columns="combo", values="auc")[COMBO_ORDER]
    sns.heatmap(auc_wide.T, cmap="Blues", vmin=0.60, vmax=0.90, linewidths=0.35, linecolor="white", cbar_kws={"label": "AUC", "shrink": 0.82}, ax=ax_heat)
    ax_heat.set_yticklabels(["Base", "k-mer", "Base+k", "Base+m6A", "Base+F1", "Base+polyA+F1", "...+codon", "All scalar"], fontsize=5.5)
    ax_heat.set_xticklabels([f"{i+1}" for i in range(10)], fontsize=5.5)
    ax_heat.set_xlabel("Cross-validation fold")
    ax_heat.set_ylabel("")
    ax_heat.set_title("Fold-level AUC stability", loc="left", fontsize=8, fontweight="bold")

    metric_names = ["auc", "accuracy", "f1", "aupr", "mcc"]
    metric_labels = ["AUC", "Accuracy", "F1", "AUPR", "MCC"]
    med = folds.groupby("combo")[metric_names].mean().reindex(COMBO_ORDER)
    sd = folds.groupby("combo")[metric_names].std().reindex(COMBO_ORDER)
    x = np.arange(len(COMBO_ORDER))
    width = 0.15
    palette = ["#4F6D8A", "#7A9BB8", "#9EB6C8", "#C1CCD5", "#C44E52"]
    for j, (m, lab, color) in enumerate(zip(metric_names, metric_labels, palette)):
        ax_metrics.errorbar(x + (j - 2) * width, med[m], yerr=sd[m], fmt="o", ms=3.8, lw=0.8, capsize=1.8, color=color, label=lab)
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(["Base", "k-mer", "Base+k", "Base+m6A", "Base+F1", "Base+polyA+F1", "...+codon", "All scalar"], rotation=35, rotation_mode="anchor", ha="right", fontsize=6)
    ax_metrics.set_ylabel("Mean ± s.d. across folds")
    ax_metrics.set_ylim(0.12, 0.94)
    ax_metrics.legend(ncol=5, loc="upper left", fontsize=6, handletextpad=0.3, columnspacing=0.8)
    ax_metrics.set_title("Multi-metric cross-validation profile", loc="left", fontsize=8, fontweight="bold")

    fig.text(0.5, 0.015, "Feature importance is the model-reported gain importance. Heatmap and error bars use the same 10 folds as the main analysis.", ha="center", fontsize=5.8, color="#4D535A")
    save_pub(fig, "Extended_Data_Figure_1_XGBoost_diagnostics")


def make_pipeline():
    """Create a biological, task-shared schematic for the mRNA degradation study."""
    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ink, muted, blue, rose, teal, pale = "#25333D", "#687780", "#3D789F", "#C75B60", "#5B9D9A", "#EFF3F5"
    ax.text(0.045, 0.955, "mRNA degradation classification", fontsize=10, fontweight="bold", color=ink, va="top")
    ax.text(0.045, 0.92, "A shared feature-to-prediction framework for binary and three-class tasks", fontsize=6.2, color=muted, va="top")

    # Hero: stylized transcript with explicit biological landmarks.
    ax.text(0.045, 0.845, "a", fontsize=8.5, fontweight="bold", color=ink)
    ax.text(0.078, 0.845, "mRNA substrate", fontsize=7.2, fontweight="bold", color=ink)
    y = 0.765
    ax.add_patch(plt.Circle((0.095, y), 0.026, facecolor=rose, edgecolor="none"))
    ax.text(0.095, y, "5′", ha="center", va="center", fontsize=6.2, color="white", fontweight="bold")
    ax.add_patch(plt.Rectangle((0.125, y - 0.017), 0.12, 0.034, facecolor="#B7CBD6", edgecolor="none"))
    ax.add_patch(plt.Rectangle((0.245, y - 0.023), 0.26, 0.046, facecolor=blue, edgecolor="none"))
    ax.add_patch(plt.Rectangle((0.505, y - 0.017), 0.16, 0.034, facecolor="#8DB7C3", edgecolor="none"))
    ax.plot(np.linspace(0.68, 0.9, 12), np.full(12, y), color=teal, lw=2.2, solid_capstyle="round")
    ax.text(0.185, y - 0.055, "5′ UTR", ha="center", fontsize=5.5, color=muted)
    ax.text(0.375, y, "coding sequence", ha="center", va="center", fontsize=5.5, color="white", fontweight="bold")
    ax.text(0.585, y - 0.055, "3′ UTR", ha="center", fontsize=5.5, color=muted)
    ax.text(0.79, y + 0.042, "poly(A) tail", ha="center", fontsize=5.5, color=teal)
    ax.text(0.045, 0.68, "Sequence architecture and post-transcriptional signals are represented as degradation-related predictors.", fontsize=5.7, color=muted)

    # Feature representation: three biologically interpretable streams.
    ax.text(0.045, 0.615, "b", fontsize=8.5, fontweight="bold", color=ink)
    ax.text(0.078, 0.615, "Feature representation", fontsize=7.2, fontweight="bold", color=ink)
    streams = [(0.08, "Transcript architecture", "GC / length / UTR descriptors", blue), (0.36, "Poly(A) and regulatory signals", "poly(A), F1, m6A / TE", teal), (0.64, "Sequence composition", "k-mer / codon features", rose)]
    for x, title, body, color in streams:
        ax.add_patch(plt.Rectangle((x, 0.55), 0.22, 0.012, facecolor=color, edgecolor="none"))
        ax.text(x, 0.535, title, fontsize=5.8, fontweight="bold", color=ink)
        ax.text(x, 0.505, body, fontsize=5.3, color=muted)
    ax.text(0.92, 0.535, "→", fontsize=12, color=muted, ha="center")
    ax.text(0.92, 0.505, "feature sets", fontsize=5.3, color=muted, ha="center")
    ax.add_patch(FancyArrowPatch((0.52, 0.49), (0.52, 0.425), arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color="#7B8C96"))

    # Shared model representation, then explicit binary/three-class branches.
    ax.text(0.045, 0.445, "c", fontsize=8.5, fontweight="bold", color=ink)
    ax.text(0.078, 0.445, "Shared learning framework", fontsize=7.2, fontweight="bold", color=ink)
    ax.add_patch(plt.Rectangle((0.08, 0.36), 0.23, 0.055, facecolor=pale, edgecolor="#8BA2AF", linewidth=0.8))
    ax.text(0.195, 0.389, "8 feature combinations", ha="center", va="center", fontsize=6.1, fontweight="bold", color=ink)
    ax.add_patch(FancyArrowPatch((0.315, 0.388), (0.43, 0.388), arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color="#7B8C96"))
    ax.add_patch(plt.Rectangle((0.43, 0.36), 0.20, 0.055, facecolor="#F6E7E8", edgecolor=rose, linewidth=0.8))
    ax.text(0.515, 0.389, "XGBoost", ha="center", va="center", fontsize=6.5, fontweight="bold", color=ink)
    ax.add_patch(FancyArrowPatch((0.635, 0.388), (0.74, 0.47), arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color="#7B8C96"))
    ax.add_patch(FancyArrowPatch((0.635, 0.388), (0.74, 0.305), arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color="#7B8C96"))
    ax.add_patch(plt.Rectangle((0.74, 0.435), 0.21, 0.06, facecolor="#E8F0F4", edgecolor=blue, linewidth=0.8))
    ax.text(0.845, 0.465, "Binary task", ha="center", va="center", fontsize=6.2, fontweight="bold", color=ink)
    ax.text(0.845, 0.443, "stable ↔ degraded", ha="center", va="center", fontsize=5.4, color=muted)
    ax.add_patch(plt.Rectangle((0.74, 0.255), 0.21, 0.06, facecolor="#F5E9EE", edgecolor=rose, linewidth=0.8))
    ax.text(0.845, 0.285, "Three-class task", ha="center", va="center", fontsize=6.2, fontweight="bold", color=ink)
    ax.text(0.845, 0.263, "stable · intermediate · degraded", ha="center", va="center", fontsize=5.2, color=muted)
    ax.add_patch(FancyArrowPatch((0.845, 0.435), (0.79, 0.18), arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color="#82909A"))
    ax.add_patch(FancyArrowPatch((0.845, 0.255), (0.88, 0.18), arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color="#82909A"))

    # Evaluation: one common validation and readout layer for both tasks.
    ax.text(0.045, 0.19, "d", fontsize=8.5, fontweight="bold", color=ink)
    ax.text(0.078, 0.19, "Common evaluation layer", fontsize=7.2, fontweight="bold", color=ink)
    ax.plot([0.08, 0.92], [0.125, 0.125], color="#B9C7CF", lw=0.8)
    evals = [(0.16, "10-fold CV", "held-out predictions", blue), (0.37, "OOF metrics", "AUC / AUPR / F1", teal), (0.58, "Statistical tests", "paired folds + BH", rose), (0.79, "Interpretation", "importance / ROC / CM", "#8B7390")]
    for x, title, body, color in evals:
        ax.add_patch(plt.Circle((x, 0.125), 0.017, facecolor=color, edgecolor="white", linewidth=0.7, zorder=3))
        ax.text(x, 0.075, title, ha="center", fontsize=5.6, fontweight="bold", color=ink)
        ax.text(x, 0.047, body, ha="center", fontsize=5.2, color=muted)
    for x1, x2 in zip([0.16, 0.37, 0.58], [0.37, 0.58, 0.79]):
        ax.add_patch(FancyArrowPatch((x1 + 0.02, 0.125), (x2 - 0.02, 0.125), arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color="#82909A"))
    ax.text(0.045, 0.015, "The same feature and validation framework supports both label granularities, while the task head determines whether the endpoint is binary or three-class.", fontsize=5.6, color=muted)
    save_pub(fig, "Figure_S1_XGBoost_experimental_pipeline")


def main():
    summary, folds, preds = load_data()
    wide, friedman, qvals = run_tests(folds)
    make_table(summary, preds)
    make_main(summary, folds, preds, wide, friedman, qvals)
    make_extended(summary, folds, preds)
    make_pipeline()
    print(f"Wrote figures to {OUT}")
    print(f"Best OOF AUC ({BEST}): {preds[BEST].assign(y_true=preds[BEST].y_true.astype(int)).pipe(lambda d: __import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(d.y_true, d.y_proba)):.4f}")


if __name__ == "__main__":
    main()
