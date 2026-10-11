from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "xgboost_3class_results"
OUT = ROOT / "outputs/xgboost_3class_results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})
sns.set_theme(style="white", context="paper")

COMBOS = [
    "base_only", "only_full_kmer123", "base_full_kmer123", "base_m6a_micro_transposon",
    "base_f1", "base_polyA_f1", "base_polyA_f1_codon", "all_scalar",
]
DISPLAY = {
    "base_only": "Base only", "only_full_kmer123": "k-mer only", "base_full_kmer123": "Base + k-mer",
    "base_m6a_micro_transposon": "Base + m6A + TE", "base_f1": "Base + F1",
    "base_polyA_f1": "Base + poly(A) + F1", "base_polyA_f1_codon": "Base + poly(A) + F1 + codon",
    "all_scalar": "All scalar features",
}
COLORS = {
    "base_only": "#A9B6C6", "only_full_kmer123": "#C7CDD4", "base_full_kmer123": "#7E9AB7",
    "base_m6a_micro_transposon": "#91AFC9", "base_f1": "#547B9E", "base_polyA_f1": "#C44E52",
    "base_polyA_f1_codon": "#8C2D3A", "all_scalar": "#B8860B",
}
BEST = "base_polyA_f1"
CLASS_NAMES = ["Class 0", "Class 1", "Class 2"]


def save_pub(fig, stem, close=True):
    """将图片保存为多种格式，并按需关闭 Figure。"""
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    # fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    # fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    if close:
        plt.close(fig)


def add_label(ax, label):
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def load_data():
    summary = pd.read_csv(DATA / "all_combos_summary.csv")
    summary["combo"] = summary["combo"].astype(str).str.strip()
    folds = []
    preds = {}
    for combo in COMBOS:
        df = pd.read_csv(DATA / f"{combo}_fold_metrics.csv")
        df["combo"] = combo
        folds.append(df)
        preds[combo] = pd.read_csv(DATA / f"{combo}_predictions.csv")
    return summary, pd.concat(folds, ignore_index=True), preds


def bh_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return np.clip(out, 0, 1)


def run_stats(folds):
    wide = folds.pivot(index="fold", columns="combo", values="mutil_auc")[COMBOS]
    fr = friedmanchisquare(*[wide[c].to_numpy() for c in COMBOS])
    raw = []
    for combo in COMBOS:
        if combo == BEST:
            continue
        result = wilcoxon(wide[BEST], wide[combo], alternative="two-sided", method="auto")
        raw.append((combo, float(result.statistic), float(result.pvalue)))
    qvals = bh_adjust([r[2] for r in raw])
    records = [{"test": "Friedman", "comparison": "all combinations", "statistic": fr.statistic, "p_value": fr.pvalue, "q_value": np.nan}]
    for (combo, stat, p), q in zip(raw, qvals):
        records.append({"test": "Wilcoxon signed-rank", "comparison": f"{BEST} vs {combo}", "statistic": stat, "p_value": p, "q_value": q})
    pd.DataFrame(records).to_csv(OUT / "statistical_tests_3class.csv", index=False)
    return wide, fr, {combo: q for (combo, _, _), q in zip(raw, qvals)}


def macro_curves(y, proba):
    y_bin = label_binarize(y, classes=[0, 1, 2])
    grid = np.linspace(0, 1, 201)
    roc_parts, pr_parts = [], []
    for k in range(3):
        fpr, tpr, _ = roc_curve(y_bin[:, k], proba[:, k])
        roc_parts.append(np.interp(grid, fpr, tpr))
        precision, recall, _ = precision_recall_curve(y_bin[:, k], proba[:, k])
        order = np.argsort(recall)
        pr_parts.append(np.interp(grid, recall[order], precision[order]))
    return grid, np.mean(roc_parts, axis=0), np.mean(pr_parts, axis=0)


def q_label(q):
    if q < 0.001:
        return "q < 0.001"
    if q < 0.01:
        return "q < 0.01"
    if q < 0.05:
        return "q < 0.05"
    return "n.s."

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def make_table(summary, preds):
    """单独导出三分类性能表，不再将表格嵌入组合图。"""
    best = preds[BEST]
    y_best = best["y_true"].astype(int).to_numpy()
    tab = summary.sort_values("oof_f1", ascending=False).copy()

    table_data = pd.DataFrame({
        "Feature set": tab["combo"].map(DISPLAY),
        "OOF AUC": tab["oof_auc"].map(lambda x: f"{x:.3f}"),
        "OOF accuracy": tab["oof_acc"].map(lambda x: f"{x:.3f}"),
        "Macro F1": tab["oof_f1"].map(lambda x: f"{x:.3f}"),
        "Weighted F1": tab["oof_f1_w"].map(lambda x: f"{x:.3f}"),
        "10-fold macro AUC (mean +/- s.d.)": tab.apply(
            lambda r: f"{r.mutil_auc_mean:.3f} +/- {r.mutil_auc_std:.3f}", axis=1
        ),
    })
    table_data.to_csv(
        OUT / "Table_1_XGBoost_3class_degradation_classification.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.axis("off")
    ax.set_title(
        "XGBoost three-class classification across degradation-related feature sets",
        loc="left", fontsize=8, pad=5, fontweight="bold",
    )
    rows = table_data.astype(str).values.tolist()
    table = ax.table(
        cellText=rows, colLabels=table_data.columns.tolist(),
        cellLoc="center", colLoc="center", bbox=[0, 0.13, 1, 0.77],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.1)
    widths = [0.29, 0.12, 0.14, 0.12, 0.14, 0.22]
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D8DDE3")
        cell.set_linewidth(0.45)
        cell.set_facecolor(
            "#EEF1F4" if r == 0
            else ("#F8E6E7" if rows[r - 1][0] == DISPLAY[BEST] else "white")
        )
        if r == 0:
            cell.set_text_props(weight="bold")
        if c < len(widths):
            cell.set_width(widths[c])

    best_oof_f1 = float(summary.loc[summary["combo"] == BEST, "oof_f1"].iloc[0])
    upper_bound = float(summary.loc[summary["combo"] == "all_scalar", "oof_f1"].iloc[0])
    ax.text(
        0, 0.04,
        f"Selected parsimonious model: {DISPLAY[BEST]}; OOF macro F1 = {best_oof_f1:.3f}; "
        f"all-scalar upper bound = {upper_bound:.3f} (n = {len(y_best)} samples)",
        transform=ax.transAxes, fontsize=6.2, color="#40474F",
    )
    save_pub(fig, "Table_1_XGBoost_3class_degradation_classification")


def make_main(summary, folds, preds, wide, friedman, qvals):
    """绘制不含汇总表的 XGBoost 三分类结果主图（面板 a-e）。"""

    # ==================== 1. 创建整张组合图并划分各面板 ====================
    fig = plt.figure(figsize=(8.4, 6.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 0.95, 0.20], hspace=0.54)
    top_gs = gs[0].subgridspec(1, 3, width_ratios=[1.97, 1.0, 1.0], wspace=0.30)
    middle_gs = gs[1].subgridspec(1, 2, wspace=0.42)

    # 第一行：箱线图、ROC 曲线和 PR 曲线；第二行：特征重要性和相关性热图；
    # 第三行：统一图例与说明。性能汇总表由 make_table() 单独导出。
    ax_box = fig.add_subplot(top_gs[0, 0])
    ax_roc = fig.add_subplot(top_gs[0, 1])
    ax_pr = fig.add_subplot(top_gs[0, 2])
    ax_top = fig.add_subplot(middle_gs[0, 0])
    ax_corr = fig.add_subplot(middle_gs[0, 1])
    ax_legend = fig.add_subplot(gs[2, 0])
    for ax, label in zip([ax_box, ax_roc, ax_pr, ax_top, ax_corr], "abcde"):
        add_label(ax, label)

    # ==================== 2. 面板 a：10 折交叉验证 AUC 箱线图 ====================
    # 箱线图展示各组合的 AUC 分布，散点展示每一折的实际结果。
    long = folds.copy()
    long["display"] = long["combo"].map(DISPLAY)
    sns.boxplot(data=long, x="display", hue="display", y="mutil_auc", order=[DISPLAY[c] for c in COMBOS], palette=[COLORS[c] for c in COMBOS],
                width=0.62, fliersize=0, linewidth=0.65, ax=ax_box, legend=False, dodge=False, )
    sns.stripplot(data=long, x="display", y="mutil_auc", order=[DISPLAY[c] for c in COMBOS], color="#30343B", size=2.1, alpha=0.65, jitter=0.08, ax=ax_box)
    ax_box.set_ylabel("10-fold macro AUC")
    ax_box.set_xlabel("")
    short = ["Base", "k-mer", "Base+k", "Base+m6A", "Base+F1", "Base+polyA+F1", "...+codon", "All scalar"]
    ax_box.set_xticks(np.arange(len(COMBOS)))
    ax_box.set_xticklabels(short, rotation=45, rotation_mode="anchor", ha="right", fontsize=5.3)
    ax_box.set_ylim(0.58, 0.90)
    # Match the binary panel with explicit 0.05-spaced macro-AUC ticks so
    # values such as 0.60 and 0.80 are immediately readable.
    auc_ticks = np.arange(0.60, 0.901, 0.05)
    ax_box.set_yticks(auc_ticks)
    ax_box.set_yticklabels([f"{value:.2f}" for value in auc_ticks], fontsize=6.0)
    ax_box.set_axisbelow(True)
    ax_box.grid(axis="y", color="#DDE2E7", linewidth=0.45, alpha=0.75)
    ax_box.text(0.02, 0.98, f"Friedman P = {friedman.pvalue:.2e}", transform=ax_box.transAxes, ha="left", va="top", fontsize=5.7)
    for i, combo in enumerate(COMBOS):
        if combo != BEST:
            # 标注最佳组合与其他组合比较后的 BH 校正 q 值。
            ax_box.text(i, 0.585, q_label(qvals[combo]), ha="center", va="bottom", rotation=90, rotation_mode="anchor", fontsize=5.2, color="#5A6068")

    # ==================== 3. 面板 b、c：OOF ROC 曲线与 PR 曲线 ====================
    # 对每个特征组合计算宏平均曲线，并用更粗的线突出最佳组合。
    handles, labels = [], []
    for combo in COMBOS:
        pred = preds[combo]
        yy = pred["y_true"].astype(int).to_numpy()
        pp = pred[["y_proba_class0", "y_proba_class1", "y_proba_class2"]].to_numpy()
        grid, macro_tpr, macro_prec = macro_curves(yy, pp)
        macro_auc = np.trapezoid(macro_tpr, grid)
        (line,) = ax_roc.plot(grid, macro_tpr, color=COLORS[combo], lw=1.35 if combo != BEST else 2.2, alpha=0.9)
        ax_pr.plot(grid, macro_prec, color=COLORS[combo], lw=1.35 if combo != BEST else 2.2, alpha=0.9)
        handles.append(line)
        labels.append(f"{DISPLAY[combo]} (macro AUC = {macro_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="#B8BEC6", lw=0.8, ls="--")
    ax_roc.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="Macro true-positive rate")
    ax_roc.set_title("Macro OOF ROC curves", loc="left", fontsize=7.5, fontweight="bold")
    ax_roc.set_box_aspect(1)
    ax_pr.axhline(1 / 3, color="#B8BEC6", lw=0.8, ls="--")
    # 三分类随机基线为 1/3；裁掉其下方没有实际信息的空白区域。
    ax_pr.set(xlim=(0, 1), ylim=(1 / 3, 1), xlabel="Recall", ylabel="Macro precision")
    ax_pr.set_title("Macro OOF PR curves", loc="left", fontsize=7.5, fontweight="bold")
    ax_pr.set_box_aspect(1)

    # ==================== 4. 面板 d：最佳模型前 10 个重要特征 ====================
    # 先选出重要性最高的 10 个特征，再按重要性升序排列以绘制水平条形图。
    fi = pd.read_csv(DATA / f"{BEST}_feature_importance.csv")
    fi = fi.sort_values("importance", ascending=False).head(10).sort_values("importance")
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
    feature_names = [feature_display.get(str(x), str(x).replace("_", " ")) for x in fi["feature"]]
    values = fi["importance"].astype(float).to_numpy()
    bars = ax_top.barh(feature_names, values, color="#8069A1", edgecolor="white", linewidth=0.3)
    ax_top.set_yticks(np.arange(len(fi)))
    ax_top.set_yticklabels(feature_names, fontsize=5.2)
    ax_top.set_xlabel("Mean gain importance")
    ax_top.set_ylabel("")
    ax_top.tick_params(axis="y", length=0, pad=2)
    ax_top.set_title(f"Top features: {DISPLAY[BEST]}", loc="left", fontsize=7.5, fontweight="bold")
    ax_top.set_axisbelow(True)
    ax_top.grid(axis="x", color="#E5E9EC", linewidth=0.45)

    max_value = float(values.max())
    label_offset = max(max_value * 0.018, 0.001)
    for bar, value in zip(bars, values):
        ax_top.text(
            value + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left", va="center", fontsize=5.2, color="#30363B",
            clip_on=False,
        )
    ax_top.set_xlim(0, max_value * 1.18)

    # ==================== 5. 面板 e：特征间 Pearson 相关性热图 ====================
    corr = pd.read_csv(ROOT / "data_final0809" / "feature_correlation_matrix.csv", index_col=0)
    corr.index = corr.index.astype(str).str.replace("_", " ", regex=False)
    corr.columns = corr.columns.astype(str).str.replace("_", " ", regex=False)
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, linewidths=0.25, linecolor="white", cbar_kws={"label": "Pearson r", "shrink": 0.80}, ax=ax_corr)
    ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=45, rotation_mode="anchor", ha="right", fontsize=5.2)
    ax_corr.set_yticklabels(ax_corr.get_yticklabels(), fontsize=5.2)
    ax_corr.set_xlabel("")
    ax_corr.set_ylabel("")
    ax_corr.set_title("Feature correlation structure", loc="left", fontsize=7.5, fontweight="bold")

    # ==================== 6. 添加统一图例和整图说明 ====================
    ax_legend.axis("off")
    ax_legend.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=5.1, handlelength=2.0, labelspacing=0.35, columnspacing=0.8, frameon=False)
    ax_legend.text(0.5, 0.04, "Macro one-vs-rest curves; class labels follow source y_true encoding (0, 1 and 2); percentage confusion matrix is provided in Extended Data.", transform=ax_legend.transAxes, fontsize=5.1, ha="center", color="#4D535A")
    fig.text(0.5, 0.005, "Boxplots show median, interquartile range and individual folds; pairwise tests compare the best feature set with each alternative using BH-adjusted q values.", ha="center", fontsize=5.7, color="#4D535A")

    # ==================== 7. 保存并显示完整的组合图 ====================
    # 先保存但暂不关闭 Figure，显示完成后再释放内存。
    save_pub(fig, "Figure_1_XGBoost_3class_degradation_classification", close=False)
    plt.show()
    plt.close(fig)


def make_extended(summary, folds, preds):
    auc_wide = folds.pivot(index="fold", columns="combo", values="mutil_auc")[COMBOS]
    metric_names = ["mutil_auc", "accuracy", "f1", "f1_weight", "logloss"]
    metric_labels = ["Macro AUC", "Accuracy", "Macro F1", "Weighted F1", "Logloss"]
    best = preds[BEST]
    y_best = best["y_true"].astype(int).to_numpy()
    cm = confusion_matrix(y_best, best["y_pred"].astype(int), labels=[0, 1, 2], normalize="true") * 100
    fig = plt.figure(figsize=(7.2, 6.7))
    # 顶行采用 2:1 布局：折次热图占两列，3×3 混淆矩阵只占一列。
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.25], hspace=0.43, wspace=0.42)
    ax_heat = fig.add_subplot(gs[0, :2])
    ax_cm = fig.add_subplot(gs[0, 2])
    ax_metrics = fig.add_subplot(gs[1, :])
    for ax, label in zip([ax_heat, ax_cm, ax_metrics], "abc"):
        add_label(ax, label)
    sns.heatmap(auc_wide.T, cmap="Blues", vmin=0.58, vmax=0.90, linewidths=0.35, linecolor="white", cbar_kws={"label": "Macro AUC", "shrink": 0.82}, ax=ax_heat)
    ax_heat.set_yticklabels(["Base", "k-mer", "Base+k", "Base+m6A", "Base+F1", "Base+polyA+F1", "...+codon", "All scalar"], fontsize=5.4)
    ax_heat.set_xticks(np.arange(10) + 0.5)
    ax_heat.set_xticklabels([str(i + 1) for i in range(10)], fontsize=5.4)
    ax_heat.set_xlabel("Cross-validation fold")
    ax_heat.set_ylabel("")
    ax_heat.set_title("Fold-level macro AUC stability", loc="left", fontsize=8, fontweight="bold")
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Purples", vmin=0, vmax=100,
                cbar=False, square=True, linewidths=0.6, linecolor="white",
                annot_kws={"fontsize": 6.5}, ax=ax_cm)
    ax_cm.set_anchor("N")
    ax_cm.set_xticklabels(CLASS_NAMES, fontsize=5.2)
    ax_cm.set_yticklabels(CLASS_NAMES, fontsize=5.2)
    ax_cm.tick_params(axis="both", length=0, pad=1.2)
    ax_cm.set_xlabel("Predicted", fontsize=6, labelpad=3)
    ax_cm.set_ylabel("True", fontsize=6, labelpad=3)
    ax_cm.set_title("Confusion matrix (%)", loc="left", fontsize=7.5,
                    pad=4, fontweight="bold")
    means = folds.groupby("combo")[metric_names].mean().reindex(COMBOS)
    sds = folds.groupby("combo")[metric_names].std().reindex(COMBOS)
    x = np.arange(len(COMBOS))
    offsets = np.linspace(-0.14, 0.14, len(metric_names))
    palette = ["#4F6D8A", "#7A9BB8", "#9EB6C8", "#C1CCD5", "#C44E52"]
    for off, metric, label, color in zip(offsets, metric_names, metric_labels, palette):
        ax_metrics.errorbar(x + off, means[metric], yerr=sds[metric], fmt="o", ms=3.7, lw=0.8, capsize=1.7, color=color, label=label)
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(["Base", "k-mer", "Base+k", "Base+m6A", "Base+F1", "Base+polyA+F1", "...+codon", "All scalar"], rotation=35, rotation_mode="anchor", ha="right", fontsize=6)
    ax_metrics.set_ylabel("Mean +/- s.d. across folds")
    ax_metrics.set_ylim(0.0, 1.0)
    ax_metrics.legend(ncol=5, loc="upper left", fontsize=5.8, handletextpad=0.3, columnspacing=0.8)
    ax_metrics.set_title("Multi-metric three-class validation profile", loc="left", fontsize=8, fontweight="bold")
    fig.text(0.5, 0.015, "Feature importance uses model-reported gain importance; error bars show fold-level standard deviations.", ha="center", fontsize=5.8, color="#4D535A")
    save_pub(fig, "Extended_Data_Figure_1_XGBoost_3class_diagnostics")


def main():
    summary, folds, preds = load_data()
    wide, friedman, qvals = run_stats(folds)
    make_table(summary, preds)  # 独立性能表
    make_main(summary, folds, preds, wide, friedman, qvals)  # 主图画图
    make_extended(summary, folds, preds)  # 附图画图
    print(f"Wrote three-class figures to {OUT}")
    print(f"Selected parsimonious feature set: {BEST}; all_scalar retained as performance upper bound")


if __name__ == "__main__":
    main()
