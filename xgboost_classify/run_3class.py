"""
三分类版本：多特征组合 + 十折交叉验证 + 结果落盘
====================================================

和 mrna_xgboost_kflod.py（二分类）的关系：
    特征怎么算、一个特征组合怎么从 block 拼出 X，这两部分逻辑和分类数无关，
    所以直接从 mrna_xgboost_kflod.py 里 import 复用（build_scalar_feature_table /
    assemble_feature_matrix），不重复写一遍容易漏改、漏同步。
    唯一和二分类不一样的地方是：
        1. 交叉验证函数换成 k.py 里的 cross_validate_xgb_3class
        2. xgb_params 的 objective 换成 'multi:softprob'，并且要传 num_class
        3. 预测概率 oof_pred_proba 是 (n_samples, num_class) 的矩阵，
           不是二分类那样一列，存CSV时要展开成每个类别一列
        4. 混淆矩阵的类别标签数量按实际 label 种类数来，不写死
"""

import json
import os
from importlib import metadata, util
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # 批处理只保存图片，避免 Tkinter 在多线程退出时崩溃
import matplotlib.pyplot as plt

from k_fold import XGB_SEARCH_SPACE, cross_validate_xgb_3class

# 特征怎么算、怎么按组合拼接，直接复用二分类脚本里的函数，避免两份代码重复维护。
from run_binary import (
    build_scalar_feature_table,
    assemble_feature_matrix,
)


# ─── 外部指定的多个特征组合 ─────────────────────────────────────
# 用法和 mrna_xgboost_kflod.py 里的 FEATURE_COMBOS 完全一样：
#   scalar_blocks 可选值见 build_feature_blocks 里的 block 名：
#   'full' 'base' 'polyA' 'f1' 'm6a' 'transposon' 'micro' 'structure' 'codon'
#   kmer: None 或 dict(k_list=...)；k-mer 使用固定完整词表，不做全数据频率筛选
FEATURE_COMBOS = {
    'base_only': {
        'scalar_blocks': ['base'],
        'kmer': None,
    },
    'only_full_kmer123': {
        'scalar_blocks': None,
        'kmer': {'k_list': (1, 2, 3)},
    },
    'base_f1': {
        'scalar_blocks': ['base', 'f1'],
        'kmer': None,
    },
    'base_polyA_f1': {
        'scalar_blocks': ['base', 'polyA', 'f1'],
        'kmer': None,
    },

    # 'base_structure': {
    #     'scalar_blocks': ['base', 'structure'],  # 二级结构先不放在这
    #     'kmer': None,
    # },
    'base_polyA_f1_codon': {
        'scalar_blocks': ['base', 'polyA', 'f1', 'codon'],
        'kmer': None,
    },
    'base_m6a_micro_transposon': {
        'scalar_blocks': ['base', 'm6a', 'micro', 'transposon'],  # 转座子元件
        'kmer': None,
    },
    'all_scalar': {
        'scalar_blocks': ['base', 'polyA', 'f1', 'm6a', 'transposon', 'micro', 'codon'],
        'kmer': None,
    },
    # 下面都其他的特征，不用放在正文里 不对，这里改不了没法指定序列
    'only_3utr_kmer123': {
        'scalar_blocks': None,
        'kmer': {'k_list': (1, 2, 3)},
        'str_type': '3utr'
    },
    'only_cds_kmer123': {
        'scalar_blocks': None,
        'kmer': {'k_list': (1, 2, 3)},
        'str_type': 'cds'
    },
    'base_full_kmer123': {
        'scalar_blocks': ['base'],
        'kmer': {'k_list': (1, 2, 3)},
        'str_type': 'seq'
    },
    'base_full_kmer1_7': {
        'scalar_blocks': ['base'],
        'kmer': {'k_list': (1, 2, 3, 4, 5, 6, 7)},
        'str_type': 'seq'
    },
    # 需要把polyA的三个数值给拆开
    'base_polyA_MII': {
        'scalar_blocks': ['base', 'polyA_MII'],
        'kmer': None,
    },
    'base_polyA_MI': {
        'scalar_blocks': ['base', 'polyA_MI'],
        'kmer': None,
    },
    'base_polyA_GV': {
        'scalar_blocks': ['base', 'polyA_GV'],
        'kmer': None,
    },

    'base_polyA_MII_F1': {
        'scalar_blocks': ['base', 'polyA_MII', 'f1'],
        'kmer': None,
    },
    'base_polyA_MI_F1': {
        'scalar_blocks': ['base', 'polyA_MI', 'f1'],
        'kmer': None,
    },
    'base_polyA_GV_F1': {
        'scalar_blocks': ['base', 'polyA_GV', 'f1'],
        'kmer': None,
    },
    # 一些其他的
    'base_polyA_f1_micro': {
        'scalar_blocks': ['base', 'polyA', 'f1', 'micro'],
        'kmer': None,
    },
    'base_polyA_f1_m6a': {
        'scalar_blocks': ['base', 'polyA', 'f1','m6a'],
        'kmer': None,
    },
    'base_polyA_f1_transposon': {
        'scalar_blocks': ['base', 'polyA', 'f1', 'transposon'],
        'kmer': None,
    },
}


# ─── Reproducible evaluation/search configuration ──────────────────────────
# Hyperparameter optimization is nested inside each outer CV split: the held-
# out test fold is never used for tuning or early stopping.
CV_N_SPLITS = 10
CV_RANDOM_STATE = 42
ENABLE_HYPERPARAMETER_SEARCH = True
OPTUNA_N_TRIALS = 30
OPTUNA_TIMEOUT = None  # seconds per outer fold; None means no time limit


def _require_optuna():
    """Fail before data processing when the requested search dependency is absent."""
    if ENABLE_HYPERPARAMETER_SEARCH and util.find_spec('optuna') is None:
        raise ImportError(
            "ENABLE_HYPERPARAMETER_SEARCH=True requires Optuna. "
            "Install it in the active environment with: pip install optuna"
        )


def _package_version(distribution_name):
    """Return a package version for the reproducibility manifest."""
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None


if __name__ == '__main__':
    _require_optuna()

    OUT_DIR = '../outputs/xgboost_3class_results'
    os.makedirs(OUT_DIR, exist_ok=True)

    # 需要分别加载全长、cds、3utr —— 和二分类脚本里的加载/合并/清洗逻辑完全一致，
    # 唯一区别是这里的 df['label'] 应该是 0/1/2 三个类别。
    df = pd.read_csv('./data/3class_tab.csv')
    df1 = pd.read_csv('./data/gene_3utr_mirna_summary.csv', encoding='gbk')
    df2 = pd.read_csv('./data/seqs_pc_3utr.csv')
    df3 = pd.read_csv('./data/seqs_pc_cds.csv')
    df4 = pd.read_csv('./data/gene_mirna_20d_features.csv')
    df5 = pd.read_csv('./data/cluster_gene_te_gene_level.tsv', sep='\t')
    # df6 = pd.read_csv('./data_v3/rna_structure_features_v3.csv')

    df = df.merge(df1, how='left', on='gene_name')
    df = df.merge(df2, how='left', on='gene_name')
    df = df.merge(df3, how='left', on='gene_name')
    df = df.merge(df4, how='left', on='gene_name')
    df = df.merge(df5, how='left', on='gene_name')
    # df = df.merge(df6, how='left', on='gene_name')

    df.dropna(subset=['seq'], inplace=True)
    df.drop(df[df['utr3_len'] == 0].index, inplace=True)
    df.drop(df[df['cds_len'] == 0].index, inplace=True)
    df.drop(df[df['cds'].str.len() % 3 != 0].index, inplace=True)
    df.reset_index(drop=True, inplace=True)  # 保证 feat_df / X / y 索引严格对齐

    y = df['label'].values  # 三分类 0/1/2
    classes = np.unique(y)
    num_class = len(classes)
    expected_classes = np.arange(3)
    if not np.array_equal(classes, expected_classes):
        raise ValueError(
            "Three-class XGBoost requires labels encoded exactly as 0, 1 and 2; "
            f"observed labels were {classes.tolist()}."
        )
    class_counts = pd.Series(y).value_counts().sort_index()
    if int(class_counts.min()) < CV_N_SPLITS:
        raise ValueError(
            f"Each class needs at least {CV_N_SPLITS} samples for "
            f"{CV_N_SPLITS}-fold stratified CV; counts were "
            f"{class_counts.to_dict()}."
        )

    # 所有候选标量特征只算这一次，供下面所有组合复用
    feat_df, block_to_cols = build_scalar_feature_table(df)

    xgb_params = dict(
        objective='multi:softprob',
        num_class=num_class,
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.03,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        eval_metric='mlogloss',
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )

    # Save the complete analysis contract once so every result can be traced
    # to the same folds, seed, model settings and software environment.
    run_config = {
        'task': 'three_class_classification',
        'outer_cv': {
            'splitter': 'StratifiedKFold',
            'n_splits': CV_N_SPLITS,
            'shuffle': True,
            'random_state': CV_RANDOM_STATE,
        },
        'hyperparameter_search': {
            'enabled': ENABLE_HYPERPARAMETER_SEARCH,
            'method': 'Optuna TPESampler',
            'objective': 'macro one-vs-rest ROC AUC',
            'n_trials_per_outer_fold': OPTUNA_N_TRIALS,
            'timeout_seconds_per_outer_fold': OPTUNA_TIMEOUT,
            'test_fold_used_for_tuning': False,
            'search_space': XGB_SEARCH_SPACE,
        },
        'class_distribution': {
            str(int(label)): int(count)
            for label, count in class_counts.items()
        },
        'xgb_base_params': xgb_params,
        'feature_combinations': list(FEATURE_COMBOS),
        'software_versions': {
            'numpy': _package_version('numpy'),
            'pandas': _package_version('pandas'),
            'scikit-learn': _package_version('scikit-learn'),
            'xgboost': _package_version('xgboost'),
            'optuna': _package_version('optuna'),
        },
    }
    with open(os.path.join(OUT_DIR, 'run_config.json'), 'w', encoding='utf-8') as file:
        json.dump(run_config, file, ensure_ascii=False, indent=2)

    summary_rows = []

    for combo_name, combo_cfg in FEATURE_COMBOS.items():
        print(f"\n{'='*60}\n运行特征组合: {combo_name}\n{'='*60}")

        X, fname = assemble_feature_matrix(df, feat_df, block_to_cols, combo_cfg)
        print(f"[{combo_name}] 特征维度: {X.shape}")

        results = cross_validate_xgb_3class(  # 三分类十折交叉验证
            X, y,
            n_splits=CV_N_SPLITS,
            random_state=CV_RANDOM_STATE,
            xgb_params=xgb_params,
            verbose=1,
            optimize_hyperparameters=ENABLE_HYPERPARAMETER_SEARCH,
            n_trials=OPTUNA_N_TRIALS,
            optuna_timeout=OPTUNA_TIMEOUT,
        )

        # 保存每个外层折的最佳参数以及全部trial，便于复现和撰写Methods。
        optuna_path = os.path.join(OUT_DIR, f'{combo_name}_optuna_results.json')
        with open(optuna_path, 'w', encoding='utf-8') as file:
            json.dump(
                results['optimization_results'],
                file,
                ensure_ascii=False,
                indent=2,
            )

        # ---- 存储该组合的逐样本预测结果：真值、预测标签、每个类别的预测概率、所属折号 ----
        # oof_pred_proba 形状是 (n_samples, num_class)，展开成每个类别一列，
        # 方便后续画每一类各自的 ROC/PR 曲线（一对多 one-vs-rest）。
        proba_cols = {
            f'y_proba_class{c}': results['oof_pred_proba'][:, i]
            for i, c in enumerate(classes)
        }
        pred_df = pd.DataFrame({
            'gene_name': df['gene_name'].values,
            'fold': results['oof_fold'],
            'y_true': results['y'],
            'y_pred': results['pred_y'],
            **proba_cols,
        })
        pred_path = os.path.join(OUT_DIR, f'{combo_name}_predictions.csv')
        pred_df.to_csv(pred_path, index=False)

        # ---- 存储该组合的特征重要性（十折平均） ----
        imp_df = pd.DataFrame({
            'feature': fname,
            'importance': results['avg_feature_importance'],
        }).sort_values('importance', ascending=False)
        imp_path = os.path.join(OUT_DIR, f'{combo_name}_feature_importance.csv')
        imp_df.to_csv(imp_path, index=False)

        # ---- 存储该组合每一折各自的验证集指标 ----
        fold_metrics_df = pd.DataFrame(results['fold_metrics'])
        fold_metrics_path = os.path.join(OUT_DIR, f'{combo_name}_fold_metrics.csv')
        fold_metrics_df.to_csv(fold_metrics_path, index=False)

        # ---- 存储该组合每一折各自的特征重要性（未平均，长表格式：fold, feature, importance） ----
        fold_imp_rows = []
        for fold_i, imp_arr in enumerate(results['feature_importances'], start=1):
            for feat_name, imp_val in zip(fname, imp_arr):
                fold_imp_rows.append({'fold': fold_i, 'feature': feat_name, 'importance': imp_val})
        fold_imp_df = pd.DataFrame(fold_imp_rows)
        fold_imp_path = os.path.join(OUT_DIR, f'{combo_name}_fold_feature_importance.csv')
        fold_imp_df.to_csv(fold_imp_path, index=False)

        print(f"[{combo_name}] 已保存: {pred_path}")
        print(f"[{combo_name}] 已保存: {imp_path}")
        print(f"[{combo_name}] 已保存: {fold_metrics_path}")
        print(f"[{combo_name}] 已保存: {fold_imp_path}")

        # ---- 汇总该组合的整体指标，最后统一存一份对比表 ----
        row = {
            'combo': combo_name,
            'n_features': X.shape[1],
            'oof_auc': results['oof_auc'],
            'oof_acc': results['oof_acc'],
            'oof_f1': results['oof_f1'],
            'oof_f1_w': results['oof_f1_w'],
        }
        for metric_name, (mean_v, std_v) in results['summary'].items():
            row[f'{metric_name}_mean'] = mean_v
            row[f'{metric_name}_std'] = std_v
        summary_rows.append(row)

        # ---- 每个组合各自的混淆矩阵图 ----
        cm = confusion_matrix(results['y'], results['pred_y'])
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'pred_{c}' for c in classes],
                    yticklabels=[f'true_{c}' for c in classes])
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix - {combo_name}')
        plt.savefig(os.path.join(OUT_DIR, f'{combo_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ---- 每个组合各自的Top10重要性图 ----
        top10 = imp_df.head(10)
        plt.figure(figsize=(10, 8))
        plt.barh(top10['feature'], top10['importance'], color='steelblue')
        plt.xlabel('Importance')
        plt.title(f'Top 10 Feature Importance - {combo_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'{combo_name}_top10_feature_importance.png'), dpi=150)
        plt.close()

    # ---- 所有组合跑完后，存一份总对比表 ----
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, 'all_combos_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n所有特征组合已跑完，汇总对比表已保存: {summary_path}")
    print(summary_df)
