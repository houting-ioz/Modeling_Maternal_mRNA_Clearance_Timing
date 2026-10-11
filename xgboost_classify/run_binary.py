import json
import os
import xgboost as xgb
import numpy as np
import pandas as pd
from itertools import product
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # 批处理只保存图片，避免 Tkinter 在多线程退出时崩溃
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

from k_fold import cross_validate_xgb

# ─── 1. k-mer特征提取 ───────────────────────────────────────────
def extract_kmer_features(seq, k_list=(1, 2, 3)):
    """提取多阶k-mer频率，返回固定长度向量"""
    seq = seq.upper().replace('T', 'U')
    bases = ['A', 'U', 'G', 'C']
    features = []

    for k in k_list:
        kmers = [''.join(p) for p in product(bases, repeat=k)]
        counts = {km: 0 for km in kmers}
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if kmer in counts:
                counts[kmer] += 1
        total = sum(counts.values()) + 1e-9
        features += [v / total for v in counts.values()]

    return features  # 4 + 16 + 64 = 84维


from collections import Counter

STOP_CODONS = {'UAA', 'UAG', 'UGA', 'TAA', 'TAG', 'TGA'}


def extract_codon_frequency(cds_seq):
    """
    提取密码子频率，排除终止密码子，共返回61维特征。

    Parameters
    ----------
    cds_seq : str
        必须是纯CDS/ORF序列，从起始密码子(ATG)开始，长度是3的整数倍。
        不能包含UTR区域，否则阅读框会错位。
    """
    seq = cds_seq.upper().replace('T', 'U')  # 统一成RNA字母表，和你之前的k-mer函数保持一致

    if len(seq) % 3 != 0:
        raise ValueError(f"序列长度 {len(seq)} 不是3的倍数，可能没有正确截取CDS区域")

    # 非重叠地按3个碱基切分（这是和滑窗k-mer最本质的区别）
    codons = [seq[i:i + 3] for i in range(0, len(seq), 3)]

    counts = Counter(codons)
    # 排除终止密码子
    for stop in STOP_CODONS:
        counts.pop(stop, None)

    # 生成61个非终止密码子的固定顺序列表，保证特征顺序稳定
    bases = ['A', 'U', 'G', 'C']
    all_codons = [a + b + c for a in bases for b in bases for c in bases]
    non_stop_codons = [c for c in all_codons if c not in STOP_CODONS]  # 61个

    total = sum(counts.get(c, 0) for c in non_stop_codons) + 1e-9
    freq = [counts.get(c, 0) / total for c in non_stop_codons]

    return freq, non_stop_codons  # 返回频率list，以及对应的密码子名字列表


# ─── 2. 特征块定义 ──────────────────────────────────────────────
# 改动说明：
# 原来的 build_feature_vector 是"手动改代码 -> 决定用哪些特征"，每次只能跑一种组合。
# 现在拆成两步：
#   1) build_feature_blocks(row)：对一行样本，把所有"候选"特征都算出来，
#      按语义分组（block），每组一个名字，返回 {block名: (数值list, 特征名list)}。
#      这一步只在数据加载时对整张表跑一次，和要跑多少个特征组合无关。
#   2) 下面的 FEATURE_COMBOS 在外部声明"这次要跑哪些组合"，每个组合只是
#      从已经算好的 block 里挑几个名字拼起来，不需要重新计算，速度很快。
#
# 注意：原代码里 base_name 写的是 'polyA_GV_median'，但实际算的数值是
# row['polyA_MII_median']（看 numeric_feats 里的注释掉的写法），命名和实际取值
# 对不上，是个笔误。下面按"实际使用的列"（polyA_MII_median）来正确命名，
# 如果你本意是要用 GV 而不是 MII，请把 blocks['polyA'] 里的列名换回来。
def build_feature_blocks(row):
    """
    计算单个样本的全部候选特征块。
    返回: dict[block_name] = (values: list[float], names: list[str])
    """
    seq = row['seq'].upper().replace('T', 'U')  # cdna -> mRNA，全长
    cds = row['cds'].upper().replace('T', 'U')
    utr3 = row['3utr'].upper().replace('T', 'U')

    blocks = {}

    # ---- 全长序列 GC含量/偏度 ----（原代码算了但没有拼进最终特征，这里作为可选block暴露出来）
    gc_content0 = (seq.count('G') + seq.count('C')) / (len(seq) + 1e-9)
    au_skew0 = (seq.count('A') - seq.count('U')) / (seq.count('A') + seq.count('U') + 1e-9)
    gc_skew0 = (seq.count('G') - seq.count('C')) / (seq.count('G') + seq.count('C') + 1e-9)
    blocks['full'] = (
        [gc_content0, au_skew0, gc_skew0],
        ['full_gc', 'full_au_skew', 'full_gc_skew'],
    )

    # ---- CDS / 3'UTR GC含量、偏度、长度 ----（原代码里始终使用的一组基础特征）
    gc_content1 = (cds.count('G') + cds.count('C')) / (len(cds) + 1e-9)
    au_skew1 = (cds.count('A') - cds.count('U')) / (cds.count('A') + cds.count('U') + 1e-9)
    gc_skew1 = (cds.count('G') - cds.count('C')) / (cds.count('G') + cds.count('C') + 1e-9)

    gc_content2 = (utr3.count('G') + utr3.count('C')) / (len(utr3) + 1e-9)
    au_skew2 = (utr3.count('A') - utr3.count('U')) / (utr3.count('A') + utr3.count('U') + 1e-9)
    gc_skew2 = (utr3.count('G') - utr3.count('C')) / (utr3.count('G') + utr3.count('C') + 1e-9)

    blocks['base'] = (
        [gc_content1, au_skew1, gc_skew1, gc_content2, au_skew2, gc_skew2,
         np.log1p(row['cds_len']), np.log1p(row['utr3_len'])],
        ['cds_gc', 'cds_au', 'cds_gcp', '3utr_gc', '3utr_au', '3utr_gcp', 'cds_len', 'utr3_len'],
    )

    # ---- polyA ----
    blocks['polyA'] = ([np.log1p(row['polyA_MII_median']), np.log1p(row['polyA_GV_median']), np.log1p(row['polyA_MI_median'])],
                       ['polyA_MII_median', 'polyA_GV_median', 'polyA_MI_median'])
    # 单个的poly A
    blocks['polyA_MII'] = ([np.log1p(row['polyA_MII_median'])], ['polyA_MI_median'])
    blocks['polyA_GV'] = ([np.log1p(row['polyA_GV_median'])], ['polyA_GV_median'])
    blocks['polyA_MI'] = ([np.log1p(row['polyA_MI_median'])], ['polyA_MI_median'])

    # ---- 翻译效率 F1 TPM ----
    blocks['f1'] = ([np.log1p(row['F1_TPM_median'])], ['F1_TPM_median'])

    # ---- m6A 位点数 ----
    blocks['m6a'] = ([row['m6A_site_count']], ['m6A_site_count'])

    # ---- 转座元件个数 ----
    blocks['transposon'] = (
        [row['LINE_n'], row['SINE_n'], row['LTR_n'], row['DNA_n']],
        ['LINE_n', 'SINE_n', 'LTR_n', 'DNA_n'],
    )

    # ---- microRNA 结合位点强度 ----
    blocks['micro'] = (np.log1p([row['max_site_strength']]), ['max_site_strength'])

    # ---- RNA 二级结构 ----
    # blocks['structure'] = (
    #     [row['mfe'], row['pairing_ratio'],
    #      row['local_mean_pair_prob'], row['local_max_pair_prob'], row['used_local_folding'],
    #      row['win_mfe_mean'], row['win_mfe_std'], row['win_mfe_min'], row['win_mfe_max'],
    #      row['win_pairing_ratio_mean'], row['win_pairing_ratio_std']],
    #     ['mfe', 'pairing_ratio', 'local_mean_pair_prob', 'local_max_pair_prob', 'used_local_folding',
    #      'win_mfe_mean', 'win_mfe_std', 'win_mfe_min', 'win_mfe_max',
    #      'win_pairing_ratio_mean', 'win_pairing_ratio_std'],
    # )

    # ---- CDS 密码子频率（61维）----
    freq, non_stop_codons = extract_codon_frequency(row['cds'])
    blocks['codon'] = (freq, [f'codon_{c}' for c in non_stop_codons])

    return blocks  # 返回的字典？


def build_scalar_feature_table(df):
    """
    对整张表跑一次 build_feature_blocks，把所有候选标量特征（不含k-mer）
    横向拼成一张宽表 feat_df（index 和 df 对齐），并返回每个 block 对应的列名，
    方便后面按组合挑选列，而不用每个特征组合都重新算一遍。
    """
    rows = []
    block_to_cols = None
    for _, row in df.iterrows():
        blocks = build_feature_blocks(row)  # 这里是拿到所有的数值特征
        if block_to_cols is None:
            block_to_cols = {b: names for b, (vals, names) in blocks.items()}  # 拿到数值特征的列名
        row_values = {}
        for _, (vals, names) in blocks.items():
            row_values.update(dict(zip(names, vals)))
        rows.append(row_values)
    feat_df = pd.DataFrame(rows, index=df.index)
    return feat_df, block_to_cols  # 加上了密码子频率


# from imblearn.over_sampling import RandomOverSampler
# from sklearn.preprocessing import RobustScaler
# from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

def compress_with_svdcompress_with_svd(X_sparse, n_components=100, random_state=0):
    """
    如果裁剪词表后维度还是偏高（比如几千维），可以再用 TruncatedSVD
    （对稀疏矩阵高效，等价于对计数矩阵做 PCA）压缩到几十到一百维的
    稠密表示，通常能保留大部分方差，且下游模型（Lasso/树模型/MLP）
    吃这种稠密低维输入会更稳定。

    注意：SVD 压缩后就不再是"某个具体k-mer的频率"了，而是k-mer组合模式的
    线性组合，可解释性会下降，如果你的项目需要报告"哪个motif起作用"，
    压缩后这一步就不适合再做，直接用裁剪后的稀疏矩阵进入 Lasso 更合适
    （Lasso选出的每个系数还能对应回具体的k-mer字符串）。
    """
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_dense = svd.fit_transform(X_sparse)
    explained = svd.explained_variance_ratio_.sum()
    return X_dense, svd, explained

import scipy.sparse as sp

# k-mer 特征使用预先固定的完整词表。每条序列独立计数和归一化，
# 不使用 min_df/max_df/max_features，因此可以在交叉验证前计算。
# 缓存键同时包含数据对象、序列区域和 k 阶列表，避免全长/CDS/3'UTR串用。
_KMER_CACHE = {}


def get_kmer_features(df, str_type, k_list=(1, 2, 3)):
    """使用固定完整词表提取逐序列、逐 k 阶归一化的稀疏 k-mer 特征。"""
    str_type = 'seq' if str_type is None else str_type
    k_list = tuple(k_list)
    key = (id(df), str_type, k_list)

    if key not in _KMER_CACHE:
        if str_type not in df.columns:
            raise KeyError(f"序列列不存在: {str_type}，可选列包括: {list(df.columns)}")

        sequences = [str(seq).upper().replace('T', 'U') for seq in df[str_type]]
        bases = ('A', 'U', 'G', 'C')
        blocks = []
        names = []

        for k in k_list:
            kmers = [''.join(p) for p in product(bases, repeat=k)]
            kmer_to_col = {kmer: col for col, kmer in enumerate(kmers)}
            data = []
            indices = []
            indptr = [0]

            for seq in sequences:
                counts = Counter(
                    seq[i:i + k]
                    for i in range(len(seq) - k + 1)
                    if seq[i:i + k] in kmer_to_col
                )
                total = sum(counts.values())

                if total:
                    for kmer, count in counts.items():
                        indices.append(kmer_to_col[kmer])
                        data.append(count / total)

                indptr.append(len(data))

            block = sp.csr_matrix(
                (data, indices, indptr),
                shape=(len(sequences), len(kmers)),
                dtype=np.float64,
            )
            blocks.append(block)
            names.extend(f'k{k}_{kmer}' for kmer in kmers)

        X_kmer = sp.hstack(blocks, format='csr')
        _KMER_CACHE[key] = (X_kmer, names)

    return _KMER_CACHE[key]


def assemble_feature_matrix(df, feat_df, block_to_cols, combo):
    """
    根据一个特征组合的配置，从已经算好的 feat_df / k-mer缓存里拼出 X 和对应的特征名列表。

    combo 格式示例：
        {
            'scalar_blocks': ['base', 'polyA', 'f1'],   # 见 build_feature_blocks 里的 block 名
            'kmer': None,                                # 不用k-mer
        }
        或
        {
            'scalar_blocks': ['base'],
            'kmer': {'k_list': (1, 2, 3)},
        }
    """
    scalar_blocks = combo.get('scalar_blocks', [])
    cols = []
    names = []
    X_scalar = None
    if scalar_blocks is not None:
        for b in scalar_blocks:
            if b not in block_to_cols:
                raise KeyError(f"未知的特征block名: {b}，可选: {list(block_to_cols.keys())}")
            cols += block_to_cols[b]

        X_scalar = feat_df[cols].values if cols else None
        names = list(cols)
    # 上面拿到了指定的数值特征 下面处理k-mer的信息
    kmer_cfg = combo.get('kmer')
    # c = kmer_cfg.get('str_type')

    if kmer_cfg is not None:  # 不为None的地方要指定 str_type
        str_type = combo.get('str_type')
        str_type = 'seq' if str_type is None else str_type

        X_kmer, kmer_names = get_kmer_features(
            df,
            str_type=str_type,
            k_list=kmer_cfg.get('k_list', (1, 2, 3)),  # 后面的参数当这个值缺失的时候才是默认值，即使为nane也是有值的
        )
        names = names + list(kmer_names)
        if X_scalar is not None:
            X = sp.hstack([sp.csr_matrix(X_scalar), X_kmer], format='csr')
        else:
            X = X_kmer
    else:
        if X_scalar is None:
            raise ValueError(f"特征组合里既没有 scalar_blocks 也没有 kmer，配置为空: {combo}")
        X = X_scalar

    return X, names


# ─── 3. 外部指定的多个特征组合 ─────────────────────────────────────
# 在这里增删/修改，每一个 key 是这次实验里这个组合的名字（会用在输出文件名里），
# value 里：
#   scalar_blocks: 要拼接的标量特征block名列表，可选值见 build_feature_blocks 里的
#                  blocks['xxx'] = (...)，目前有:
#                  'full'(全长GC/偏度) 'base'(cds+3utr GC/偏度/长度，原来默认一直用的)
#                  'polyA' 'f1'(翻译效率) 'm6a' 'transposon' 'micro'(miRNA结合强度)
#                  'structure'(二级结构) 'codon'(61维密码子频率)
#   kmer: None 表示不加k-mer特征；否则传 dict 指定固定完整词表使用的 k_list
# 一次运行会把下面所有组合依次跑完。 在这修改
FEATURE_COMBOS = {
    'base_only': {
        'scalar_blocks': ['base'],
        'kmer': None,
    },
    'only_full_kmer123': {
        'scalar_blocks': None,
        'kmer': {'k_list': (1, 2, 3)},
        'str_type': 'seq'  # seq
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


if __name__ == '__main__':
    OUT_DIR = '../outputs/xgboost_classify_results'
    os.makedirs(OUT_DIR, exist_ok=True)

    # 需要分别加载全长、cds、3utr
    df = pd.read_csv('../data/binary_tab.csv')  # 全长  这里是去掉了lnc所有只有8千多
    df1 = pd.read_csv('../data/gene_3utr_mirna_summary.csv', encoding='gbk')  # microRNA结合特征
    df2 = pd.read_csv('../data/seqs_pc_3utr.csv')  # 3utr  cds应该是当时一起做了
    df3 = pd.read_csv('../data/seqs_pc_cds.csv')  # cds
    df4 = pd.read_csv('../data/gene_mirna_20d_features.csv')
    df5 = pd.read_csv('../data/cluster_gene_te_gene_level.tsv', sep='\t')
    # df6 = pd.read_csv('./data_v3/rna_structure_features_v3.csv')

    df = df.merge(df1, how='left', on='gene_name')
    df = df.merge(df2, how='left', on='gene_name')
    df = df.merge(df3, how='left', on='gene_name')
    df = df.merge(df4, how='left', on='gene_name')
    df = df.merge(df5, how='left', on='gene_name')
    # df = df.merge(df6, how='left', on='gene_name')

    df.dropna(subset=['seq'], inplace=True)
    # df.drop(df[df['seq'].str.len() > 1e4].index, inplace=True)
    df.drop(df[df['utr3_len'] == 0].index, inplace=True)
    df.drop(df[df['cds_len'] == 0].index, inplace=True)
    df.drop(df[df['cds'].str.len() % 3 != 0].index, inplace=True)
    df.reset_index(drop=True, inplace=True)  # 保证后面 feat_df / X / y 索引严格对齐

    y = df['label'].values  # 二分类 0/1
    num_class = len(np.unique(y))

    # 拿到所有的数值特征及其列名
    feat_df, block_to_cols = build_scalar_feature_table(df)

    xgb_params = dict(
        objective='binary:logistic',
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
        eval_metric='logloss',
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )

    summary_rows = []
    for combo_name, combo_cfg in FEATURE_COMBOS.items():
        print(f"\n{'=' * 60}\n运行特征组合: {combo_name}\n{'=' * 60}")

        X, fname = assemble_feature_matrix(df, feat_df, block_to_cols, combo_cfg)  # 在这处理了序列
        print(f"[{combo_name}] 特征维度: {X.shape}")

        results = cross_validate_xgb(  # 进行十折交叉验证
            X, y,
            n_splits=10,
            xgb_params=xgb_params,
            verbose=1,  # 每一轮打印一次
            optimize_hyperparameters=True,
            n_trials=30,
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

        # ---- 存储该组合的逐样本预测结果：真值、预测标签、预测概率、所属折号 ----
        pred_df = pd.DataFrame({
            'gene_name': df['gene_name'].values,
            'fold': results['oof_fold'],  # 该样本是在第几折被当作验证集预测出来的
            'y_true': results['y'],
            'y_pred': results['pred_y'],
            'y_proba': results['oof_pred_proba'],  # 预测为正类(1)的概率
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

        # ---- 存储该组合每一折各自的验证集指标（auc/logloss/acc/f1/aupr/mcc/best_iteration） ----
        # results['fold_metrics'] 本身就是一个list，每个元素是一折的指标dict，直接转成表即可
        fold_metrics_df = pd.DataFrame(results['fold_metrics'])
        fold_metrics_path = os.path.join(OUT_DIR, f'{combo_name}_fold_metrics.csv')
        fold_metrics_df.to_csv(fold_metrics_path, index=False)

        # ---- 存储该组合每一折各自的特征重要性（未平均，长表格式：fold, feature, importance） ----
        # 这样后续如果想看某个特征的重要性在10折间的波动（画箱线图/误差棒），
        # 不用只依赖平均值，可以直接从这张表里按feature筛选10行。
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

        # ---- 汇总该组合的整体指标，最后统一存一份对比表，方便后续画图对比不同组合 ----
        row = {
            'combo': combo_name,
            'n_features': X.shape[1],
            'oof_auc': results['oof_auc'],
            'oof_acc': results['oof_acc'],
            'oof_f1': results['oof_f1'],
        }
        for metric_name, (mean_v, std_v) in results['summary'].items():
            row[f'{metric_name}_mean'] = mean_v
            row[f'{metric_name}_std'] = std_v
        summary_rows.append(row)

        # ---- 每个组合各自的混淆矩阵图（文件名带上组合名，避免互相覆盖） ----
        cm = confusion_matrix(results['y'], results['pred_y'])
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'pred_{i}' for i in range(num_class)],
                    yticklabels=[f'true_{i}' for i in range(num_class)])
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

        # ---- 所有组合跑完后，存一份总对比表，方便后续统一画图比较各组合表现 ----  下面的接口都是有的，直接替换文件就行，把画图的函数找到就行
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, 'all_combos_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n所有特征组合已跑完，汇总对比表已保存: {summary_path}")
    print(summary_df)
    # for combo_name, combo_cfg in FEATURE_COMBOS.items():
    #     print(f"\n{'='*60}\n运行特征组合: {combo_name}\n{'='*60}")
    #
    #     X, fname = assemble_feature_matrix(df, feat_df, block_to_cols, combo_cfg)
    #     print(f"[{combo_name}] 特征维度: {X.shape}")
    #
    #     results = cross_validate_xgb(  # 十折交叉验证
    #         X, y,
    #         n_splits=10,
    #         xgb_params=xgb_params,
    #         verbose=1,  # 每一轮打印一次
    #     )
    #
    #     # ---- 存储该组合的逐样本预测结果：真值、预测标签、预测概率 ----
    #     pred_df = pd.DataFrame({
    #         'gene_name': df['gene_name'].values,
    #         'y_true': results['y'],
    #         'y_pred': results['pred_y'],
    #         'y_proba': results['oof_pred_proba'],  # 预测为正类(1)的概率
    #     })
    #     pred_path = os.path.join(OUT_DIR, f'{combo_name}_predictions.csv')
    #     pred_df.to_csv(pred_path, index=False)
    #
    #     # ---- 存储该组合的特征重要性（十折平均） ----
    #     imp_df = pd.DataFrame({
    #         'feature': fname,
    #         'importance': results['avg_feature_importance'],
    #     }).sort_values('importance', ascending=False)
    #     imp_path = os.path.join(OUT_DIR, f'{combo_name}_feature_importance.csv')
    #     imp_df.to_csv(imp_path, index=False)
    #
    #     print(f"[{combo_name}] 已保存: {pred_path}")
    #     print(f"[{combo_name}] 已保存: {imp_path}")
    #
    #     # ---- 汇总该组合的整体指标，最后统一存一份对比表，方便后续画图对比不同组合 ----
    #     row = {
    #         'combo': combo_name,
    #         'n_features': X.shape[1],
    #         'oof_auc': results['oof_auc'],
    #         'oof_acc': results['oof_acc'],
    #         'oof_f1': results['oof_f1'],
    #     }
    #     for metric_name, (mean_v, std_v) in results['summary'].items():
    #         row[f'{metric_name}_mean'] = mean_v
    #         row[f'{metric_name}_std'] = std_v
    #     summary_rows.append(row)
    #
    #     # ---- 每个组合各自的混淆矩阵图（文件名带上组合名，避免互相覆盖） ----
    #     cm = confusion_matrix(results['y'], results['pred_y'])
    #     plt.figure()
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #                 xticklabels=[f'pred_{i}' for i in range(num_class)],
    #                 yticklabels=[f'true_{i}' for i in range(num_class)])
    #     plt.ylabel('True')
    #     plt.xlabel('Predicted')
    #     plt.title(f'Confusion Matrix - {combo_name}')
    #     plt.savefig(os.path.join(OUT_DIR, f'{combo_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    #     plt.close()
    #
    #     # ---- 每个组合各自的Top10重要性图 ----
    #     top10 = imp_df.head(10)
    #     plt.figure(figsize=(10, 8))
    #     plt.barh(top10['feature'], top10['importance'], color='steelblue')
    #     plt.xlabel('Importance')
    #     plt.title(f'Top 10 Feature Importance - {combo_name}')
    #     plt.gca().invert_yaxis()
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(OUT_DIR, f'{combo_name}_top10_feature_importance.png'), dpi=150)
    #     plt.close()
    #
    # # ---- 所有组合跑完后，存一份总对比表，方便后续统一画图比较各组合表现 ----
    # summary_df = pd.DataFrame(summary_rows)
    # summary_path = os.path.join(OUT_DIR, 'all_combos_summary.csv')
    # summary_df.to_csv(summary_path, index=False)
    # print(f"\n所有特征组合已跑完，汇总对比表已保存: {summary_path}")
    # print(summary_df)
