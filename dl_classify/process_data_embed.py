import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from collections import Counter


def extract_kmer_features(seq, k_list=[1, 2, 3]):
    feats = {}
    for k in k_list:
        total = len(seq) - k + 1
        for i in range(total):
            kmer = seq[i:i + k]
            feats[kmer] = feats.get(kmer, 0) + 1
        for key in feats:
            if len(key) == k:
                feats[key] /= (total + 1e-9)
    return feats

def extract_seq_features(seq):
    seq = seq.upper().replace('T', 'U')  # 统一为RNA
    gc_content = (seq.count('G') + seq.count('C')) / (len(seq) + 1e-9)
    au_skew = (seq.count('A') - seq.count('U')) / (seq.count('A') + seq.count('U') + 1e-9)
    gc_skew = (seq.count('G') - seq.count('C')) / (seq.count('G') + seq.count('C') + 1e-9)
    return gc_content, au_skew, gc_skew

def preprocess(df):
    seq_feats_cds = df['cds'].apply(lambda s: extract_seq_features(s))

    seq_feats_3utr = df['3utr'].apply(lambda s: extract_seq_features(s))

    df['gc_content1'] = seq_feats_cds.apply(lambda x: x[0])
    df['au_skew1'] = seq_feats_cds.apply(lambda x: x[1])
    df['gc_skew1'] = seq_feats_cds.apply(lambda x: x[2])
    df['gc_content2'] = seq_feats_3utr.apply(lambda x: x[0])
    df['au_skew2'] = seq_feats_3utr.apply(lambda x: x[1])
    df['gc_skew2'] = seq_feats_3utr.apply(lambda x: x[2])

    # 先处理数值数值特征列  确定特征
    num_cols = ['F1_TPM_median', # 'm6A_site_count',
                'polyA_GV_median', 'polyA_MI_median', 'polyA_MII_median',
                 'cds_len', 'utr3_len',
                'gc_content1', 'au_skew1', 'gc_skew1',
                'gc_content2', 'au_skew2', 'gc_skew2']  # 要计算分区的gc augc  把这些列加到df里

    # 缺失值：用中位数填充（比均值对偏态分布更稳健）
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # 对长尾特征做 log1p 变换（先于scaling）
    log_cols = ['F1_TPM_median', 'polyA_GV_median', 'polyA_MI_median',
                'polyA_MII_median', 'cdna_len', 'cds_len', 'utr3_len']
    df[log_cols] = np.log1p(df[log_cols])

    # RobustScaler（对异常值不敏感，比StandardScaler更适合你的数据）
    scaler = RobustScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])   # 这里的fit必须对各个数据集分开做

    # 这里是数值特征
    feature_df = df[num_cols].reset_index(drop=True)
    labels = df['label'].values
    print('----------------------结束处理df数据--------------------------------')
    return feature_df, None, labels, imputer, scaler


# 类别不平衡
def get_class_weights(labels):
    counter = Counter(labels)
    total = len(labels)
    weights = {cls: total / (len(counter) * cnt) for cls, cnt in counter.items()}
    return weights
