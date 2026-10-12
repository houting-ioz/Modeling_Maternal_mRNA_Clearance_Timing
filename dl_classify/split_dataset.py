import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split_csv(path, target_column='label',
                         train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                         random_state=42):
    """
    分层划分CSV文件为训练集、验证集、测试集

    参数:
        input_csv: 输入CSV文件路径
        target_column: 目标列名（类别标签列）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子
    """

    # 1. 读取数据
    df = pd.read_csv(path)
    # df1 = pd.read_csv('./data_v3/table_lnc_final.csv')
    # df = pd.concat([df, df1])  # 9911 后面是一起的结果

    df.dropna(subset=['seq'], inplace=True)
    df.drop(df[df['seq'].str.len() > 6e4].index, inplace=True)  # 分割的时候就去掉了大于6e4的
    df.drop(df[df['utr3_len'] == 0].index, inplace=True)  # 去掉没有3‘utr

    print(f"原始数据集形状: {df.shape}")
    print(f"类别分布:")
    print(df[target_column].value_counts())

    # 2. 检查比例总和是否为1
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"比例总和应为1.0，当前为{total_ratio}")

    # 先把序列长度筛选一下  token长度大于4096  不用筛了直接在token哪里做截断
    # MAX_LENGTH = 4096
    # l = df['sequences'].str.split(' ').apply(len)


    df_filtered = df # [l <= MAX_LENGTH].copy()

    # 3. 第一次分割：训练集和临时集
    train_df, temp_df = train_test_split(
        df_filtered,
        test_size=(1 - train_ratio),  # 临时集比例 = 验证集 + 测试集
        stratify=df_filtered[target_column],  # 分层抽样
        random_state=random_state
    )

    # 4. 第二次分割：验证集和测试集
    # 计算验证集在临时集中的比例
    val_ratio_in_temp = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_in_temp),  # 测试集在临时集中的比例
        stratify=temp_df[target_column],  # 继续分层抽样
        random_state=random_state
    )

    # 5. 打印分割结果
    print(f"\n分割结果:")
    print(f"训练集: {len(train_df)} 个样本 ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"验证集: {len(val_df)} 个样本 ({len(val_df) / len(df) * 100:.1f}%)")
    print(f"测试集: {len(test_df)} 个样本 ({len(test_df) / len(df) * 100:.1f}%)")

    # 6. 检查类别分布
    print(f"\n训练集类别分布:")
    print(train_df[target_column].value_counts(normalize=True))

    print(f"\n验证集类别分布:")
    print(val_df[target_column].value_counts(normalize=True))

    print(f"\n测试集类别分布:")
    print(test_df[target_column].value_counts(normalize=True))

    return train_df, val_df, test_df

if __name__ == '__main__':
    path = "./data/binary_tab.csv"
    train_df, val_df, test_df = stratified_split_csv(
        path,
        target_column='label',  # 指定标签列名
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )

    # 保存为CSV文件
    train_df.to_csv("../data/train.csv", index=False)
    val_df.to_csv("../data/val.csv", index=False)
    test_df.to_csv("../data/test.csv", index=False)