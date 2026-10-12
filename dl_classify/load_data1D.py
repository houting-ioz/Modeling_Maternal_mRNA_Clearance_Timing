import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import random
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from process_data import preprocess
import glob, os, h5py
from itertools import product
from collections import Counter

"""
说明，该脚本只能加载1d的编码，两个分类任务能够通用
"""

def parse_cmdline_args(parser):

    parser.add_argument(
        '--cuda',
        type=int,
        default=-1,
        help='Which GPU device to use. Uses cpu if `-1`.'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Which random seed to use. No random seed if `-1`.'
    )

    parser.add_argument(
        '--numworkers',
        type=int,
        default=0,
        help='How many workers to use for the data loader.'
    )

    parser.add_argument(
        '--learningrate',
        type=float,
        default=5e-4,
        help='Learning rate.'
    )

    parser.add_argument(
        '--batchsize',
        type=int,
        default=64,
        help='Batch size.'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs.'
    )

    parser.add_argument(
        '--loss_print_interval',
        type=int,
        default=10,
        help='How frequently to print the loss during training.'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='output_dir',
        help='Location for the training log and best'
             ' model (based on validation set MAE).'
    )

    parser.add_argument(
        '--overwrite',
        type=str,
        choices=['true', 'false'],
        default='false',
        help='Whether to overwrite the results folder.'
        )

    parser.add_argument(
        '--logfile_path',
        type=str,
        default='training.log',
        help='Location for the log file.'
        )

    parser.set_defaults(feature=True)
    args = parser.parse_args()

    d = {'true': True,
         'false': False}

    args.overwrite = d[args.overwrite]

    return args

def compute_mae_and_rmse(model, data_loader, device, criterion):
    with torch.no_grad():
        # criterion = torch.nn.BCEWithLogitsLoss()
        mae, mse, num_examples = 0., 0., 0
        y_test = torch.tensor([])
        y_pred = torch.tensor([])
        y_pred_prob = torch.tensor([])
        loss = 0.0

        for i, (features, embed, targets) in enumerate(data_loader):

            features = features.to(device)
            embed = embed.to(device)
            targets = targets.float().to(device)

            logits = model(embed)  # 还是(32,)  每个函数都要改，直接另外起一个脚本
            loss += criterion(logits, targets.long())  # .float() 3分类是long
            # predicted_labels = torch.argmax(logits, dim=1)

            # # 这里是二分类的写法
            # probs = torch.sigmoid(logits)  # sigmoid用来算二分类，多分类得用softmax
            # y_pred_prob = torch.cat((y_pred_prob, probs))
            # predicted_labels = (probs >= 0.5).long()   # 这里是二分类的写法
            # 三分类的写法
            probs = torch.softmax(logits, dim=1)  # 转成3个类的概率，每行加起来正好=1
            predicted_labels = probs.argmax(dim=1)  # 拿概率最大的那个当预测结果
            y_pred_prob = torch.cat((y_pred_prob, probs))

            # predicted_labels = label_from_logits_func(logits)

            # 这里的关键思路是当MAE不在下降是就停止
            # 我需要在这里计算一下精度和准确率，以及在最外面要画混淆矩阵
            # f1 += f1_score(predicted_labels, targets, average='macro')  # 这里调用的有问题 要在外面计算
            y_test = torch.cat((y_test, targets))
            y_pred = torch.cat((y_pred, predicted_labels))
            # y_pred.extend(predicted_labels)

            num_examples += targets.size(0)
            mae += torch.sum(torch.abs(predicted_labels - targets))
            mse += torch.sum((predicted_labels - targets)**2)

        mae = mae / num_examples
        mse = mse / num_examples
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')  # 还得补指标，在外面算测试集上的
        # print("预测类别分布:", torch.bincount(y_pred.int(), minlength=2))
        # print("真实类别分布:", torch.bincount(y_test.int(), minlength=2))
        return mae, np.sqrt(mse), f1, accuracy, y_test, y_pred, y_pred_prob, loss/len(data_loader)


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    # adopted from https://github.com/galatolofederico/pytorch-balanced-batch/blob/master/sampler.py
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)

    def _get_label(self, dataset, idx, labels=None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max*len(self.keys)

def load_all_samples(file_path, max_samples=None):
    """将所有样本加载到numpy数组中"""
    gene = []
    embed = []

    with h5py.File(file_path, 'r') as f:  # 这里读的只是1个.h5文件
        sample_keys = [k for k in f.keys() if k.startswith('data_')]   # 这个key就有问题
        # sample_keys.sort()  # 按顺序排序，前面的数据没有问题，这里的排序有问题！！
        sample_keys = sorted(sample_keys, key=lambda x: int(x.split('_')[1]))
        if max_samples:  # 9699个
            sample_keys = sample_keys[:max_samples]

        for key in sample_keys:
            group = f[key]
            gene.append(group['gene'][()])
            embed.append(group['embed'][()])
            # classes.append(group['class'][()])  # 这个类别还是不对，不对，train因为那边开始就是打乱的，所以要看valid-可以了
    gene = [x.decode('utf-8') for x in gene]
    return gene, embed

def load_embed(pt_path):
    genes = []
    embeds = []
    # files = sorted(glob.glob(os.path.join(
    #     pt_path, f'*.h5')), key=lambda x: int(x.split('-')[-1].split('.')[0]))

    files = [os.path.join(pt_path, '0.h5')]

    for file in files:
        gene, embed = load_all_samples(file)  # 一次只能加载一个文件
        genes.extend(gene)
        embeds.extend(embed)

    return genes, embeds

def extract_kmer_features(seq, k_list=[1, 2, 3]):
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

class myDataset(Dataset):

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)  # 加载原始的表格特征

        # 拼接其他df
        df1 = pd.read_csv('../data/gene_3utr_mirna_summary.csv', encoding='gbk')  # microRNA结合特征
        df2 = pd.read_csv('../data/seqs_pc_3utr.csv')  # 3utr
        df3 = pd.read_csv('../data/seqs_pc_cds.csv')  # cds
        df4 = pd.read_csv('../data/gene_mirna_20d_features.csv')
        df5 = pd.read_csv('./.data/cluster_gene_te_gene_level.tsv', sep='\t')

        df = df.merge(df1, how='left', on='gene_name')
        df = df.merge(df2, how='left', on='gene_name')
        df = df.merge(df3, how='left', on='gene_name')
        df = df.merge(df4, how='left', on='gene_name')
        df = df.merge(df5, how='left', on='gene_name')

        df.dropna(subset=['seq'], inplace=True)
        df.drop(df[df['utr3_len'] == 0].index, inplace=True)
        df.drop(df[df['cds_len'] == 0].index, inplace=True)
        df.drop(df[df['cds'].str.len() % 3 != 0].index, inplace=True)

        # 取k-ker或密码子频率或glms的编码  在这直接更换embed，没有外部接口
        embed = []
        gene = []
        for i, row in df.iterrows():
            kmer = extract_kmer_features(row['seq'], k_list=[6])  # k-mer
            # kmer, _ = extract_codon_frequency(row['cds'])  # 密码子频率
            embed.append(kmer)
            gene.append(row['gene_name'])
        # 读取embed  替换路径即可
        # gene, embed = load_embed(pt_path='E:/2026_0/DeepLearning/DeepLearning/mRNA/data/embed/generator/0720/cds/mean')
        df_emb = pd.DataFrame({
            'gene_name': gene,
            'embed': embed,
        })

        dim = len(df_emb['embed'].iloc[0])  # 向量维度，比如768
        col_names = [f'embed_{i}' for i in range(dim)]
        df = df.merge(df_emb, how='left', on='gene_name')  # 把embed合并到原始的df上
        df.dropna(subset=['embed'], inplace=True)

        feature_df, embed_df, labels, imputer, scaler = preprocess(df)  # 主要是这个函数
        self.y = torch.from_numpy(labels).to(torch.int64)  # 拿到所有的label
        self.features = torch.from_numpy(
            feature_df.values).to(torch.float32)
        self.embed = torch.from_numpy(
            embed_df.values).to(torch.float32)

    def __getitem__(self, index):
        features = self.features[index]
        embed = self.embed[index]
        label = self.y[index]
        return features, embed, label

    def __len__(self):
        return self.y.shape[0]

def get_dataloaders(batch_size, train_csv_path,
                            valid_csv_path, test_csv_path,
                            balanced=False, num_workers=0, num_classes=16):

    train_dataset = myDataset(csv_path=train_csv_path)  # 7456
    valid_dataset = myDataset(csv_path=valid_csv_path)  # 416
    test_dataset = myDataset(csv_path=test_csv_path)  # 414

    if balanced:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=False,
                                  sampler=BalancedBatchSampler(
                                      train_dataset, labels=train_dataset.y),
                                  num_workers=num_workers)

    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=num_workers)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              drop_last=False,
                              shuffle=False,
                              num_workers=num_workers)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             drop_last=False,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, valid_loader, test_loader
