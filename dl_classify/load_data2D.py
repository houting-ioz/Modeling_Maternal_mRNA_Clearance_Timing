import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import random

def parse_cmdline_args(parser):

    parser.add_argument(
        '--cuda',
        type=int,
        default=0,
        help='Which GPU device to use. Uses cpu if `-1`.'
    )

    parser.add_argument(
        '--pt_path',
        type=str,
        # efault='/data',
        help='Which GPU device to use. Uses cpu if `-1`.'
    )

    parser.add_argument(
        '--encoder_type',
        type=str,
        default='cnn',
        # help='Which GPU device to use. Uses cpu if `-1`.'
    )

    parser.add_argument(
        '--pool_type',
        type=str,
        default='mean',
        # help='Which GPU device to use. Uses cpu if `-1`.'
    )

    parser.add_argument(
        '--in_dim',
        type=int,
        default=4,
        # help='Which GPU device to use. Uses cpu if `-1`.'
    )
    parser.add_argument(
        '--layers',
        type=int,
        default=2,
        # help='Which GPU device to use. Uses cpu if `-1`.'
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

############################################################
# Metrics
############################################################

# def loss_corn(logits, y_train, num_classes):
#     sets = []
#     for i in range(num_classes-1):
#         label_mask = y_train > i-1  # y_train是真值，是一个32维的向量只看当前这个label
#         label_tensor = (y_train[label_mask] > i).to(torch.int64)
#         sets.append((label_mask, label_tensor))
#
#     num_examples = 0
#     losses = 0.
#     for task_index, s in enumerate(sets):
#         train_examples = s[0]  # 为什么是32维的
#         train_labels = s[1]
#
#         if len(train_labels) < 1:
#             continue
#
#         num_examples += len(train_labels)
#         # 这里的一个数值表示的是该类别大于类别1的概率  这里应该是竖着取的，看
#         pred = logits[train_examples, task_index]  # 拿到某一个样本的15概率，这个不是一般分类的类别，因为是n-1个概率
#
#         # k-1个交叉熵
#         # loss = criterion(logits, y_train)
#         loss = -torch.sum(F.logsigmoid(pred)*train_labels
#                           + (F.logsigmoid(pred) - pred)*(1-train_labels)
#                           )
#         losses += loss
#     return losses/num_examples

from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
def compute_mae_and_rmse(model, data_loader, device):
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():

        mae, mse, num_examples = 0., 0., 0
        y_test = torch.tensor([])
        y_pred = torch.tensor([])
        y_pred_prob = torch.tensor([])
        loss = 0.0

        for i, (embed, mask, features, targets) in enumerate(data_loader):

            features = features.to(device)
            embed = embed.to(device)
            targets = targets.float().to(device)
            mask = mask.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(seq=embed, mask=mask, num_feat=features)  # 还是(32,)  每个函数都要改，直接另外起一个脚本
                loss += criterion(logits, targets.float())
            probs = torch.sigmoid(logits).cpu()
            y_pred_prob = torch.cat((y_pred_prob, probs))

            # probas = torch.sigmoid(logits)
            # probas_cum = torch.cumprod(probas, dim=1)
            # pred_labels = (probas_cum > 0.5).sum(dim=1)

            # print("probas_cum样例:\n", probas_cum[:5])
            predicted_labels = (probs >= 0.5).long()

            # predicted_labels = label_from_logits_func(logits)

            # 这里的关键思路是当MAE不在下降是就停止
            # 我需要在这里计算一下精度和准确率，以及在最外面要画混淆矩阵
            # f1 += f1_score(predicted_labels, targets, average='macro')  # 这里调用的有问题 要在外面计算
            y_test = torch.cat((y_test, targets.cpu()))
            y_pred = torch.cat((y_pred, predicted_labels))
            # y_pred.extend(predicted_labels)

            num_examples += targets.size(0)
            mae += torch.sum(torch.abs(predicted_labels - targets.cpu()))
            mse += torch.sum((predicted_labels - targets.cpu())**2)

        mae = mae / num_examples
        mse = mse / num_examples
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')  # 还得补指标，在外面算测试集上的
        print("预测类别分布:", torch.bincount(y_pred.int(), minlength=2))
        print("真实类别分布:", torch.bincount(y_test.int(), minlength=2))
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

from process_data_embed import preprocess

import glob, os, h5py

def load_all_samples(file_path, max_samples=None):
    """将所有样本加载到numpy数组中"""
    gene = []
    embed = []

    with h5py.File(file_path, 'r',  driver='core') as f:  # 这里读的只是1个.h5文件
        sample_keys = [k for k in f.keys() if k.startswith('data_')]   # 这个key就有问题
        # sample_keys.sort()  # 按顺序排序，前面的数据没有问题，这里的排序有问题！！
        sample_keys = sorted(sample_keys, key=lambda x: int(x.split('_')[1]))
        if max_samples:  # 9699个
            sample_keys = sample_keys[:max_samples]

        for key in sample_keys:
            group = f[key]
            gene.append(group['gene'][()])
            s = group['embed'][()].squeeze(0)
            if s.shape[0] > MAX_LEN:
                # print(s.shape[0])
                s = s[0:MAX_LEN]
            embed.append(s.astype(np.float16))
        #     # classes.append(group['class'][()])  # 这个类别还是不对，不对，train因为那边开始就是打乱的，所以要看valid-可以了
            # 列表推导式比 for + append 略快，且减少一次函数调用
        # gene = [f[k]['gene'][()].decode('utf-8') for k in sample_keys]
        # embed = [f[k]['embed'][()].squeeze(0) for k in sample_keys]
    gene = [x.decode('utf-8') for x in gene]
    return gene, embed

from enformer_pytorch.data import str_to_one_hot
from tqdm import tqdm
def load_embed(pt_path):
    genes = []
    embeds = []
    files = sorted(
        glob.glob(os.path.join(pt_path, '*.h5')),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )

    # files = [os.path.join(pt_path, '0.h5')]

    for file in tqdm(files, total=len(files), desc='Loading mRNA embedding...'):
        gene, embed = load_all_samples(file)  # 一次只能加载一个文件
        genes.extend(gene)
        embeds.extend(embed)  # 1,n_token,4096

    # embeds = [arr.squeeze(0) for arr in embeds]  # 去掉了第一个1的维度
    return genes, embeds

def pad_or_truncate(seq, max_len=8000):
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        return seq

class myDataset(Dataset):

    def __init__(self, csv_path, df_emb):  # 训练集分开的
        df = pd.read_csv(csv_path)  # 这里都还是原始的表格特征

        # 拼接其他df
        df1 = pd.read_csv('./data_v25/gene_3utr_mirna_summary.csv', encoding='gbk')  # microRNA结合特征
        df2 = pd.read_csv('./data_v25/seqs_pc_3utr.csv')  # 3utr
        df3 = pd.read_csv('./data_v25/seqs_pc_cds.csv')  # cds
        df4 = pd.read_csv('./data_v25/gene_mirna_20d_features.csv')
        df5 = pd.read_csv('./data_v25/cluster_gene_te_gene_level.tsv', sep='\t')

        df = df.merge(df1, how='left', on='gene_name')
        df = df.merge(df2, how='left', on='gene_name')
        df = df.merge(df3, how='left', on='gene_name')
        df = df.merge(df4, how='left', on='gene_name')
        df = df.merge(df5, how='left', on='gene_name')


        df.dropna(subset=['seq'], inplace=True)
        df.drop(df[df['utr3_len'] == 0].index, inplace=True)
        df.drop(df[df['cds_len'] == 0].index, inplace=True)
        df.drop(df[df['cds'].str.len() % 3 != 0].index, inplace=True)

        # df_emb已获取到的embedding
        embed_dict = dict(zip(df_emb['gene_name'], df_emb['embed']))
        df['embed'] = df['gene_name'].map(embed_dict)
        df.dropna(subset=['embed'], inplace=True)


        feature_df, _, labels, imputer, scaler = preprocess(df)
        self.y = torch.from_numpy(labels).to(torch.int64)  # 拿到所有的label
        self.features = torch.from_numpy(
            feature_df.values).to(torch.float32)
        self.embed = df['embed'].tolist()# 这里长度不一致导致报错
        self.embed = [torch.tensor(arr) for arr in self.embed]

    def __getitem__(self, index):
        features = self.features[index]
        embed = self.embed[index]
        label = self.y[index]
        return features, embed, label

    def __len__(self):
        return self.y.shape[0]

from torch.utils.data import WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence

MAX_LEN = 8710  # 95% 8710 90% 6659

def process_sequence(embed):  # embed: list
    for i in range(len(embed)):
        seq_len = embed[i].shape[0]
        if seq_len > MAX_LEN:
            embed[i] = embed[i][0:MAX_LEN]  # 往前面截取， 而且每个batch的长度都不一样，只能保证一个batch里的数据是一样长的
    return embed
    # seq_len = embed.shape[0]
    # if seq_len > MAX_LEN:
    #     # 截断策略，见下面选项
    #     embed = embed[:MAX_LEN]
    # return embed

def collate_fn(batch):
    """
    batch: list of dict{'dna_embed': [seq_len_i, embed_dim], 'num_feat': [num_dim], 'label': int}
    """
    # 从batch里面取出来 分别取出各个特征
    dna_embeds = [item[1] for item in batch]
    # 直接在这里进行截取
    dna_embeds = process_sequence(dna_embeds)

    num_feats = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[2] for item in batch])

    lengths = torch.tensor([e.shape[0] for e in dna_embeds])  # 得再最开是去掉太长的序列 95%分位数？不行直接采用截取的方式
    padded = pad_sequence(dna_embeds, batch_first=True, padding_value=0.0)  # [B, max_len, embed_dim]

    max_len = padded.shape[1]
    mask = torch.arange(max_len)[None, :] < lengths[:, None]  # [B, max_len]，True表示真实token
    # mask 是二维的， 标志的是当前这个token是不是被填充的，每个token是一个4096的
    return padded, mask, num_feats, labels  # 这里直接就有mask了  padded就是dna embed

def get_dataloaders(batch_size, train_csv_path, valid_csv_path, test_csv_path,
                            balanced=False, num_workers=0, num_classes=16, pt_path=None):
    embed = []
    gene = []

    # 在外层加载embedding
    # 获取各序列one-hot的代码
    # for idx, row in df.iterrows():
    #     # 截取到95%
    #     seq = pad_or_truncate(row['seq'], max_len=8710)
    #     onehot = str_to_one_hot(seq)  # 这里已经是one-hot了
    #     embed.append(onehot)
    #     gene.append(row['gene_name'])
    if pt_path == None:
        df_emb = None
    else:
        gene, embed = load_embed(pt_path=pt_path)  # 读取glms的embedding，直接替换路径
        # 需要筛选gene name， 但是不能放在df里了
        df_emb = pd.DataFrame({
            'gene_name': gene,
            'embed': embed,
        })


    train_dataset = myDataset(csv_path=train_csv_path, df_emb=df_emb)  # 7456
    valid_dataset = myDataset(csv_path=valid_csv_path, df_emb=df_emb)  # 416
    test_dataset = myDataset(csv_path=test_csv_path, df_emb=df_emb)  # 414

    if balanced:  # 需要修改
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=False,
                                  sampler=BalancedBatchSampler(
                                      train_dataset, labels=train_dataset.y),
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)

    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=False,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              drop_last=False,
                              shuffle=False,
                              num_workers=num_workers,
                              collate_fn=collate_fn)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             drop_last=False,
                             shuffle=False,
                             num_workers=num_workers,
                             collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader
