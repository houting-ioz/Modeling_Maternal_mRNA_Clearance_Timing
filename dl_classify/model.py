import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MultilayerPerceptron_onlyembed(nn.Module):

    def __init__(self, num_features, num_classes,
                 num_hidden_1=256, num_hidden_2=128, num_hidden_3=64,
                 dropout_rate=0.1):
        super().__init__()  # 还有梯度裁剪啥的

        self.num_classes = num_classes
        self.my_network = torch.nn.Sequential(

            # 1st hidden layer
            torch.nn.Linear(num_features, num_hidden_1, bias=False),
            torch.nn.BatchNorm1d(num_hidden_1),   # BN在激活前
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),        # Dropout减弱

            # 2nd hidden layer
            torch.nn.Linear(num_hidden_1, num_hidden_2, bias=False),
            torch.nn.BatchNorm1d(num_hidden_2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),

            # 3rd hidden layer（新增，缓解欠拟合）
            torch.nn.Linear(num_hidden_2, num_hidden_3, bias=False),
            torch.nn.BatchNorm1d(num_hidden_3),
            torch.nn.GELU(),


            # 输出层  分类这里也要改
            torch.nn.Linear(num_hidden_3, self.num_classes - 1)
            # torch.nn.Linear(num_hidden_3, self.num_classes)
        )

    def forward(self, x):
        return self.my_network(x).squeeze(-1)

class MultilayerPerceptron_onlyembed_3class(torch.nn.Module):

    def __init__(self, num_features, num_classes,
                 num_hidden_1=256, num_hidden_2=128, num_hidden_3=64,
                 dropout_rate=0.1):
        super().__init__()

        self.num_classes = num_classes
        self.my_network = torch.nn.Sequential(

            # 1st hidden layer
            torch.nn.Linear(num_features, num_hidden_1, bias=False),
            torch.nn.BatchNorm1d(num_hidden_1),   # BN在激活前
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),        # Dropout减弱

            # 2nd hidden layer
            torch.nn.Linear(num_hidden_1, num_hidden_2, bias=False),
            torch.nn.BatchNorm1d(num_hidden_2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),

            # 3rd hidden layer（新增，缓解欠拟合）
            torch.nn.Linear(num_hidden_2, num_hidden_3, bias=False),
            torch.nn.BatchNorm1d(num_hidden_3),
            torch.nn.GELU(),


            # 输出层  分类这里也要改
            # torch.nn.Linear(num_hidden_3, self.num_classes - 1)
            torch.nn.Linear(num_hidden_3, self.num_classes)
        )

    def forward(self, x):
        return self.my_network(x).squeeze(-1)

# mlp 中加入数值特征对应的2分类模型
# 3分类模型只需修改最后一层线性层的
class FusionMLP(nn.Module):
    def __init__(
        self,
        dna_dim: int,        # DNA模型mean token向量的维度，比如768
        num_dim: int,        # 数值特征的维度
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # 分支1：处理DNA embedding
        self.dna_branch = nn.Sequential(
            nn.Linear(dna_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 分支2：处理数值特征
        self.num_branch = nn.Sequential(
            nn.Linear(num_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 融合后的分类头
        fusion_dim = hidden_dim + hidden_dim // 2
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes-1),  # 1,2都可以需要更换损失函数，先这样吧
        )

    def forward(self, dna_feat: torch.Tensor, num_feat: torch.Tensor) -> torch.Tensor:
        """
        dna_feat: [batch_size, dna_dim]  DNA模型mean token向量
        num_feat: [batch_size, num_dim]  数值特征
        """
        dna_out = self.dna_branch(dna_feat)
        num_out = self.num_branch(num_feat)


        fused = torch.cat([dna_out, num_out], dim=1)  # 特征拼接融合  横着拼接到后面
        logits = self.classifier(fused)
        return logits.squeeze(-1)
class FusionMLP_Gated(nn.Module):
    def __init__(
            self,
            dna_dim: int,  # DNA模型mean token向量的维度，比如768
            num_dim: int,  # 数值特征的维度
            hidden_dim: int = 128,
            num_classes: int = 2,
            dropout: float = 0.3,
    ):
        super().__init__()

        # 分支1：处理DNA embedding
        self.dna_branch = nn.Sequential(
            nn.Linear(dna_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 分支2：处理数值特征
        self.num_branch = nn.Sequential(
            nn.Linear(num_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 融合后的分类头
        # fusion_dim = hidden_dim + hidden_dim // 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes - 1),  # 1,2都可以需要更换损失函数，先这样吧
        )
        # 门控
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Sigmoid()  # 变成0-1的权重
        )

    def forward(self, dna_feat: torch.Tensor, num_feat: torch.Tensor) -> torch.Tensor:
        """
        dna_feat: [batch_size, dna_dim]  DNA模型mean token向量
        num_feat: [batch_size, num_dim]  数值特征
        """
        dna_out = self.dna_branch(dna_feat)
        num_out = self.num_branch(num_feat)

        # fused = torch.cat([dna_out, num_out], dim=1)  # 特征拼接融合  横着拼接到后面
        # 门控直接替代这里的concat
        g = self.gate(torch.cat([dna_out, num_out], dim=1))
        fused = g * dna_out + (1 - g) * num_out  # 逐元素加权融合，维度仍是hidden_dim

        logits = self.classifier(fused)
        return logits.squeeze(-1)


class dlModel(nn.Module):
    def __init__(self, embed_dim, num_dim, hidden_dim=128, num_classes=2, encoder_type="cnn", pool_type='attn', layers=2):
        super().__init__()
        if encoder_type == "cnn":
            self.dna_encoder = ResNet1d(embed_dim, hidden_dim, layers=(layers,))
        elif encoder_type == "transformer":
            self.dna_encoder = DNATransformerEncoder(embed_dim, hidden_dim, layers=layers)
            hidden_dim = embed_dim
        elif encoder_type == "bilstm":  # 先不单独拿出来
            self.dna_encoder = DNABiLSTMEncoder(embed_dim, hidden_dim, num_layers=2, pool_type="last")


        self.classifier = ClassificationHead(num_dim=num_dim, in_dim=hidden_dim, hidden_dim=128, num_classes=num_classes,
                                             pool_type=pool_type)

    def forward(self, seq, mask, num_feat):
        d = self.dna_encoder(seq, mask)  # [B, hidden_dim]  # forward里传mask

        return self.classifier(d, num_feat, mask).squeeze(-1)

class BasicBlock1d(nn.Module):
    """标准 1D 残差基本块：Conv-BN-ReLU-Conv-BN + shortcut"""
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, downsample=None):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x, mask=None):
        """
        x: [B, C, L]
        mask: [B, 1, L]  1=有效，0=padding；传入则在关键节点清零padding位置，防止边界信息泄漏
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if mask is not None:
            out = out * mask  # 卷积后重新清零padding，防止泄漏信息污染BN统计和下一层卷积

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        if mask is not None:
            out = out * mask  # 残差相加后再次清零，保证block输出的padding位置严格为0

        return out
class ResNet1d(nn.Module):
    """
    通用 1D-ResNet 编码器（无下采样版本，保持序列长度不变，输出逐token特征）
    输入: (B, L, embed_dim)
    输出: (B, L, embed_dim)
    """

    def __init__(self, embed_dim, hidden_dim, kernel_size=5, layers=(2,)):
        super().__init__()
        self.in_channels = embed_dim

        # 输入归一化：Evo2 embedding数值分布不可控，先标准化，和其他encoder保持一致
        self.input_norm = nn.LayerNorm(embed_dim)

        self.layer1 = self._make_layer(hidden_dim, layers[0], kernel_size, stride=1)

        self._init_weights()

    def _make_layer(self, out_channels, blocks, kernel_size, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        layers = nn.ModuleList([
            BasicBlock1d(self.in_channels, out_channels, kernel_size, stride, downsample)
        ])
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock1d(self.in_channels, out_channels, kernel_size))

        return layers  # 改成ModuleList，因为forward需要手动传mask给每个block

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):
        """
        x: (B, L, embed_dim)
        mask: (B, L)  1/True=有效, 0/False=padding
        """
        # print("embed.shape:", x.shape)  # 应该是 (B, 72, embed_dim) 或类似
        # print("mask.shape:", mask.shape)  # 应该是 (B, 768)

        mask_f = mask.unsqueeze(-1).float()      # (B, L, 1)，先在[B,L,C]维度下清零一次
        x = x * mask_f
        x = self.input_norm(x)
        x = x * mask_f                            # LayerNorm对padding位置(全0输入)会输出非零的bias项，需再清零一次

        x = x.transpose(1, 2)                     # [B, embed_dim, L]  conv1d要求channel在前
        mask_c = mask.unsqueeze(1).float()         # (B, 1, L)，用于channel-first格式下的清零

        for block in self.layer1:
            x = block(x, mask_c)

        x = x.transpose(1, 2)                      # 转回 [B, L, embed_dim]
        return x

class DNATransformerEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, nhead=4, layers=2, dropout=0.1):
        super().__init__()

        # 1. 输入归一化：Evo2 embedding 数值范围不可控，必须先标准化
        self.input_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            # dim_feedforward=embed_dim * 4,   # 默认是2048，显式指定更可控
            batch_first=True,
            dropout=dropout,
            activation='gelu',               # gelu比relu更常用于transformer，梯度更平滑
            norm_first=True,                 # Pre-LN，训练更稳定，尤其是小层数+无warmup时
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        # 2. Pre-LN结构下，TransformerEncoder最后需要额外加一个LayerNorm
        #    （PyTorch官方实现里 norm_first=True 时若不传norm，最后一层输出没有做归一化）
        self.final_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask):
        """
        x: (B, L, embed_dim)  Evo2提取的embedding
        mask: (B, L)  True=有效位置, False=padding位置
        """
        # 输入归一化，防止Evo2 embedding数值分布异常直接冲击Transformer
        x = self.input_norm(x)

        # Transformer要求src_key_padding_mask: True表示要mask掉(padding)，与我们mask定义相反
        pad_mask = ~mask  # (B, L)  True=padding

        # 关键修复：防止某个样本整行全部被mask（全padding），
        # 否则该样本所有query对所有key的attention score全为-inf，softmax输出nan
        fully_masked = pad_mask.all(dim=1)  # (B,)  哪些样本整行都是padding
        if fully_masked.any():
            # 对这些样本，强制解开第一个位置的mask，避免softmax除0
            # （反正这些样本本身是无效样本，后续loss计算时应该用样本级mask再屏蔽掉）
            pad_mask = pad_mask.clone()
            pad_mask[fully_masked, 0] = False

        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.final_norm(x)

        return x

class DNABiLSTMEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers=1, dropout=0.2, pool_type="attn"):
        super().__init__()
        self.pool_type = pool_type

        # 1. 输入归一化：和Transformer版本一样，Evo2 embedding数值分布不可控，先标准化
        self.input_norm = nn.LayerNorm(embed_dim)

        # 双向LSTM，hidden_size设为hidden_dim//2，双向拼接后正好是hidden_dim
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,  # 这里指定的layer数
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,  # 单层LSTM设置dropout会报warning
        )

        # if pool_type == "attn":
        #     self.attn = nn.Linear(hidden_dim, 1)

        # self.proj = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        # )

        self._init_weights()

    def _init_weights(self):
        # LSTM权重初始化：input-hidden用xavier，hidden-hidden用orthogonal（LSTM训练更稳定的常见做法）
        # bias初始化为0，但forget gate bias设为1，帮助训练初期减少遗忘、缓解梯度消失
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # LSTM的bias排列顺序是[b_ii|b_if|b_ig|b_io]，即4个门each占1/4长度
                # 把forget gate（第2段）的bias初始化为1
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x, mask):
        """
        x: [B, L, embed_dim]
        mask: [B, L]，True=真实token，False=padding
        """
        # 输入归一化，防止Evo2 embedding数值分布异常直接冲击LSTM
        x = self.input_norm(x)

        lengths = mask.sum(dim=1).long().cpu()  # 每个样本的真实长度，pack要求在cpu上

        # 关键修复：防止某个样本长度为0（全padding），
        # pack_padded_sequence/cuDNN LSTM要求所有长度必须>0，否则直接报错崩溃
        zero_len_mask = (lengths == 0)
        if zero_len_mask.any():
            lengths = lengths.clamp(min=1)
            # 注意：这些样本本身是无效样本，LSTM会"看到"第一个位置(实际是padding=0)，
            # 输出无意义，务必在loss计算时用样本级mask把这些无效样本排除掉

        # pack之前无需手动按长度降序排列，enforce_sorted=False自动处理（速度差异很小）
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)

        # if self.pool_type == "last":
        #     h_forward = h_n[-2]
        #     h_backward = h_n[-1]
        #     pooled = torch.cat([h_forward, h_backward], dim=1)
        #     return pooled

        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
        # out: [B, L, hidden_dim]（双向拼接后的维度）

        # pooled = self._pool(out, mask)
        return out

class ClassificationHead(nn.Module):  # 全部都在外面用这个分类头，池化加分类
    def __init__(self, num_dim, in_dim=256, hidden_dim=128, num_classes=2, pool_type="attn", dropout=0.3):
        super().__init__()
        self.pool_type = pool_type

        if pool_type == "attn":
            self.attn = nn.Linear(in_dim, 1)  # attention pooling的打分层

        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes - 1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),  # 这里是128
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        # 融合数值特征的模块，如果需要融合还需修改对应forward的里的部分
        # self.num_branch = nn.Sequential(
        #     nn.Linear(num_dim, in_dim),
        #     nn.BatchNorm1d(in_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.3),
        # )
        # self.gate = nn.Sequential(nn.Linear(in_dim * 2, in_dim), nn.Sigmoid())

    def forward(self, x, num_feat, mask=None):
        """
        x: [bs, n_token, in_dim]  模型输出
        mask: [bs, n_token]，True/1表示真实token，False/0表示padding；如果序列没有padding可以传None
        """
        pooled = self._pool(x, mask)  # [bs, in_dim]

        # 在不融合数值特征的情况下不需要下面这个部分
        # concat 数值特征
        # n = self.num_branch(num_feat)
        # g = self.gate(torch.cat([pooled, n], dim=1))  # 这里能采用cross attention的方式吗
        # # print(g.shape)
        # fused = g * pooled + (1 - g) * n

        logits = self.classifier(pooled)  # [bs, num_classes]
        return logits

    def _pool(self, x, mask):
        if mask is None:
            # 没有padding的情况，直接普通mean pooling
            if self.pool_type == "mean":
                return x.mean(dim=1)
            elif self.pool_type == "max":
                return x.max(dim=1).values
            elif self.pool_type == "attn":
                scores = self.attn(x).squeeze(-1)  # [bs, n_token]
                weights = torch.softmax(scores, dim=1)
                return (x * weights.unsqueeze(-1)).sum(dim=1)

        # 有padding的情况，必须用mask
        mask = mask.bool()
        if self.pool_type == "mean":
            m = mask.unsqueeze(-1).float()
            summed = (x * m).sum(dim=1)
            count = m.sum(dim=1).clamp(min=1e-6)
            return summed / count

        elif self.pool_type == "max":
            x_masked = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            return x_masked.max(dim=1).values

        elif self.pool_type == "attn":
            scores = self.attn(x).squeeze(-1)  # [bs, n_token]
            scores = scores.masked_fill(~mask, float('-inf'))
            weights = torch.softmax(scores, dim=1)
            return (x * weights.unsqueeze(-1)).sum(dim=1)