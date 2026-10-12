import argparse
import os
import time

import torch
import torch.nn.functional as F
from load_data1D import parse_cmdline_args
from load_data1D import compute_mae_and_rmse
from load_data1D import get_dataloaders

from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score, average_precision_score, matthews_corrcoef, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import numpy as np
from model import MultilayerPerceptron_onlyembed_3class

# Argparse helper
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args = parse_cmdline_args(parser)
NUM_WORKERS = args.numworkers
LEARNING_RATE = args.learningrate
NUM_EPOCHS = args.epochs
BATCH_SIZE = 128  # args.batchsize
OUTPUT_DIR = args.output_dir
LOSS_PRINT_INTERVAL = args.loss_print_interval

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(args.output_dir)
BEST_MODEL_PATH = os.path.join(args.output_dir, 'best_model.pt')
LOGFILE_PATH = os.path.join(args.output_dir, 'training.log')

if args.cuda >= 0 and torch.cuda.is_available():
    DEVICE = torch.device(f'cuda:{args.cuda}')
else:
    DEVICE = torch.device('cpu')

if args.seed == -1:
    RANDOM_SEED = None
else:
    RANDOM_SEED = args.seed

NUM_CLASSES = 3
NUM_FEATURES = 3072

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def format_params(num_params):
    """将参数量转换为论文里常见的 K/M/B 格式"""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)

if __name__ == '__main__':
    set_seed(RANDOM_SEED)

    train_loader, valid_loader, test_loader = get_dataloaders(  # 直接返回的数据迭代器
        batch_size=BATCH_SIZE,
        train_csv_path='../data/train.csv',  # 需要更换为3分类的对应文件
        valid_csv_path='../data/val.csv',
        test_csv_path='../data/test.csv',
        balanced=True,  # True
        num_workers=NUM_WORKERS,
        num_classes=NUM_CLASSES)


    model = MultilayerPerceptron_onlyembed_3class(num_features=NUM_FEATURES,
                                num_classes=NUM_CLASSES)
    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {format_params(total_params)} ({total_params:,})")
    print(f"Trainable parameters: {format_params(trainable_params)} ({trainable_params:,})")

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6    # 最低衰减
    )

    criterion = torch.nn.CrossEntropyLoss()

    best_valid_loss = torch.tensor(float('inf'))

    s = (f'Script: {__file__}\n'
        f'PyTorch version: {torch.__version__}\n'
        f'Device: {DEVICE}\n'
        f'Learning rate: {LEARNING_RATE}\n'
        f'Batch size: {BATCH_SIZE}\n')

    print(s)
    with open(LOGFILE_PATH, 'w') as f:
        f.write(f'{s}\n')
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):

        model.train()
        for batch_idx, (features, embed, targets) in enumerate(train_loader):

            features = features.to(DEVICE)
            embed = embed.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(embed)

            loss = criterion(logits, targets)  # 三分类还要改这里  二分类需要.float()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not batch_idx % LOSS_PRINT_INTERVAL:
                s = (f'Epoch: {epoch:03d}/{NUM_EPOCHS:03d} | '
                     f'Batch {batch_idx:04d}/'
                     f'{len(train_loader):04d} | '
                     f'Loss: {loss:.4f}')
                print(s)
                with open(LOGFILE_PATH, 'a') as f:
                    f.write(f'{s}\n')

        scheduler.step()

        # eval
        model.eval()
        with torch.no_grad():
            valid_mae, valid_rmse, f1, accuracy, y_test, y_pred, _, loss = compute_mae_and_rmse(  # 直接计算mae、rmse
                model=model,
                data_loader=valid_loader,
                device=DEVICE,
                criterion=criterion
            )

            if loss < best_valid_loss:
                best_valid_loss = loss
                best_epoch = epoch
                torch.save(model.state_dict(), BEST_MODEL_PATH)

            s = (f'MAE Current Valid: {valid_mae:.4f} Ep. {epoch}'
                 f' | Acc Current Valid: {accuracy:.4f} Ep. {epoch}'
                 f' | F1 Current Valid: {f1:.4f} Ep. {epoch}'
                 f' | Best Valid: {best_valid_loss:.4f} Ep. {best_epoch}'  # 这里有打印最好的epoch
                 )
            s += f'\nTime elapsed: {(time.time() - start_time) / 60:.2f} min'
            print(s)
            with open(LOGFILE_PATH, 'a') as f:
                f.write('%s\n' % s)

    # Final test
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    with torch.no_grad():
        train_mae, train_rmse, train_f1, accuracy, y_test, y_pred,_,_ = compute_mae_and_rmse(  # 计算的指标不够
            model=model,
            data_loader=train_loader,
            device=DEVICE,
            criterion=criterion)
        valid_mae, valid_rmse, valid_f1, accuracy, y_test11, y_pred11, _, _ = compute_mae_and_rmse(
            model=model,
            data_loader=valid_loader,
            device=DEVICE,
            criterion=criterion
        )
        test_mae, test_rmse, test_f1, accuracy, y_test1, y_pred1, y_pred_prob, _ = compute_mae_and_rmse(
            model=model,
            data_loader=test_loader,
            device=DEVICE,
            criterion=criterion
        )
    print(classification_report(y_test1, y_pred1, digits=3,
                                target_names=['class_0', 'class_1', 'class_2']))

    metrics = {
        'mutil_auc': roc_auc_score(y_test1, y_pred_prob,  # shape=(n_samples, 3), 每列是各类别概率
                                   multi_class='ovr',  # 'ovr' (one-vs-rest) 或 'ovo' (one-vs-one)
                                   average='weighted',  # 同样支持weighted/macro
                                   sample_weight=None
                                   ),
        'logloss': log_loss(y_test1, y_pred_prob, sample_weight=None),
        'accuracy': accuracy_score(y_test1, y_pred1, sample_weight=None),
        'f1': f1_score(y_test1, y_pred1, average='macro', sample_weight=None),
        'f1_weight': f1_score(y_test1, y_pred1, average='weighted', sample_weight=None)
    }
    print(f"\n AUC (test): {metrics['mutil_auc']:.4f}")
    print(f"\n logloss (test): {metrics['logloss']:.4f}")
    print(f"\n ACC (test): {metrics['accuracy']:.4f}")
    print(f"\n F1 (test): {metrics['f1']:.4f}")
    print(f"\n F1_Weight (test): {metrics['f1_weight']:.4f}")


    cm = confusion_matrix(y_test1, y_pred1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'pred_{i}' for i in range(NUM_CLASSES)],
                yticklabels=[f'true_{i}' for i in range(NUM_CLASSES)])
    plt.ylabel('True');
    plt.xlabel('Predicted')
    plt.savefig('./data_v6/confusion_matrix.png', dpi=300, bbox_inches='tight')


    s = ('\n\n=========================================\n\n'  # 最后跑所有的数据集
        'Performance of best model based on validation set MAE:\n'
        f'Train MAE / RMSE / F1: {train_mae:.4f} / {train_rmse:.4f} / {train_f1}\n'
        f'Valid MAE / RMSE / F1: {valid_mae:.4f} / {valid_rmse:.4f} / {valid_f1}\n'
        f'Test  MAE / RMSE / F1: {test_mae:.4f} / {test_rmse:.4f} / {test_f1}')

    print(s)
