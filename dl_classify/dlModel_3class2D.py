import argparse
import os
import shutil
import time
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score, average_precision_score, matthews_corrcoef, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.nn as nn
from load_data2D import parse_cmdline_args
from load_data2D import compute_mae_and_rmse
from load_data2D import get_dataloaders
from model import dlModel

# Argparse helper
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args = parse_cmdline_args(parser)

NUM_WORKERS = args.numworkers
LEARNING_RATE = args.learningrate
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
OUTPUT_DIR = args.output_dir
LOSS_PRINT_INTERVAL = args.loss_print_interval
BEST_MODEL_PATH = os.path.join(args.output_dir, 'best_model.pt')
LOGFILE_PATH = os.path.join(args.output_dir, 'training.log')
NUM_CLASSES = 3
NUM_FEATURES = args.in_dim

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

def to_numpy(x):
    """兼容 torch.Tensor / list / numpy，统一转成 numpy 数组"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(args.output_dir)
    if args.cuda >= 0 and torch.cuda.is_available():
        DEVICE = torch.device(f'cuda:{args.cuda}')
    else:
        DEVICE = torch.device('cpu')

    if args.seed == -1:
        RANDOM_SEED = None
    else:
        RANDOM_SEED = args.seed

    model = dlModel(embed_dim=args.in_dim, num_dim=12, hidden_dim=256, num_classes=3,
                    encoder_type=args.encoder_type, layers=args.layers, pool_type=args.pool_type)
    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {format_params(total_params)} ({total_params:,})")
    print(f"Trainable parameters: {format_params(trainable_params)} ({trainable_params:,})")

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    from torch.optim.lr_scheduler import LambdaLR

    warmup_steps = 500
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / warmup_steps))

    train_loader, valid_loader, test_loader = get_dataloaders(  # 直接返回的数据迭代器
        batch_size=BATCH_SIZE,
        train_csv_path='../data/train.csv',  # xgboost是分两类，而深度学习必须分成三个集合
        valid_csv_path='../data/val.csv',
        test_csv_path='../data/test.csv',
        balanced=True,  # True
        num_workers=NUM_WORKERS,
        num_classes=NUM_CLASSES,
        pt_path=args.pt_path)

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
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(1, NUM_EPOCHS + 1):
        losses = 0.0
        model.train()
        for batch_idx, (embed, mask, features, targets) in enumerate(
                train_loader):
            features = features.to(DEVICE)
            embed = embed.to(DEVICE)
            mask = mask.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(seq=embed, mask=mask, num_feat=features)
                loss = criterion(logits, targets.long())

            scaler.scale(loss).backward()
            losses += loss.item()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        losses = losses / len(train_loader)
        s = (f'Epoch: {epoch:03d}/{NUM_EPOCHS:03d} | '
             # f'Batch {batch_idx:04d}/'
             f'{len(train_loader):04d} | '
             f'Loss: {losses:.4f}')
        print(s)
        with open(LOGFILE_PATH, 'a') as f:
            f.write(f'{s}\n')

        model.eval()
        with torch.no_grad():
            valid_mae, valid_rmse, f1, accuracy, y_test, y_pred, _, loss = compute_mae_and_rmse(  # 直接计算mae、rmse
                model=model,
                data_loader=valid_loader,
                device=DEVICE,
            )

            if loss < best_valid_loss:
                best_valid_loss = loss
                best_epoch = epoch
                torch.save(model.state_dict(), BEST_MODEL_PATH)

            s = (f'loss Current Valid: {loss:.4f} Ep. {epoch}' 
                 f' | Acc Current Valid: {accuracy:.4f} Ep. {epoch}'
                 f' | F1 Current Valid: {f1:.4f} Ep. {epoch}'
                 f' | Best Valid: {best_valid_loss:.4f} Ep. {best_epoch}'
                 )
            s += f'\nTime elapsed: {(time.time() - start_time) / 60:.2f} min'
            print(s)
            with open(LOGFILE_PATH, 'a') as f:
                f.write('%s\n' % s)

    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    with torch.no_grad():
        test_mae, test_rmse, test_f1, accuracy, y_test1, y_pred1, y_pred_prob, _ = compute_mae_and_rmse(
            model=model,
            data_loader=test_loader,
            device=DEVICE,
        )
        y_true_arr = to_numpy(y_test1).reshape(-1)
        y_pred_arr = to_numpy(y_pred1).reshape(-1)
        y_prob_arr = to_numpy(y_pred_prob)

        assert y_prob_arr.ndim == 2, f"y_pred_prob 应该是二维 (n, num_class)，实际是 {y_prob_arr.shape}"
        num_class = y_prob_arr.shape[1]

        result_df = pd.DataFrame({
            'y_true': y_true_arr,
            'y_pred': y_pred_arr,
        })
        for c in range(num_class):
            result_df[f'y_proba_class{c}'] = y_prob_arr[:, c]

        result_df.to_csv('./data_v6/test_predictions_class3_CNN.csv', index=False)
        print(
            f"已保存 {len(result_df)} 条测试集预测结果到 /data_v6/test_predictions_class3_CNN.csv，共 {num_class} 个类别")
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
        plt.savefig('./data_v7/confusion_matrix.png', dpi=300, bbox_inches='tight')


    s = ('\n\n=========================================\n\n'  # 最后跑所有的数据集
         'Performance of best model based on validation set MAE:\n'
         f'Test  MAE / RMSE / F1: {test_mae:.4f} / {test_rmse:.4f} / {test_f1}')

    print(s)
