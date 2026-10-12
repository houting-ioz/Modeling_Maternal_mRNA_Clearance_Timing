import argparse
import os
import time
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score, average_precision_score, matthews_corrcoef, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from load_data2D import parse_cmdline_args
from load_data2D import compute_mae_and_rmse
from load_data2D import get_dataloaders
from model import dlModel
import random
import numpy as np

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


NUM_CLASSES = 2  # 这里还没有修改, 需要重新做数据
NUM_FEATURES = args.in_dim  # 95 one-hot是4维的

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
    """兼容 torch.Tensor / list / numpy，统一转成 numpy 一维数组"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


if __name__ == '__main__':
    if RANDOM_SEED is not None:
        torch.manual_seed(RANDOM_SEED)

    model = dlModel(embed_dim=args.in_dim, num_dim=12, hidden_dim=256, num_classes=2, encoder_type=args.encoder_type,
                    pool_type=args.pool_type, layers=args.layers)
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
        train_csv_path='../data/train.csv',
        valid_csv_path='../data/val.csv',
        test_csv_path='../data/test.csv',
        balanced=True,  # True
        num_workers=NUM_WORKERS,
        num_classes=NUM_CLASSES,
        pt_path=args.pt_path)

    criterion = torch.nn.BCEWithLogitsLoss()
    best_valid_loss = torch.tensor(float('inf'))
    s = (f'Script: {__file__}\n'
         f'PyTorch version: {torch.__version__}\n'
         f'Device: {DEVICE}\n'
         f'Learning rate: {LEARNING_RATE}\n'
         f'Batch size: {BATCH_SIZE}\n')
    print(s)
    with open(LOGFILE_PATH, 'w') as f:
        f.write(f'{s}\n')

    scaler = torch.amp.GradScaler('cuda')
    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        losses = 0.0
        model.train()
        for batch_idx, (embed, mask, features, targets) in enumerate(
                train_loader):
            # dna_embed, dna_mask, num_feat, targets
            features = features.to(DEVICE)  # 95维度的
            embed = embed.to(DEVICE)
            mask = mask.to(DEVICE)
            targets = targets.to(DEVICE)  # 是0开始的标签

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(seq=embed, mask=mask, num_feat=features)
                loss = criterion(logits, targets.float())

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

    # Final
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    with torch.no_grad():

        # train_mae, train_rmse, train_f1, accuracy, y_test, y_pred, _, _ = compute_mae_and_rmse(  # 计算的指标不够
        #     model=model,
        #     data_loader=train_loader,
        #     device=DEVICE,
        #     )
        #
        # valid_mae, valid_rmse, valid_f1, accuracy, y_test11, y_pred11, _, _ = compute_mae_and_rmse(
        #     model=model,
        #     data_loader=valid_loader,
        #     device=DEVICE,
        #     )

        test_mae, test_rmse, test_f1, accuracy, y_test1, y_pred1, y_pred_prob, _ = compute_mae_and_rmse(
            model=model,
            data_loader=test_loader,
            device=DEVICE,
        )

        y_true_arr = to_numpy(y_test1).reshape(-1)
        y_pred_arr = to_numpy(y_pred1).reshape(-1)
        y_prob_arr = to_numpy(y_pred_prob)

        # 如果是softmax两类输出 (n, 2)，取正类(第1列)概率；如果已经是sigmoid输出 (n,)，直接用
        if y_prob_arr.ndim == 2:
            y_prob_arr = y_prob_arr[:, 1]
        else:
            y_prob_arr = y_prob_arr.reshape(-1)

        result_df = pd.DataFrame({
            'y_true': y_true_arr,
            'y_pred': y_pred_arr,
            'y_proba': y_prob_arr,
        })
        result_df.to_csv('./data_v6/test_predictions_class2_CNN.csv', index=False)
        print(f"已保存 {len(result_df)} 条测试集预测结果到 /data_v6/test_predictions_class2_CNN.csv")

        print(classification_report(y_test1, y_pred1, digits=3,
                                    target_names=['class_0', 'class_1']))

        metrics = {
            'auc': roc_auc_score(y_test1, y_pred_prob, sample_weight=None),
            'logloss': log_loss(y_test1, y_pred_prob, sample_weight=None),
            'accuracy': accuracy_score(y_test1, y_pred1, sample_weight=None),
            'f1': f1_score(y_test1, y_pred1, sample_weight=None),
            'aupr': average_precision_score(y_test1, y_pred_prob, sample_weight=None),
            'mcc': matthews_corrcoef(y_test1, y_pred1, sample_weight=None),
        }

        # print(f"Macro-F1: {f1_score(y_test111, y_pred111, average='macro'):.3f}")
        print(f"\n AUC (test): {metrics['auc']:.4f}")
        print(f"\n logloss (test): {metrics['logloss']:.4f}")
        print(f"\n ACC (test): {metrics['accuracy']:.4f}")
        print(f"\n F1 (test): {metrics['f1']:.4f}")
        print(f"\n AUPR (test): {metrics['aupr']:.4f}")
        print(f"\n MCC (test): {metrics['mcc']:.4f}")

        cm = confusion_matrix(y_test1, y_pred1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'pred_{i}' for i in range(NUM_CLASSES)],
                    yticklabels=[f'true_{i}' for i in range(NUM_CLASSES)])
        plt.ylabel('True');
        plt.xlabel('Predicted')
        plt.savefig('./data_v6/confusion_matrix.png', dpi=300, bbox_inches='tight')

        # plt.show()

        # xgb.plot_importance(model, max_num_features=20, importance_type='gain')
        # plt.tight_layout()
        # plt.savefig('./data_v5_3utr/importance_matrix.png', dpi=300, bbox_inches='tight')

    s = ('\n\n=========================================\n\n'  # 最后跑所有的数据集
         'Performance of best model based on validation set MAE:\n'
         # f'Train MAE / RMSE / F1: {train_mae:.4f} / {train_rmse:.4f} / {train_f1}\n'
         # f'Valid MAE / RMSE / F1: {valid_mae:.4f} / {valid_rmse:.4f} / {valid_f1}\n'
         f'Test  MAE / RMSE / F1: {test_mae:.4f} / {test_rmse:.4f} / {test_f1}')

    print(s)


