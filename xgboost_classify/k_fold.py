"""
高效提取 1-7mer 特征：稀疏表示 + 自动词表裁剪
================================================

问题根源
--------
1到7阶 k-mer 理论组合数是 4^1+...+4^7 = 21,844（这正是 Saluki 论文里用的维度）。
但你原来的写法对每条序列都：
    1. 先枚举出该阶所有 4^k 种可能的 k-mer 作为字典的 key
    2. 逐位置滑窗计数
    3. 除以总数得到密集向量

问题在于：
    - 绝大多数长 k-mer（尤其 k=5,6,7）在一条几百 nt 的序列里根本不会出现，
      密集向量里 99% 以上都是 0，纯粹浪费内存
    - 每条序列都要重新枚举一次完整的 4^k 空间，序列数一多，速度和内存都会爆炸

解决思路
--------
不要"先生成完整密集向量再筛选"，而是从一开始就用稀疏矩阵表示，并且在构建词表阶段
就用 min_df（最小文档频率）自动剔除几乎不出现的 k-mer —— 这是 NLP 里处理超大词表
(比如字符 n-gram) 的标准做法，直接复用即可。

sklearn 的 CountVectorizer 配合 analyzer='char', ngram_range=(1,7) 可以在一次
调用里完成"滑窗提取所有 1 到 7 阶 k-mer + 稀疏矩阵存储 + 词表频率过滤"，比手写
循环快得多，也不需要显式枚举 4^k 空间。
"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def extract_kmer_features_sparse(
    sequences,
    k_range=(1, 7),
    min_df=5,
    max_df=0.95,
    max_features=None,
):
    """
    高效提取 k_range 范围内所有 k-mer 的频率特征，返回稀疏矩阵。

    Parameters
    ----------
    sequences : list[str]
        RNA 序列列表（内部会自动转大写并把 T 换成 U，和你原函数行为一致）
    k_range : (int, int)
        (最小k, 最大k)，例如 (1,7) 表示提取 1-mer 到 7-mer
    min_df : int or float
        某个 k-mer 至少要在多少条序列里出现过，才会被纳入词表。
        这是控制维度最有效的一步——大部分长 k-mer 只在个别序列里出现一两次，
        min_df=5 意味着"至少要在5条不同序列里出现过"才保留，能把词表从
        理论上的 21,844 维大幅裁剪到实际有意义的规模（通常能压到几千维以内）。
        可以传整数（绝对出现序列数）或 0~1 的小数（出现序列比例）。
    max_df : float
        某个 k-mer 如果几乎在所有序列里都出现（比如 > 95%），说明它区分度很低
        （常见的短 k-mer 比如单个 'A' 几乎每条序列都有），也一并过滤掉。
    max_features : int or None
        额外硬性限制最终保留的最大特征数，按全局出现频率取 top-N。
        如果你需要一个确定的、不随数据变化的维度上限（比如要固定成 2048），
        用这个参数比 min_df 更直接。

    Returns
    -------
    X : scipy.sparse.csr_matrix, shape (n_sequences, n_selected_kmers)
        每行是一条序列的 k-mer 频率向量（已按每条序列的总 k-mer 数归一化）
    vectorizer : CountVectorizer
        训练好的 vectorizer，可用 vectorizer.get_feature_names_out() 拿到
        每一列对应的具体 k-mer 字符串
    """
    seqs_clean = [s.upper().replace('T', 'U') for s in sequences]

    vectorizer = CountVectorizer(
        analyzer='char',
        ngram_range=k_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        lowercase=False,
    )
    try:
        counts = vectorizer.fit_transform(seqs_clean)  # 稀疏矩阵，原始计数  传进去的seq list
    except ValueError:
        # 常见于单独处理很小的k（比如k=1只有4种、k=2只有16种可能）时：
        # 序列只要够长，几乎每条都会包含全部这几种k-mer，document frequency
        # 逼近100%，超过 max_df 阈值，导致该阶全部特征被过滤、词表清空报错。
        # 这种情况下 max_df 过滤本身没有意义（低阶k-mer的信息量本来就在于
        # 频率高低的差异，而不在于"是否出现过"），所以自动退化为不做 max_df
        # 过滤，只保留 min_df 这一道筛选。
        vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=k_range,
            min_df=min_df,
            max_df=1.0,
            max_features=max_features,
            lowercase=False,
        )
        counts = vectorizer.fit_transform(seqs_clean)

    # 归一化：每条序列内部，同一阶 k 的计数应该除以该阶的总 k-mer 数
    # CountVectorizer 把不同阶的 k-mer 混在了一起，这里做一个更简单但足够合理的
    # 近似处理：对每行整体做 L1 归一化（行内所有特征加起来=1），
    # 如果你需要严格按"每阶分别归一化"，可以对每个 k 单独调用本函数再拼接，
    # 见下面 extract_kmer_features_by_order 函数。
    X = normalize(counts, norm='l1', axis=1)

    return X, vectorizer


def extract_kmer_features_by_order(
    sequences,
    k_list=(1, 2, 3, 4, 5, 6, 7),
    min_df=5,
    max_df=0.95,
    max_features_per_k=None,
):
    """
    和上面的区别：严格按每一阶 k 分别构建词表、分别归一化（每阶内部频率和为1，
    这和你原来的实现语义完全一致），最后再横向拼接成一个大稀疏矩阵。
    如果你在意"归一化必须严格按阶分开"这个细节，用这个函数。

    max_features_per_k : int or None
        每一阶最多保留多少个 k-mer（而不是全局限制），例如 k=7 时可以设
        max_features_per_k=500，即使理论上有 16384 种，也只保留出现频率
        最高的 500 种，其余归到 "低频k-mer" 被过滤掉。
    """
    blocks = []
    all_names = []
    for k in k_list:
        X_k, vec_k = extract_kmer_features_sparse(
            sequences,
            k_range=(k, k),
            min_df=min_df,
            max_df=max_df,
            max_features=max_features_per_k,
        )
        blocks.append(X_k)
        all_names += [f"k{k}_{name}" for name in vec_k.get_feature_names_out()]

    X_all = sp.hstack(blocks, format='csr')
    return X_all, all_names


def compress_with_svd(X_sparse, n_components=100, random_state=0):
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


"""
XGBClassifier 十折交叉验证
==========================

为什么不能直接用 sklearn 的 cross_val_score / cross_validate：
    - early_stopping_rounds 需要每一折单独的验证集（eval_set），不能全局共用一个
    - sample_weight 也需要按每一折的训练子集重新切片，cross_val_score 的
      fit_params 机制不支持"每折自动切片"这种操作
    - 这两点合在一起，标准做法是自己写 StratifiedKFold 手动循环

用 StratifiedKFold 而不是普通 KFold：
    因为是二分类任务，且你之前那批miRNA数据显示类别本身不算极端不平衡
    但也有一定偏斜（约3:1），分层抽样能保证每一折里正负样本比例和整体一致，
    避免某一折偶然抽到类别失衡更严重的子集导致评估结果波动。
"""
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score, average_precision_score, matthews_corrcoef
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight


def _take_rows(X, indices):
    """按位置提取行，同时兼容 pandas、NumPy 和 scipy 稀疏矩阵。"""
    return X.iloc[indices] if hasattr(X, 'iloc') else X[indices]


def _make_stratified_folds(X, y, n_splits, random_state):
    """固定分层折；每个数组恰好对应一个完整的验证/测试折。"""
    if n_splits < 3:
        raise ValueError("n_splits 至少为3，才能分别设置训练折、验证折和测试折。")
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    return [fold_idx for _, fold_idx in skf.split(X, y)]


XGB_SEARCH_SPACE = {
    'max_depth': {'kind': 'int', 'low': 2, 'high': 8},
    'learning_rate': {'kind': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
    'min_child_weight': {'kind': 'float', 'low': 1.0, 'high': 20.0, 'log': True},
    'subsample': {'kind': 'float', 'low': 0.6, 'high': 1.0, 'log': False},
    'colsample_bytree': {'kind': 'float', 'low': 0.5, 'high': 1.0, 'log': False},
    'gamma': {'kind': 'float', 'low': 0.0, 'high': 5.0, 'log': False},
    'reg_alpha': {'kind': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
    'reg_lambda': {'kind': 'float', 'low': 1e-3, 'high': 100.0, 'log': True},
}


def _suggest_xgb_params(trial):
    """Optuna TPE搜索空间；不搜索n_estimators，由早停确定树数。"""
    suggestions = {}
    for name, spec in XGB_SEARCH_SPACE.items():
        if spec['kind'] == 'int':
            suggestions[name] = trial.suggest_int(name, spec['low'], spec['high'])
        else:
            suggestions[name] = trial.suggest_float(
                name,
                spec['low'],
                spec['high'],
                log=spec['log'],
            )
    return suggestions


def _best_n_estimators(model, fallback):
    """将XGBoost从0开始的best_iteration转换为树的数量。"""
    best_iteration = getattr(model, 'best_iteration', None)
    return best_iteration + 1 if best_iteration is not None else fallback


def _optimize_xgb_on_validation(
    X_train,
    y_train,
    X_validation,
    y_validation,
    base_params,
    task,
    n_trials,
    random_state,
    sample_weight=None,
    timeout=None,
):
    """仅用固定训练折和验证折执行TPE搜索，绝不访问外层测试折。"""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "启用Optuna参数优化需要先安装optuna：pip install optuna"
        ) from exc

    if n_trials < 1:
        raise ValueError("n_trials 必须至少为1。")
    if task not in {'binary', 'multiclass'}:
        raise ValueError("task 必须是'binary'或'multiclass'。")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    startup_trials = min(10, max(1, n_trials // 3))
    sampler = optuna.samplers.TPESampler(
        seed=random_state,
        n_startup_trials=startup_trials,
    )
    study = optuna.create_study(direction='maximize', sampler=sampler)

    def objective(trial):
        candidate_params = base_params.copy()
        candidate_params.update(_suggest_xgb_params(trial))
        candidate_params['eval_metric'] = (
            'logloss' if task == 'binary' else 'mlogloss'
        )

        model = XGBClassifier(**candidate_params)
        fit_kwargs = {
            'eval_set': [(X_validation, y_validation)],
            'verbose': False,
        }
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        model.fit(X_train, y_train, **fit_kwargs)

        n_estimators = _best_n_estimators(
            model, candidate_params.get('n_estimators', 100)
        )
        trial.set_user_attr('best_n_estimators', int(n_estimators))

        validation_proba = model.predict_proba(X_validation)
        if task == 'binary':
            return roc_auc_score(y_validation, validation_proba[:, 1])
        return roc_auc_score(
            y_validation,
            validation_proba,
            multi_class='ovr',
            average='macro',
        )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1,
        show_progress_bar=False,
    )

    best_params = base_params.copy()
    best_params.update(study.best_params)
    best_params['eval_metric'] = (
        'logloss' if task == 'binary' else 'mlogloss'
    )
    best_n_estimators = int(
        study.best_trial.user_attrs['best_n_estimators']
    )
    trials = [
        {
            'number': trial.number,
            'value': trial.value,
            'state': trial.state.name,
            'params': dict(trial.params),
            'best_n_estimators': trial.user_attrs.get('best_n_estimators'),
        }
        for trial in study.trials
    ]
    final_best_params = best_params.copy()
    final_best_params.pop('early_stopping_rounds', None)
    final_best_params['n_estimators'] = best_n_estimators
    return best_params, best_n_estimators, {
        'best_validation_score': float(study.best_value),
        'best_params': final_best_params,
        'searched_params': dict(study.best_params),
        'best_n_estimators': best_n_estimators,
        'trials': trials,
    }

def cross_validate_xgb(
    X, y,
    n_splits=10, random_state=42,
    xgb_params=None,
    verbose=50,
    optimize_hyperparameters=False,
    n_trials=30,
    optuna_timeout=None,
):
    """
    对给定的 XGBClassifier 配置做分层交叉验证。预先固定全部折；每轮取1折
    作为测试集、相邻1折作为早停验证集，其余折用于训练。早停确定 boosting
    轮数后，用除测试折外的全部折重新拟合模型。n_splits=10 时即为8:1:1。

    Parameters
    ----------
    X : array-like or pd.DataFrame, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
        二分类标签 (0/1)
    sample_weight : array-like or None, shape (n_samples,)
        和你原来 fit() 里传的 sample_weight 含义一致，会按每折的训练/验证
        索引自动切片，不需要你手动处理
    n_splits : int
        折数，默认10
    xgb_params : dict or None
        传给 XGBClassifier 的超参数字典，不传就用你原来那套默认配置
    verbose : int
        每隔多少轮打印一次训练日志，传 False 可完全静音
    optimize_hyperparameters : bool
        是否在每个外层折的8折训练/1折验证上运行Optuna TPE搜索
    n_trials : int
        每个外层折的TPE试验次数，默认30
    optuna_timeout : float or None
        每个外层折允许的最大搜索秒数；None表示不限制

    Returns
    -------
    results : dict
        包含每折的验证集指标（auc/logloss/accuracy/f1）、
        out-of-fold 预测概率、每折训练好的模型列表、
        平均后的特征重要性
    """
    if xgb_params is None:
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
            random_state=random_state,
            n_jobs=-1,
        )
    y = np.asarray(y)
    # 拿到十折里每一折的id
    fold_indices = _make_stratified_folds(X, y, n_splits, random_state)

    fold_metrics = []
    models = []
    oof_pred_proba = np.zeros(len(y))  # out-of-fold 预测，每个样本恰好被预测一次  这个是把每一折的预测结果都拿到，与真实的去做计算画图，
    oof_label = np.zeros(len(y))
    oof_fold = np.zeros(len(y), dtype=int)  # 记录每个样本是在第几折被当作验证集预测出来的，方便按折还原/画图
    feature_importances = []  # 每一折各自的特征重要性（不是平均后的），方便看重要性在折间的波动
    optimization_results = []

    # one_x = None
    # one_y

    for test_fold_idx, test_idx in enumerate(fold_indices):
        fold_idx = test_fold_idx + 1
        early_stop_fold_idx = (test_fold_idx + 1) % n_splits
        early_stop_idx = fold_indices[early_stop_fold_idx]
        inner_train_idx = np.concatenate([
            indices for i, indices in enumerate(fold_indices)
            if i not in (test_fold_idx, early_stop_fold_idx)
        ])
        # 早停完成后，最终模型使用验证折在内的其余 n_splits - 1 折。
        train_idx = np.concatenate([inner_train_idx, early_stop_idx])

        X_train_fold = _take_rows(X, train_idx)
        X_test_fold = _take_rows(X, test_idx)
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        # 早停验证集是预先固定十折中的完整一折，不是再次随机抽样得到的。
        X_inner_train = _take_rows(X, inner_train_idx)
        X_early_stop = _take_rows(X, early_stop_idx)
        y_inner_train = y[inner_train_idx]
        y_early_stop = y[early_stop_idx]

        # 当前 final 版本不启用类别权重；保留原有行为。
        sw_inner_train = None
        sw_train_fold = None
        sw_test_fold = None

        # 第一步：TPE在固定验证折上选参数；关闭时保持原来的单次早停流程。
        if optimize_hyperparameters:
            tuned_params, best_n_estimators, optimization = (
                _optimize_xgb_on_validation(
                    X_inner_train,
                    y_inner_train,
                    X_early_stop,
                    y_early_stop,
                    base_params=xgb_params,
                    task='binary',
                    n_trials=n_trials,
                    random_state=random_state + fold_idx,
                    sample_weight=sw_inner_train,
                    timeout=optuna_timeout,
                )
            )
            optimization['fold'] = fold_idx
            optimization_results.append(optimization)
            print(
                f"[Fold {fold_idx}/{n_splits}] Optuna最佳验证AUC="
                f"{optimization['best_validation_score']:.4f}"
            )
        else:
            tuned_params = xgb_params.copy()
            tuned_params['eval_metric'] = 'logloss'
            early_model = XGBClassifier(**tuned_params)
            fit_kwargs = dict(
                eval_set=[(X_early_stop, y_early_stop)],
                verbose=verbose,
            )
            if sw_inner_train is not None:
                fit_kwargs['sample_weight'] = sw_inner_train
            early_model.fit(X_inner_train, y_inner_train, **fit_kwargs)
            best_n_estimators = _best_n_estimators(
                early_model, tuned_params.get('n_estimators', 100)
            )

        # 第二步：固定最佳轮数，用完整外层训练折重新拟合模型。
        final_params = tuned_params.copy()
        final_params.pop('early_stopping_rounds', None)
        final_params['n_estimators'] = best_n_estimators
        model = XGBClassifier(**final_params)
        final_fit_kwargs = {}
        if sw_train_fold is not None:
            final_fit_kwargs['sample_weight'] = sw_train_fold
        model.fit(X_train_fold, y_train_fold, **final_fit_kwargs)

        # 第三步：外层测试折只在最终模型确定后使用一次。
        test_proba = model.predict_proba(X_test_fold)[:, 1]
        test_pred = model.predict(X_test_fold)
        oof_pred_proba[test_idx] = test_proba
        oof_label[test_idx] = test_pred
        oof_fold[test_idx] = fold_idx

        metrics = {
            'fold': fold_idx,
            'best_iteration': best_n_estimators - 1,
            'n_estimators': best_n_estimators,
            'auc': roc_auc_score(y_test_fold, test_proba, sample_weight=sw_test_fold),
            'logloss': log_loss(y_test_fold, test_proba, sample_weight=sw_test_fold),
            'accuracy': accuracy_score(y_test_fold, test_pred, sample_weight=sw_test_fold),
            'f1': f1_score(y_test_fold, test_pred, sample_weight=sw_test_fold),
            'aupr': average_precision_score(y_test_fold, test_proba, sample_weight=sw_test_fold),
            'mcc': matthews_corrcoef(y_test_fold, test_pred, sample_weight=sw_test_fold),
        }
        fold_metrics.append(metrics)
        models.append(model)
        feature_importances.append(model.feature_importances_)

        print(f"[Fold {fold_idx}/{n_splits}] "
              f"best_iter={metrics['best_iteration']}, "
              f"AUC={metrics['auc']:.4f}, "
              f"logloss={metrics['logloss']:.4f}, "
              f"acc={metrics['accuracy']:.4f}, "
              f"f1={metrics['f1']:.4f}")

    # 汇总平均 ± 标准差
    metric_names = ['auc', 'logloss', 'accuracy', 'f1', 'aupr', 'mcc']  # 还是看十折的结果，下面先不用管
    summary = {}
    for name in metric_names:
        values = [m[name] for m in fold_metrics]
        summary[name] = (np.mean(values), np.std(values))

    print("\n=== 十折交叉验证汇总 (mean ± std) ===")
    for name, (mean_v, std_v) in summary.items():
        print(f"{name:10s}: {mean_v:.4f} ± {std_v:.4f}")

    # out-of-fold 整体指标（比逐折平均更能代表"整个数据集上的真实表现"，
    # 因为每个样本只被预测过一次，不存在折间平均带来的方差膨胀）
    oof_auc = roc_auc_score(y, oof_pred_proba, sample_weight=None)  # sample_weight=sw
    oof_acc = accuracy_score(y, oof_label, sample_weight=None)
    oof_f1 = f1_score(y, oof_label, sample_weight=None)
    oof_aupr = average_precision_score(y, oof_pred_proba, sample_weight=None)
    oof_mcc = matthews_corrcoef(y, oof_label, sample_weight=None)

    print(f"\nOut-of-fold AUC (整体，非逐折平均): {oof_auc:.4f}")
    print(f"\nOut-of-fold ACC (整体，非逐折平均): {oof_acc:.4f}")
    print(f"\nOut-of-fold F1 (整体，非逐折平均): {oof_f1:.4f}")
    print(f"\nOut-of-fold AUPR (整体，非逐折平均): {oof_aupr:.4f}")
    print(f"\nOut-of-fold MCC (整体，非逐折平均): {oof_mcc:.4f}")

    avg_importance = np.mean(feature_importances, axis=0)

    return {
        'x': X,
        'y': y,
        'pred_y': oof_label,
        'oof_fold': oof_fold,  # 每个样本属于第几折（验证集角色）
        'fold_metrics': fold_metrics,  # 每一折各自的指标（auc/logloss/acc/f1/aupr/mcc/best_iteration）
        'summary': summary,
        'oof_pred_proba': oof_pred_proba,
        'oof_auc': oof_auc,
        'oof_acc': oof_acc,
        'oof_f1': oof_f1,
        'models': models,
        'avg_feature_importance': avg_importance,
        'feature_importances': feature_importances,  # 每一折各自的特征重要性（未平均），list长度=n_splits
        'optimization_results': optimization_results,
    }


def cross_validate_xgb_3class(
    X, y, sample_weight=None,
    n_splits=10, random_state=42,
    xgb_params=None,
    verbose=50,
    optimize_hyperparameters=False,
    n_trials=30,
    optuna_timeout=None,
):
    """
    对给定的 XGBClassifier 配置做分层交叉验证。预先固定全部折；每轮取1折
    作为测试集、相邻1折作为早停验证集，其余折用于训练。早停确定 boosting
    轮数后，用除测试折外的全部折重新拟合模型。n_splits=10 时即为8:1:1。

    Parameters
    ----------
    X : array-like or pd.DataFrame, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
        多分类整数标签，必须连续编码为 0, ..., num_class - 1
    sample_weight : array-like or None, shape (n_samples,)
        和你原来 fit() 里传的 sample_weight 含义一致，会按每折的训练/验证
        索引自动切片，不需要你手动处理
    n_splits : int
        折数，默认10
    xgb_params : dict or None
        传给 XGBClassifier 的超参数字典，不传就用你原来那套默认配置
    verbose : int
        每隔多少轮打印一次训练日志，传 False 可完全静音
    optimize_hyperparameters : bool
        是否在每个外层折的8折训练/1折验证上运行Optuna TPE搜索
    n_trials : int
        每个外层折的TPE试验次数，默认30
    optuna_timeout : float or None
        每个外层折允许的最大搜索秒数；None表示不限制

    Returns
    -------
    results : dict
        包含每折的验证集指标（auc/logloss/accuracy/f1）、
        out-of-fold 预测概率、每折训练好的模型列表、
        平均后的特征重要性
    """
    if xgb_params is None:
        xgb_params = dict(
            objective='multi:softprob',  # 二分类修改 'multi:softprob' binary:logistic
            num_class=3,
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
            random_state=random_state,
            n_jobs=-1,
        )

    y = np.asarray(y)
    classes = np.unique(y)
    num_class = len(classes)
    expected_classes = np.arange(num_class)
    if num_class < 3:
        raise ValueError(
            f"cross_validate_xgb_3class requires at least 3 classes; observed {classes.tolist()}."
        )
    if not np.array_equal(classes, expected_classes):
        raise ValueError(
            "Multiclass labels must be consecutive integers from 0 to "
            f"{num_class - 1}; observed {classes.tolist()}."
        )
    configured_num_class = int(xgb_params.get('num_class', num_class))
    if configured_num_class != num_class:
        raise ValueError(
            f"xgb_params['num_class']={configured_num_class} does not match "
            f"the {num_class} classes observed in y."
        )

    fold_indices = _make_stratified_folds(X, y, n_splits, random_state)

    fold_metrics = []
    models = []
    oof_pred_proba = np.zeros((len(y), num_class), dtype=float)
    oof_label = np.empty(len(y), dtype=y.dtype)
    oof_fold = np.zeros(len(y), dtype=int)  # 记录每个样本是在第几折被当作验证集预测出来的
    feature_importances = []  # 每一折各自的特征重要性（不是平均后的）
    optimization_results = []

    for test_fold_idx, test_idx in enumerate(fold_indices):
        fold_idx = test_fold_idx + 1
        early_stop_fold_idx = (test_fold_idx + 1) % n_splits
        early_stop_idx = fold_indices[early_stop_fold_idx]
        inner_train_idx = np.concatenate([
            indices for i, indices in enumerate(fold_indices)
            if i not in (test_fold_idx, early_stop_fold_idx)
        ])
        # 早停完成后，最终模型使用验证折在内的其余 n_splits - 1 折。
        train_idx = np.concatenate([inner_train_idx, early_stop_idx])

        X_train_fold = _take_rows(X, train_idx)
        X_test_fold = _take_rows(X, test_idx)
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        # 早停验证集是预先固定十折中的完整一折，不是再次随机抽样得到的。
        X_inner_train = _take_rows(X, inner_train_idx)
        X_early_stop = _take_rows(X, early_stop_idx)
        y_inner_train = y[inner_train_idx]
        y_early_stop = y[early_stop_idx]

        # 当前 final 版本不启用类别权重；保留原有行为。
        sw_inner_train = None
        sw_train_fold = None
        sw_test_fold = None

        # 第一步：TPE在固定验证折上选参数；关闭时保持原来的单次早停流程。
        if optimize_hyperparameters:
            tuned_params, best_n_estimators, optimization = (
                _optimize_xgb_on_validation(
                    X_inner_train,
                    y_inner_train,
                    X_early_stop,
                    y_early_stop,
                    base_params=xgb_params,
                    task='multiclass',
                    n_trials=n_trials,
                    random_state=random_state + fold_idx,
                    sample_weight=sw_inner_train,
                    timeout=optuna_timeout,
                )
            )
            optimization['fold'] = fold_idx
            optimization_results.append(optimization)
            print(
                f"[Fold {fold_idx}/{n_splits}] Optuna最佳验证macro-AUC="
                f"{optimization['best_validation_score']:.4f}"
            )
        else:
            tuned_params = xgb_params.copy()
            tuned_params['eval_metric'] = 'mlogloss'
            early_model = XGBClassifier(**tuned_params)
            fit_kwargs = dict(
                eval_set=[(X_early_stop, y_early_stop)],
                verbose=verbose,
            )
            if sw_inner_train is not None:
                fit_kwargs['sample_weight'] = sw_inner_train
            early_model.fit(X_inner_train, y_inner_train, **fit_kwargs)
            best_n_estimators = _best_n_estimators(
                early_model, tuned_params.get('n_estimators', 100)
            )

        # 第二步：固定最佳轮数，用完整外层训练折重新拟合模型。
        final_params = tuned_params.copy()
        final_params.pop('early_stopping_rounds', None)
        final_params['n_estimators'] = best_n_estimators
        model = XGBClassifier(**final_params)
        final_fit_kwargs = {}
        if sw_train_fold is not None:
            final_fit_kwargs['sample_weight'] = sw_train_fold
        model.fit(X_train_fold, y_train_fold, **final_fit_kwargs)

        # 第三步：外层测试折只在最终模型确定后使用一次。
        test_proba = model.predict_proba(X_test_fold)
        test_pred = model.predict(X_test_fold)
        oof_pred_proba[test_idx] = test_proba
        oof_label[test_idx] = test_pred
        oof_fold[test_idx] = fold_idx

        metrics = {
            'fold': fold_idx,
            'best_iteration': best_n_estimators - 1,
            'n_estimators': best_n_estimators,
            'mutil_auc': roc_auc_score(
                y_test_fold,
                test_proba,
                multi_class='ovr',
                average='macro',
                sample_weight=sw_test_fold,
            ),
            'logloss': log_loss(
                y_test_fold,
                test_proba,
                labels=classes,
                sample_weight=sw_test_fold,
            ),
            'accuracy': accuracy_score(y_test_fold, test_pred, sample_weight=sw_test_fold),
            'f1': f1_score(y_test_fold, test_pred, average='macro', sample_weight=sw_test_fold),
            'f1_weight': f1_score(y_test_fold, test_pred, average='weighted', sample_weight=sw_test_fold),
        }
        fold_metrics.append(metrics)  # 十组metrics
        models.append(model)
        feature_importances.append(model.feature_importances_)

        print(f"[Fold {fold_idx}/{n_splits}] "
              f"best_iter={metrics['best_iteration']}, "
              f"AUC={metrics['mutil_auc']:.4f}, "
              f"logloss={metrics['logloss']:.4f}, "
              f"acc={metrics['accuracy']:.4f}, "
              f"f1={metrics['f1']:.4f},"
              f"f1_weight={metrics['f1_weight']:.4f}")

    # 汇总平均 ± 标准差
    metric_names = ['mutil_auc', 'logloss', 'accuracy', 'f1', 'f1_weight']
    summary = {}
    for name in metric_names:
        values = [m[name] for m in fold_metrics]  # 拿到十组下的对应的每个指标的list 算均值和方差，全部存起来文件有点多
        summary[name] = (np.mean(values), np.std(values))

    print("\n=== 十折交叉验证汇总 (mean ± std) ===")
    for name, (mean_v, std_v) in summary.items():
        print(f"{name:10s}: {mean_v:.4f} ± {std_v:.4f}")

    # out-of-fold 整体指标（比逐折平均更能代表"整个数据集上的真实表现"，
    # 因为每个样本只被预测过一次，不存在折间平均带来的方差膨胀）
    oof_auc = roc_auc_score(
        y,
        oof_pred_proba,
        multi_class='ovr',
        average='macro',
        sample_weight=None,
    )
    oof_acc = accuracy_score(y, oof_label, sample_weight=None)
    oof_f1 = f1_score(y, oof_label, average='macro', sample_weight=None)
    oof_f1_w = f1_score(y, oof_label, average='weighted', sample_weight=None)
    # oof_mcc = matthews_corrcoef(y, oof_label, sample_weight=None)

    print(f"\nOut-of-fold AUC (整体，非逐折平均): {oof_auc:.4f}")
    print(f"\nOut-of-fold ACC (整体，非逐折平均): {oof_acc:.4f}")
    print(f"\nOut-of-fold F1 (整体，非逐折平均): {oof_f1:.4f}")
    print(f"\nOut-of-fold weighted F1 (整体，非逐折平均): {oof_f1_w:.4f}")
    # print(f"\nOut-of-fold MCC (整体，非逐折平均): {oof_mcc:.4f}")

    avg_importance = np.mean(feature_importances, axis=0)

    return {
        'x': X,
        'y': y,
        'pred_y': oof_label,
        'oof_fold': oof_fold,  # 每个样本属于第几折（验证集角色）
        'fold_metrics': fold_metrics,  # 每一折各自的指标
        'summary': summary,
        'oof_pred_proba': oof_pred_proba,
        'oof_auc': oof_auc,
        'oof_acc': oof_acc,
        'oof_f1': oof_f1,
        'oof_f1_w': oof_f1_w,
        'models': models,
        'avg_feature_importance': avg_importance,
        'feature_importances': feature_importances,  # 每一折各自的特征重要性（未平均），list长度=n_splits
        'optimization_results': optimization_results,
    }
# ----------------------------------------------------------------------
# 自测：合成一份不平衡二分类数据验证流程能跑通
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=10,
        weights=[0.75, 0.25],  # 模拟你数据里 15931:5243 约3:1的类别偏斜
        random_state=42,
    )
    # 模拟 sample_weight：比如按某种置信度加权，这里随机生成示意
    rng = np.random.default_rng(0)
    sample_weights = rng.uniform(0.5, 1.5, size=len(y))

    results = cross_validate_xgb(
        X, y,
        sample_weight=sample_weights,
        n_splits=10,
        verbose=False,  # 自测时静音，实际用时可设50看训练日志
    )

# ----------------------------------------------------------------------
# 自测：验证维度确实被显著压缩，且稀疏矩阵内存占用远小于暴力密集实现
# ----------------------------------------------------------------------
# if __name__ == "__main__":
#     rng = np.random.default_rng(0)
#     bases = ['A', 'U', 'G', 'C']
#
#     # 模拟 2000 条长度在 100-500 之间的序列（量级接近真实 mRNA UTR）
#     seqs = [
#         "".join(rng.choice(bases, size=rng.integers(100, 500)))
#         for _ in range(2000)
#     ]
#
#     print("=== 方案一：全部1-7mer混合，用 min_df 自动裁剪词表 ===")
#     X_mixed, vec = extract_kmer_features_sparse(
#         seqs, k_range=(1, 7), min_df=5, max_df=0.95
#     )
#     print(f"原始理论维度: {sum(4**k for k in range(1, 8))} (即 4^1+...+4^7)")
#     print(f"min_df 裁剪后实际维度: {X_mixed.shape[1]}")
#     print(f"稀疏矩阵非零元素占比: {X_mixed.nnz / (X_mixed.shape[0]*X_mixed.shape[1]):.4%}")
#     print(f"稀疏矩阵内存占用: {X_mixed.data.nbytes / 1e6:.2f} MB")
#     dense_equivalent_mb = X_mixed.shape[0] * X_mixed.shape[1] * 8 / 1e6
#     print(f"若用密集矩阵存储同样维度需要: {dense_equivalent_mb:.2f} MB\n")
#
#     print("=== 方案二：每阶k分别限制最多500个特征 ===")
#     X_by_order, names = extract_kmer_features_by_order(
#         seqs, k_list=(1, 2, 3, 4, 5, 6, 7),
#         min_df=5, max_features_per_k=500,
#     )
#     print(f"最终总维度: {X_by_order.shape[1]}\n")
#
#     print("=== 方案三：在方案一基础上再用 SVD 压缩到 100 维 ===")
#     X_svd, svd_model, explained = compress_with_svd(X_mixed, n_components=100)
#     print(f"压缩后维度: {X_svd.shape[1]}")
#     print(f"保留的方差比例: {explained:.2%}")
