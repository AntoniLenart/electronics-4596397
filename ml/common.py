import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2 as chi2_dist, binomtest
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = ['Normal', 'SYN', 'UDP', 'ICMP', 'GoldenEye']
MODEL_NAMES = ['LogisticRegression', 'SVM', 'RandomForest', 'XGBoost']

FEATURE_EXCLUDE = [
    'timestamp', 'ts_epoch', 'sec', 'campaign', 'window_id', 'window_kind', 'window_type',
    'attack', 'attack_syn', 'attack_udp', 'attack_icmp', 'attack_goldeneye',
    'time_bucket',
]


def load_dataset(path):
    """Load a dataset CSV sorted by time (timestamp if present, else ts_epoch)."""
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
    elif 'ts_epoch' in df.columns:
        df = df.sort_values('ts_epoch').reset_index(drop=True)
    return df


def feature_columns(df):
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]


def multiclass_labels(df):
    """0=Normal,1=SYN,2=UDP,3=ICMP,4=GoldenEye from the one-hot attack columns."""
    y = np.zeros(len(df), dtype=int)
    y[df['attack_syn'] == 1] = 1
    y[df['attack_udp'] == 1] = 2
    y[df['attack_icmp'] == 1] = 3
    y[df['attack_goldeneye'] == 1] = 4
    return y


def window_groups(df, y_multi_ground_truth):
    """Return an integer group id per row.

    Prefer the exact ground-truth ``window_id`` column; otherwise fall back to
    maximal contiguous runs of identical ground-truth multiclass label (the v2
    heuristic), which is why small gaps in 1 Hz polling must sit *inside* a
    labelled window, not at its edge.
    """
    if 'window_id' in df.columns and df['window_id'].notna().all():
        return df['window_id'].astype(int).values
    label_change = (pd.Series(y_multi_ground_truth) != pd.Series(y_multi_ground_truth).shift(1)).fillna(True)
    return label_change.cumsum().values


def window_class_map(groups, y_multi):
    """Map each window id -> its (first-row) multiclass label."""
    out = {}
    for w in np.unique(groups):
        out[w] = int(y_multi[groups == w][0])
    return out


def stratified_window_split(groups, y_multi, test_frac=0.3, seed=42):
    """Assign ~test_frac of each class's WINDOWS to the test side so every class
    (including the rare ones) appears on both sides. Returns (train_idx, test_idx,
    train_windows, test_windows)."""
    rng = np.random.RandomState(seed)
    wclass = window_class_map(groups, y_multi)
    window_ids = list(wclass.keys())

    train_windows, test_windows = [], []
    for c in range(len(CLASS_NAMES)):
        wins = [w for w in window_ids if wclass[w] == c]
        rng.shuffle(wins)
        if not wins:
            continue
        n_test = max(1, round(len(wins) * test_frac)) if len(wins) > 1 else 0
        test_windows.extend(wins[:n_test])
        train_windows.extend(wins[n_test:])

    train_mask = np.isin(groups, train_windows)
    test_mask = np.isin(groups, test_windows)
    return (np.where(train_mask)[0], np.where(test_mask)[0],
            set(train_windows), set(test_windows))


def make_estimator(model_name, task, y_train):
    """Construct one of the four classifiers, consistently class-balanced."""
    if model_name == 'LogisticRegression':
        return LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42, n_jobs=-1)
    if model_name == 'SVM':
        return SVC(kernel='rbf', class_weight='balanced', probability=False, random_state=42)
    if model_name == 'RandomForest':
        return RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    if model_name == 'XGBoost':
        if task == 'binary':
            spw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
            return XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', scale_pos_weight=spw)
        return XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss',
                             num_class=len(CLASS_NAMES), objective='multi:softprob')
    raise ValueError(model_name)


MODEL_GRIDS = {
    'LogisticRegression': {
        'binary': dict(grid={'C': [0.001, 0.01, 0.1, 1, 10, 100]}, scoring='f1'),
        'multiclass': dict(grid={'C': [0.001, 0.01, 0.1, 1, 10, 100]}, scoring='f1_macro'),
    },
    'SVM': {
        'binary': dict(grid={'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}, scoring='f1'),
        'multiclass': dict(grid={'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}, scoring='f1_macro'),
    },
    'RandomForest': {
        'binary': dict(grid={'n_estimators': [200], 'max_depth': [4, 8, 12, 20], 'bootstrap': [True, False]}, scoring='f1'),
        'multiclass': dict(grid={'n_estimators': [100, 200], 'max_depth': [4, 8, 12, 20], 'bootstrap': [True, False]}, scoring='f1_macro'),
    },
    'XGBoost': {
        'binary': dict(grid={'n_estimators': [100, 200], 'max_depth': [3, 5, 7, 10], 'learning_rate': [0.05, 0.1, 0.2]}, scoring='f1'),
        'multiclass': dict(grid={'n_estimators': [100, 200], 'max_depth': [3, 5, 7, 10], 'learning_rate': [0.05, 0.1, 0.2]}, scoring='f1_macro'),
    },
}


def make_search(model_name, task, y_train, inner_cv):
    """GridSearchCV over a StandardScaler+estimator Pipeline, so the scaler is refit
    inside each inner fold. Pass unscaled X to fit/predict."""
    cfg = MODEL_GRIDS[model_name][task]
    est = make_estimator(model_name, task, y_train)
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', est)])
    grid = {'clf__' + k: v for k, v in cfg['grid'].items()}
    return GridSearchCV(pipe, grid, cv=inner_cv, scoring=cfg['scoring'], n_jobs=-1, refit=True)


def mcnemar_test(y_true, pred_a, pred_b):
    """Paired McNemar's test on correct/incorrect outcomes (binary or multiclass;
    Dietterich 1998). a = original-label model, b = corrected-label model.
    Exact binomial when the discordant count is small (<25), else the
    continuity-corrected chi-square."""
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))
    n_disc = n01 + n10
    if n_disc == 0:
        return dict(n01=n01, n10=n10, n_discordant=0, statistic=np.nan, p_value=1.0, method='degenerate')
    if n_disc < 25:
        res = binomtest(min(n01, n10), n_disc, 0.5, alternative='two-sided')
        return dict(n01=n01, n10=n10, n_discordant=n_disc, statistic=np.nan, p_value=res.pvalue, method='exact')
    stat = (abs(n01 - n10) - 1) ** 2 / n_disc
    p = 1 - chi2_dist.cdf(stat, df=1)
    return dict(n01=n01, n10=n10, n_discordant=n_disc, statistic=stat, p_value=p, method='chi2_corrected')
