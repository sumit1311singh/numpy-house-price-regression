"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    col_means = np.nanmean(X, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)

    nan_rows, nan_cols = np.where(np.isnan(X))
    X[nan_rows, nan_cols] = col_means[nan_cols]

    return X

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3-q1

    lower = q1-k*iqr
    upper = q3+k*iqr

    return (lower, upper)

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    X_clipped = np.clip(X, lower, upper)
    return X_clipped

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    return numerator/(denominator+eps)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    y = X.copy()
    return np.column_stack([y, col])

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    labels = np.asarray(labels)

    unique_labels = np.unique(labels)
    
    c = len(unique_labels)

    encoded_labels = labels[:, None] == unique_labels[None, :]
    
    return encoded_labels.astype(float)

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)

    x_std = np.where(x_std==0, 1.0, x_std)

    return x_mean, x_std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    return (X-mean)/std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    bias = np.ones((X.shape[0], 1))
    Xb = np.concatenate([bias, X], axis=1)
    return Xb

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    samples = np.arange(n_samples)

    np.random.seed(seed)
    idx = np.random.permutation(n_samples)
    
    samples = samples[idx]
    
    return samples

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    n = len(indices)
    
    train_end = int(n*train_ratio)
    val_end = train_end + int(n*val_ratio)

    train_idx = indices[:train_end]
    val_idx =  indices[train_end:val_end]
    test_idx = indices[val_end:]

    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    x_sub = X[indices]
    y_sub = y[indices]

    return x_sub, y_sub

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    theta = np.linalg.pinv(X) @ y
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X @ theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return np.mean(np.abs(y_pred-y_true))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    return np.sqrt(np.mean((y_pred-y_true)**2))

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    
    ss_res = sum(((y_pred-y_true)**2))
    ss_tot = sum(((y_true-np.mean(y_true))**2))

    if ss_tot == 0.0:
        return 0.0

    return 1 - ss_res/ss_tot

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    residuals = y_true-y_pred
    
    r_mean = np.mean(residuals)
    r_std = np.std(residuals)
    r_median = np.median(np.abs(residuals))

    residual_summary_dict = dict()
    residual_summary_dict['mean']=r_mean
    residual_summary_dict['std']=r_std
    residual_summary_dict['median_abs']=r_median
    
    return residual_summary_dict

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    X_imputed = impute_nan_with_mean(X)

    lower, upper = compute_iqr_bounds(X_imputed, iqr_k)

    X_imputed_clipped = clip_columns(X_imputed, lower, upper)

    return X_imputed_clipped

# Step 20 - assemble_feature_matrix
import numpy as np

def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    ratio = make_ratio_feature(X_num[:, ratio_num_idx], X_num[:, ratio_den_idx])

    X_appended = append_column(X_num, ratio)

    if cat_labels is not None:
        encoded_labels = one_hot_encode(cat_labels)
        X_appended = append_column(X_appended, encoded_labels)

    return X_appended

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    n = X.shape[0]

    samples = make_shuffled_indices(n, seed)

    train_idx, val_idx, test_idx = partition_indices(samples, train_ratio, val_ratio)

    X_train, y_train = subset_xy(X, y, train_idx)
    X_val, y_val = subset_xy(X, y, val_idx)
    X_test, y_test = subset_xy(X, y, test_idx)

    split_data_dict = dict()
    split_data_dict['X_train'] = X_train
    split_data_dict['y_train'] = y_train
    split_data_dict['X_val'] = X_val
    split_data_dict['y_val'] = y_val
    split_data_dict['X_test'] = X_test
    split_data_dict['y_test'] = y_test

    return split_data_dict

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    # TODO: Fit standardizer on train, transform all splits, prepend bias...
    x_mean, x_std = fit_standardizer(splits['X_train'])

    X_train_standardized = apply_standardizer(splits['X_train'], x_mean, x_std)
    X_val_standardized = apply_standardizer(splits['X_val'], x_mean, x_std)
    X_test_standardized = apply_standardizer(splits['X_test'], x_mean, x_std)

    X_train_b = add_bias_column(X_train_standardized)
    X_val_b = add_bias_column(X_val_standardized)
    X_test_b = add_bias_column(X_test_standardized)

    std_splits = dict()
    std_splits['X_train'] = X_train_b
    std_splits['y_train'] = splits['y_train']
    std_splits['X_val'] = X_val_b
    std_splits['y_val'] = splits['y_val']
    std_splits['X_test'] = X_test_b
    std_splits['y_test'] = splits['y_test']      

    return std_splits, x_mean, x_std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r_squared(y_true, y_pred)
    rs = residual_summary(y_true, y_pred)

    metrics_dict = dict()
    metrics_dict['mae'] = mae
    metrics_dict['rmse'] = rmse
    metrics_dict['r2'] = r2
    metrics_dict['residual_summary'] = rs

    return metrics_dict

# Step 24 - house_price_pipeline
def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    # TODO: Run full clean->featurize->split->standardize->OLS->evaluate pipeline...
    X_clean = prepare_cleaned_features(X, iqr_k=iqr_k)

    X_feat = assemble_feature_matrix(X_clean, ratio_num_idx, ratio_den_idx, cat_labels=cat_labels)

    splits = make_train_val_test(X_feat, y, train_ratio, val_ratio, seed)

    std_splits, _, _ = standardize_and_add_bias(splits)

    theta = ols_fit(std_splits['X_train'], std_splits['y_train'])

    y_val_hat = ols_predict(std_splits['X_val'], theta)
    y_test_hat = ols_predict(std_splits['X_test'], theta)

    val_metrics = evaluate_predictions(std_splits['y_val'], y_val_hat)
    test_metrics = evaluate_predictions(std_splits['y_test'], y_test_hat)\
    
    final_metrics = dict()
    final_metrics['theta'] = theta
    final_metrics['y_test'] = std_splits['y_test']
    final_metrics['y_test_pred'] = y_test_hat
    final_metrics['test_metrics'] = test_metrics
    final_metrics['val_metrics'] = val_metrics

    return final_metrics

