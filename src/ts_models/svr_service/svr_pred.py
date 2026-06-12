import numpy as np
import pandas as pd
from sklearn.svm import SVR

from src.ts_models.ts_utils.timeseries_utils import (
    split_sequence,
    create_x_input,
    make_predictions_np_input
)

DEFAULT_SVR_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "epsilon": 0.1,
    "gamma": "scale",
    "shrinking": True
}



def clean_dataframe(df):
    df = df.copy()

    cols_to_drop = []

    for col in df.columns:
        invalid_mask = df[col].isna() | np.isinf(pd.to_numeric(df[col], errors="coerce"))

        invalid_count = invalid_mask.sum()
        invalid_ratio = invalid_count / len(df)

        if invalid_count == len(df):
            cols_to_drop.append(col)
        elif invalid_ratio > 0.5:
            cols_to_drop.append(col)
        elif invalid_count > 0:
            df = df.loc[~invalid_mask]

    df = df.drop(columns=cols_to_drop)

    return df.reset_index(drop=True)


def SVR_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger,
        params=None
):
    params = params or DEFAULT_SVR_PARAMS

    logger.info("START SVR_forecast")

    df_test[col_target] = "pass"

    df_test = clean_dataframe(df_test)
    df_train = clean_dataframe(df_train)

    df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
    df_train = df_train.sort_values(by=time_column).reset_index(drop=True)

    if col_for_train is None or len(col_for_train) == 0:
        col_for_train = []

    use_features = [col_target] + list(col_for_train)

    logger.info(f"use_features: {use_features}")

    df_train[use_features] = df_train[use_features].replace([np.inf, -np.inf], np.nan)

    df_train[use_features] = df_train[use_features].astype(float)
    df_train[use_features] = df_train[use_features].ffill().bfill()

    df_train[col_for_train] = df_train[col_for_train].astype(float)

    if df_train[use_features].isna().any().any():
        logger.error("NaN in train after preprocessing")
        raise ValueError("NaN values detected in training data")

    values = df_train[use_features].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        bad_idx = np.where(~np.isfinite(values))
        logger.error(f"Non-finite in train values at {bad_idx}")
        raise ValueError("Train contains inf or NaN")

    X, y = split_sequence(values, lag)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if not np.isfinite(X).all():
        bad_idx = np.where(~np.isfinite(X))
        logger.error(f"Non-finite in X at {bad_idx}")
        raise ValueError("X contains inf or NaN")

    if not np.isfinite(y).all():
        bad_idx = np.where(~np.isfinite(y))
        logger.error(f"Non-finite in y at {bad_idx}")
        raise ValueError("y contains inf or NaN")

    X_reshaped = X.reshape(X.shape[0], -1)

    model = SVR(
        kernel=params["kernel"],
        C=params["C"],
        epsilon=params["epsilon"],
        gamma=params["gamma"],
        shrinking=params["shrinking"]
    )

    logger.info("Training SVR")
    model.fit(X_reshaped, y.ravel())
    logger.info("SVR trained")

    if len(col_for_train) == 0:
        test_features = [col_target]
        n_features = 1
    else:
        test_features = use_features
        n_features = len(use_features)

    df_test_values = df_test[test_features].replace([np.inf, -np.inf], np.nan).astype(np.float64)

    x_input = create_x_input(
        df_train[test_features].astype(np.float64),
        lag
    ).astype(np.float64)

    if not np.isfinite(x_input).all():
        bad_idx = np.where(~np.isfinite(x_input))
        logger.error(f"Non-finite in x_input at {bad_idx}")
        raise ValueError("x_input contains inf or NaN")

    x_input = x_input.reshape((1, lag, n_features))

    count_pred_points = len(df_test_values)

    df_test_values_np = df_test_values.to_numpy()

    predict_values = make_predictions_np_input(
        x_input=x_input,
        x_future=df_test_values_np,
        n_features=n_features,
        model=model,
        lag=lag,
        count_pred_points=count_pred_points
    )

    predict_values = np.asarray(predict_values).ravel()

    if not np.isfinite(predict_values).all():
        bad_idx = np.where(~np.isfinite(predict_values))
        logger.error(f"Non-finite in predictions at {bad_idx}")
        raise ValueError("Predictions contain inf or NaN")

    df_test_pred = df_test[[time_column, col_target]].copy()
    df_test_pred[col_target] = predict_values

    logger.info("FINISH SVR_forecast")

    return df_test_pred