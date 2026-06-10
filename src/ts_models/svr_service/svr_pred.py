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

    df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
    df_train = df_train.sort_values(by=time_column).reset_index(drop=True)
    logger.info(f"After time sort: shape={df_train.shape}")

    if col_for_train is None or len(col_for_train) == 0:
        col_for_train = []

    use_features = [col_target] + list(col_for_train)
    logger.info(f"Use features: {use_features}")

    df_train[use_features] = df_train[use_features].replace([np.inf, -np.inf], np.nan)
    logger.info("Replaced inf with NaN")

    df_train[use_features] = df_train[use_features].astype(float)
    logger.info("Cast to float done")

    df_train[use_features] = df_train[use_features].ffill().bfill()
    logger.info("NaN filled (ffill/bfill)")

    df_train = df_train[use_features].copy()
    logger.info(f"Train reduced: shape={df_train.shape}")

    df_test_pred = df_test[[time_column, col_target]].copy()
    logger.info(f"Test pred init: shape={df_test_pred.shape}")

    if len(col_for_train) > 0:
        df_test = df_test[use_features].copy()
    else:
        df_test = df_test[[col_target]].copy()

    logger.info(f"Test features prepared: shape={df_test.shape}")

    df_train[col_target] = df_train[col_target].replace("None", None).astype(float)

    if df_train.isna().any().any():
        logger.error("NaN detected after preprocessing")
        logger.error(df_train.isna().sum())
        raise ValueError("NaN values detected in training data")

    values = df_train[use_features].astype(np.float64).values
    logger.info(f"Values shape: {values.shape}")

    if not np.isfinite(values).all():
        idx = np.where(~np.isfinite(values))
        logger.error(f"Non-finite values found at indices: {idx}")
        raise ValueError("Input contains inf or too large values")

    X, y = split_sequence(values, lag)
    logger.info(f"Split sequence: X={len(X)}, y={len(y)}")

    X = np.asarray(X).astype(np.float64)
    y = np.asarray(y).astype(np.float64)

    if not np.isfinite(X).all():
        logger.error("Non-finite values in X after split_sequence")
        raise ValueError("X contains inf or NaN")

    if not np.isfinite(y).all():
        logger.error("Non-finite values in y after split_sequence")
        raise ValueError("y contains inf or NaN")

    X_reshaped = X.reshape(X.shape[0], -1)
    logger.info(f"X reshaped: {X_reshaped.shape}")

    model = SVR(
        kernel=params["kernel"],
        C=params["C"],
        epsilon=params["epsilon"],
        gamma=params["gamma"],
        shrinking=params["shrinking"]
    )

    logger.info("Training SVR model")
    model.fit(X_reshaped, y.ravel())
    logger.info("Model trained")

    if len(col_for_train) == 0:
        x_input = create_x_input(
            df_train[[col_target]].astype(np.float64),
            lag
        ).astype(np.float64)
        n_features = 1
    else:
        x_input = create_x_input(
            df_train[use_features].astype(np.float64),
            lag
        ).astype(np.float64)
        n_features = len(use_features)

    logger.info(f"x_input shape before reshape: {x_input.shape}")

    if not np.isfinite(x_input).all():
        logger.error("Non-finite values in x_input")
        raise ValueError("x_input contains inf or NaN")

    x_input = x_input.reshape((1, lag, n_features))
    logger.info(f"x_input reshaped: {x_input.shape}")

    count_pred_points = len(df_test.values)
    logger.info(f"Prediction steps: {count_pred_points}")

    if not np.isfinite(df_test.values).all():
        logger.error("Non-finite values in df_test")
        raise ValueError("df_test contains inf or NaN")

    predict_values = make_predictions_np_input(
        x_input=x_input,
        x_future=df_test.values,
        n_features=n_features,
        model=model,
        lag=lag,
        count_pred_points=count_pred_points
    )

    predict_values = np.array(predict_values).flatten()

    logger.info(f"Prediction done: shape={predict_values.shape}")

    if not np.isfinite(predict_values).all():
        logger.error("Non-finite values in predictions")
        raise ValueError("Predictions contain inf or NaN")

    df_test_pred[col_target] = predict_values

    logger.info("FINISH SVR_forecast")

    return df_test_pred