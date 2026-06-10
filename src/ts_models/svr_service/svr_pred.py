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

    df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
    df_train = df_train.sort_values(by=time_column).reset_index(drop=True)

    if col_for_train is None or len(col_for_train) == 0:
        col_for_train = []

    use_features = [col_target] + list(col_for_train)

    df_train[use_features] = df_train[use_features].replace([np.inf, -np.inf], np.nan)
    df_train[use_features] = df_train[use_features].astype(float)
    df_train[use_features] = df_train[use_features].fillna(method="ffill").fillna(method="bfill")

    df_train = df_train[use_features].copy()
    df_test_pred = df_test[[time_column, col_target]].copy()

    if len(col_for_train) > 0:
        df_test = df_test[use_features].copy()
    else:
        df_test = df_test[[col_target]].copy()

    df_train[col_target] = df_train[col_target].replace("None", None).astype(float)

    if df_train.isna().any().any():
        raise ValueError("NaN values detected in training data")

    values = df_train[use_features].astype(np.float32).values

    X, y = split_sequence(values, lag)

    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    n_features = values.shape[1]

    model = SVR(
        kernel=params["kernel"],
        C=params["C"],
        epsilon=params["epsilon"],
        gamma=params["gamma"],
        shrinking=params["shrinking"]
    )

    X_reshaped = X.reshape(X.shape[0], -1)

    y = y.reshape(-1, 1)

    model.fit(X_reshaped, y.ravel())

    if len(col_for_train) == 0:
        x_input = create_x_input(
            df_train[[col_target]].astype(np.float32),
            lag
        ).astype(np.float32)

        n_features = 1
    else:
        x_input = create_x_input(
            df_train[use_features].astype(np.float32),
            lag
        ).astype(np.float32)

    x_input = x_input.reshape((1, lag, n_features))

    count_pred_points = len(df_test.values)

    predict_values = make_predictions_np_input(
        x_input=x_input,
        x_future=df_test.values,
        n_features=n_features,
        model=model,
        lag=lag,
        count_pred_points=count_pred_points
    )

    predict_values = np.array(predict_values).flatten()

    df_test_pred[col_target] = predict_values

    return df_test_pred