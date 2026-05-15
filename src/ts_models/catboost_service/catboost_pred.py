import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.ts_models.ts_utils.timeseries_utils import (
    split_sequence,
    create_x_input,
    make_predictions_np_input
)

model_architecture_params = {
    "loss_function": "RMSE"
}

def CatBoost_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger
):
    df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
    df_train = df_train.sort_values(by=time_column).reset_index(drop=True)

    col_for_train = [col_target] + col_for_train
    df_train = df_train[col_for_train].copy()
    df_test_pred = df_test[[time_column, col_target]].copy()
    df_test = df_test[col_for_train].copy()

    df_train[col_target] = df_train[col_target].replace("None", None).astype(float)

    nan_locations = df_train.isna()
    if nan_locations.any().any():
        logger.error("NaN values found in df_train:")
        nan_rows = df_train[nan_locations.any(axis=1)]
        logger.error(f"Rows with NaN:\n{nan_rows}")
        nan_columns = nan_locations.sum()
        logger.error(f"NaN counts per column:\n{nan_columns[nan_columns > 0]}")
        raise ValueError("NaN values detected in training data")

    values = df_train[col_for_train].values
    x_input = create_x_input(df_train, lag)
    X, y = split_sequence(values, lag)
    n_features = values.shape[1]

    model = CatBoostRegressor(**model_architecture_params, verbose=0)
    X_reshaped = X.reshape(X.shape[0], -1)
    model.fit(X_reshaped, y)

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