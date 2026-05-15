import pandas as pd
from prophet import Prophet


def Prophet_forecast(
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

    df_test_pred = df_test[[time_column, col_target]].copy()

    df_train[col_target] = df_train[col_target].replace("None", None).astype(float)

    cols = [col_target] + list(col_for_train)
    df_train = df_train[[time_column] + cols].copy()
    df_test = df_test[[time_column] + cols].copy()

    nan_locations = df_train.isna()
    if nan_locations.any().any():
        logger.error("NaN values found in df_train:")
        nan_rows = df_train[nan_locations.any(axis=1)]
        logger.error(f"Rows with NaN:\n{nan_rows}")
        raise ValueError("NaN values detected in training data")

    df_prophet = df_train.rename(columns={time_column: "ds", col_target: "y"})

    model = Prophet()

    for col in col_for_train:
        model.add_regressor(col)

    model.fit(df_prophet)

    future = df_test.rename(columns={time_column: "ds"})

    forecast = model.predict(future)

    df_test_pred[col_target] = forecast["yhat"].values

    return df_test_pred