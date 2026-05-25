import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


DEFAULT_SARIMA_PARAMS = {
    "order": (2, 0, 2),
    "seasonal_order": (1, 1, 1, 4),
    "trend": "c",
}


def SARIMA_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger,
        params=None
):

    params = params or DEFAULT_SARIMA_PARAMS

    df_train = df_train.copy()
    df_test = df_test.copy()

    col_for_train = list(col_for_train or [])

    df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
    df_test[time_column] = pd.to_datetime(df_test[time_column], errors="coerce")

    df_train = df_train.sort_values(time_column).reset_index(drop=True)
    df_test = df_test.sort_values(time_column).reset_index(drop=True)

    df_full = pd.concat([df_train, df_test], ignore_index=True)

    lag_cols = []

    if lag and lag > 0:

        for i in range(1, lag + 1):

            col = f"{col_target}_lag_{i}"

            df_full[col] = df_full[col_target].shift(i)

            lag_cols.append(col)

    col_for_train = list(dict.fromkeys(col_for_train + lag_cols))

    train_len = len(df_train)

    df_train = df_full.iloc[:train_len].copy()
    df_test = df_full.iloc[train_len:].copy()

    df_train = df_train[:-1000]


    y_train = pd.to_numeric(df_train[col_target], errors="coerce")
    y_train = y_train.replace([np.inf, -np.inf], np.nan)

    valid_mask = ~y_train.isna()

    y_train = y_train[valid_mask].reset_index(drop=True)

    exog_train = None
    exog_test = None

    if len(col_for_train) > 0:

        exog_train = (
            df_train.loc[valid_mask, col_for_train]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        exog_test = (
            df_test[col_for_train]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        if exog_train.shape[1] == 0:
            exog_train = None
            exog_test = None

    logger.info(f"train size: {len(y_train)}")
    logger.info(f"exog features: {col_for_train}")

    model = SARIMAX(
        endog=y_train,
        exog=exog_train,
        order=params["order"],
        seasonal_order=params["seasonal_order"],
        trend=params["trend"],
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # model_fit = model.fit(disp=False)
    model_fit = model.fit(method='powell', maxiter=30)

    forecast = model_fit.forecast(
        steps=len(df_test),
        exog=exog_test
    )

    df_test_pred = df_test[[time_column]].copy()
    df_test_pred[col_target] = np.array(forecast).flatten()

    return df_test_pred