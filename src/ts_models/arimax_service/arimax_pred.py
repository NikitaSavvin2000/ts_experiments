# import numpy as np
# import pandas as pd
# import pmdarima as pm
#
#
# DEFAULT_ARIMA_PARAMS = {
#     "start_p": 1,
#     "start_q": 1,
#     "max_p": 5,
#     "max_q": 5,
#     "d": None,
#     "trend": "n",
# }
#
#
# def ARIMAX_forecast(
#         col_target,
#         time_column,
#         df_train,
#         df_test,
#         lag,
#         params,
#         col_for_train,
#         logger
# ):
#     params = params or DEFAULT_ARIMA_PARAMS
#
#     df_train = df_train.copy()
#     df_test = df_test.copy()
#
#     col_for_train = list(col_for_train or [])
#
#     df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
#     df_test[time_column] = pd.to_datetime(df_test[time_column], errors="coerce")
#
#     df_train = df_train.sort_values(time_column).reset_index(drop=True)
#     df_test = df_test.sort_values(time_column).reset_index(drop=True)
#
#     df_full = pd.concat([df_train, df_test], ignore_index=True)
#
#     if lag and lag > 0:
#         lag_cols = []
#         for i in range(1, lag + 1):
#             col = f"{col_target}_lag_{i}"
#             df_full[col] = df_full[col_target].shift(i)
#             lag_cols.append(col)
#
#         col_for_train = list(dict.fromkeys(col_for_train + lag_cols))
#
#     df_train = df_full.iloc[:len(df_train)].copy()
#     df_test = df_full.iloc[len(df_train):].copy()
#
#     y_train = pd.to_numeric(df_train[col_target], errors="coerce")
#
#     if len(col_for_train) > 0:
#         exog_train = df_train[col_for_train].apply(pd.to_numeric, errors="coerce")
#         exog_test = df_test[col_for_train].apply(pd.to_numeric, errors="coerce")
#
#         valid_mask = ~(y_train.isna() | exog_train.isna().any(axis=1))
#
#         df_train = df_train.loc[valid_mask].reset_index(drop=True)
#         y_train = y_train.loc[valid_mask].reset_index(drop=True)
#         exog_train = exog_train.loc[valid_mask].reset_index(drop=True)
#     else:
#         exog_train = None
#         exog_test = None
#
#     logger.info(f"train size after clean: {len(df_train)}")
#     logger.info(f"exog features: {col_for_train}")
#
#     model = pm.auto_arima(
#         y_train,
#         exogenous=exog_train,
#         start_p=params["start_p"],
#         start_q=params["start_q"],
#         max_p=params["max_p"],
#         max_q=params["max_q"],
#         d=params["d"],
#         trend=params["trend"],
#         stepwise=True,
#         suppress_warnings=True,
#         error_action="ignore"
#     )
#
#     forecast = model.predict(
#         n_periods=len(df_test),
#         exogenous=exog_test
#     )
#
#     df_test_pred = df_test[[time_column]].copy()
#     df_test_pred[col_target] = np.array(forecast).flatten()
#
#     return df_test_pred

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


# DEFAULT_ARIMA_PARAMS = {
#     "p": 5,
#     "d": 0,
#     "q": 5,
#     "trend": "c",
# }

DEFAULT_ARIMA_PARAMS = {
    "p": 4,
    "d": 0,
    "q": 8,
    "trend": "c",
}


def ARIMAX_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        params,
        col_for_train,
        logger
):
    params = params or DEFAULT_ARIMA_PARAMS

    df_train = df_train.copy()
    df_test = df_test.copy()

    if col_for_train is None:
        col_for_train = []

    if not isinstance(col_for_train, list):
        col_for_train = list(col_for_train)

    df_train[time_column] = pd.to_datetime(
        df_train[time_column],
        errors="coerce"
    )

    df_test[time_column] = pd.to_datetime(
        df_test[time_column],
        errors="coerce"
    )

    df_train = (
        df_train
        .sort_values(time_column)
        .reset_index(drop=True)
    )

    df_test = (
        df_test
        .sort_values(time_column)
        .reset_index(drop=True)
    )

    use_exog = len(col_for_train) > 0 or (lag is not None and lag > 0)

    if use_exog:

        df_full = pd.concat(
            [df_train, df_test],
            ignore_index=True
        )

        lag_cols = []

        if lag and lag > 0:

            for i in range(1, lag + 1):

                lag_col = f"{col_target}_lag_{i}"

                df_full[lag_col] = (
                    df_full[col_target]
                    .shift(i)
                )

                lag_cols.append(lag_col)

        col_for_train = list(
            dict.fromkeys(
                col_for_train + lag_cols
            )
        )

        train_len = len(df_train)

        df_train = (
            df_full
            .iloc[:train_len]
            .copy()
        )

        df_test = (
            df_full
            .iloc[train_len:]
            .copy()
        )

    y_train = pd.to_numeric(
        df_train[col_target],
        errors="coerce"
    )

    y_train = y_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    exog_train = None
    exog_test = None

    if len(col_for_train) > 0:

        exog_train = (
            df_train[col_for_train]
            .apply(pd.to_numeric, errors="coerce")
        )

        exog_test = (
            df_test[col_for_train]
            .apply(pd.to_numeric, errors="coerce")
        )

        exog_train = exog_train.replace(
            [np.inf, -np.inf],
            np.nan
        )

        exog_test = exog_test.replace(
            [np.inf, -np.inf],
            np.nan
        )

        valid_mask = ~(
                y_train.isna()
                |
                exog_train.isna().any(axis=1)
        )

        y_train = (
            y_train[valid_mask]
            .reset_index(drop=True)
        )

        exog_train = (
            exog_train[valid_mask]
            .reset_index(drop=True)
        )

        exog_test = (
            exog_test
            .fillna(method="ffill")
            .fillna(method="bfill")
            .fillna(0)
        )

        if exog_train.shape[1] == 0:
            exog_train = None
            exog_test = None

    else:

        valid_mask = ~y_train.isna()

        y_train = (
            y_train[valid_mask]
            .reset_index(drop=True)
        )

    p = int(
        params.get(
            "p",
            DEFAULT_ARIMA_PARAMS["p"]
        )
    )

    d = int(
        params.get(
            "d",
            DEFAULT_ARIMA_PARAMS["d"]
        )
    )

    q = int(
        params.get(
            "q",
            DEFAULT_ARIMA_PARAMS["q"]
        )
    )

    trend = params.get(
        "trend",
        DEFAULT_ARIMA_PARAMS["trend"]
    )

    logger.info(
        f"ARIMA order: {(p, d, q)}"
    )

    logger.info(
        f"lag: {lag}"
    )

    logger.info(
        f"use exog: {exog_train is not None}"
    )

    model = ARIMA(
        endog=y_train,
        exog=exog_train,
        order=(p, d, q),
        trend=trend
    )

    model_fit = model.fit()

    forecast = model_fit.forecast(
        steps=len(df_test),
        exog=exog_test
    )

    df_test_pred = (
        df_test[[time_column]]
        .copy()
    )

    df_test_pred[col_target] = (
        np.array(forecast)
        .flatten()
    )

    return df_test_pred