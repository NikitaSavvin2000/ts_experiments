import numpy as np
import pandas as pd
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX

model_search_space = {
    "p": range(0, 3),
    "d": range(0, 2),
    "q": range(0, 3)
}

def _find_best_order(y, exog, logger):
    best_aic = np.inf
    best_order = (1, 1, 1)

    for order in itertools.product(
            model_search_space["p"],
            model_search_space["d"],
            model_search_space["q"]
    ):
        try:
            model = SARIMAX(
                y,
                exog=exog,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False)

            if res.aic < best_aic:
                best_aic = res.aic
                best_order = order

        except:
            continue

    logger.info(f"Best ARIMAX order: {best_order}, AIC: {best_aic}")
    return best_order


def ARIMAX_forecast(
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

    y_train = df_train[col_target].astype(float).values
    exog_train = df_train.drop(columns=[col_target]).astype(float).values
    exog_test = df_test.drop(columns=[col_target]).astype(float).values

    order = _find_best_order(y_train, exog_train, logger)

    model = SARIMAX(
        endog=y_train,
        exog=exog_train,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)

    forecast = model_fit.forecast(
        steps=len(df_test),
        exog=exog_test
    )

    df_test_pred[col_target] = np.array(forecast).flatten()

    return df_test_pred