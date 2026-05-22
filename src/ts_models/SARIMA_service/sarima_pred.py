import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


DEFAULT_SARIMA_PARAMS = {
    "p": 1,
    "d": 1,
    "q": 1,
    "P": 0,
    "D": 0,
    "Q": 0,
    "m": 12,
    "trend": "n"
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
    try:
        params = params or DEFAULT_SARIMA_PARAMS

        df_train = df_train.copy()
        df_test = df_test.copy()

        common_time_keywords = ['date', 'time', 'datetime', 'timestamp', 'ds']
        guessed_col = [
            col for col in df_train.columns
            if any(key in col.lower() for key in common_time_keywords)
        ]

        if time_column is None:
            time_column = guessed_col[0] if guessed_col else None

        logger.info(f"time_column: {time_column}")

        df_train[time_column] = pd.to_datetime(df_train[time_column], errors='coerce')
        df_train = df_train.dropna(subset=[time_column])
        df_train = df_train.sort_values(by=time_column)

        df_train[col_target] = df_train[col_target].replace("None", None).astype(float)

        if df_train[col_target].isna().any():
            raise ValueError("NaN in target")

        df_train = df_train.set_index(time_column)
        df_train = df_train[~df_train.index.duplicated(keep="last")]

        freq = pd.infer_freq(df_train.index) or "D"

        df_train = df_train.asfreq(freq)
        df_train[col_target] = df_train[col_target].ffill()

        series = df_train[col_target].dropna().tail(1000)

        model = SARIMAX(
            series,
            order=(
                params["p"],
                params["d"],
                params["q"]
            ),
            seasonal_order=(
                params["P"],
                params["D"],
                params["Q"],
                params["m"]
            ),
            trend=params["trend"],
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted = model.fit(
            disp=False,
            maxiter=100
        )

        logger.info(f"AIC: {fitted.aic}")

        forecast = fitted.forecast(steps=len(df_test))

        df_test_pred = df_test[[time_column, col_target]].copy()
        df_test_pred[col_target] = np.array(forecast)

        return df_test_pred

    except Exception as e:
        logger.error(f"SARIMA failed: {str(e)}")
        raise