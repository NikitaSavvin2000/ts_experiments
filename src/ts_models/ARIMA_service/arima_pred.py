import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA

DEFAULT_ARIMA_PARAMS = {
    "p": 4,
    "d": 1,
    "q": 8
}

DEFAULT_ARIMA_PARAMS = {
    "p": 8,
    "d": 1,
    "q": 16
}

def ARIMA_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger,
        params=None
):
    params = params or DEFAULT_ARIMA_PARAMS

    try:
        df_train = df_train.copy()
        df_test = df_test.copy()

        logger.info(f"time_column={time_column}")

        df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
        df_train = df_train.dropna(subset=[time_column])
        df_train = df_train.sort_values(by=time_column)

        freq = pd.infer_freq(df_train[time_column]) or "D"

        print(freq)
        logger.info(f"freq={freq}")

        df_train[col_target] = df_train[col_target].replace("None", np.nan).astype(float)

        if df_train[col_target].isna().any():
            df_train[col_target] = df_train[col_target].ffill()

        df_train = df_train.set_index(time_column)
        df_train = df_train[~df_train.index.duplicated(keep="last")]

        df_train = df_train.asfreq(freq)

        series = df_train[col_target].dropna()

        order = (params["p"], params["d"], params["q"])

        logger.info(f"order={order}")

        trend = "n"
        if params["d"] == 0:
            trend = "c"

        model = ARIMA(
            series,
            order=order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted = model.fit()

        logger.info("model fitted")

        forecast = fitted.forecast(steps=len(df_test))
        forecast = np.asarray(forecast)

        df_test_pred = df_test.copy()
        df_test_pred[col_target] = forecast

        return df_test_pred

    except Exception as e:
        logger.error(f"ARIMA failed: {str(e)}")
        raise