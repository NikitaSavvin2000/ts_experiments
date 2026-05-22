import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA

DEFAULT_ARIMA_PARAMS = {
    "p": 1,
    "d": 1,
    "q": 1,
    "trend": "n"
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

        common_time_keywords = [
            "date",
            "time",
            "datetime",
            "timestamp",
            "ds"
        ]

        guessed_col = [
            col for col in df_train.columns
            if any(key in col.lower() for key in common_time_keywords)
        ]

        if time_column is None:
            time_column = guessed_col[0] if guessed_col else None

        logger.info(f"Time column: {time_column}")

        df_train[time_column] = pd.to_datetime(
            df_train[time_column],
            errors="coerce"
        )

        df_train = df_train.dropna(subset=[time_column])

        df_train = df_train.sort_values(by=time_column)

        freq = pd.infer_freq(df_train[time_column]) or "D"

        logger.info(f"Inferred frequency: {freq}")

        df_train[col_target] = (
            df_train[col_target]
            .replace("None", None)
            .astype(float)
        )

        if df_train[col_target].isna().any():
            logger.error("NaN values found in target series")
            raise ValueError("NaN in target")

        df_train = df_train.set_index(time_column)

        df_train = df_train[
            ~df_train.index.duplicated(keep="last")
        ]

        df_train = df_train.asfreq(freq)

        df_train[col_target] = df_train[col_target].ffill()

        series = df_train[col_target].dropna()

        order = (
            params["p"],
            params["d"],
            params["q"]
        )

        logger.info(f"Selected ARIMA order: {order}")

        model = ARIMA(
            series,
            order=order,
            trend=params["trend"],
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted = model.fit()

        logger.info(f"ARIMA fitted. AIC: {fitted.aic}")

        forecast_values = fitted.forecast(
            steps=len(df_test)
        )

        forecast_values = np.array(forecast_values)

        df_test_pred = df_test[
            [time_column, col_target]
        ].copy()

        df_test_pred[col_target] = forecast_values

        return df_test_pred

    except Exception as e:
        logger.error(f"ARIMA failed: {str(e)}")
        raise