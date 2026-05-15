import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima


def SARIMA_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger,
):
    try:
        df_train = df_train.copy()
        df_test = df_test.copy()

        common_time_keywords = ['date', 'time', 'datetime', 'timestamp', 'ds']
        guessed_col = [
            col for col in df_train.columns
            if any(key in col.lower() for key in common_time_keywords)
        ]

        if time_column is None:
            time_column = guessed_col[0] if guessed_col else None

        logger.info(f"Time column: {time_column}")

        df_train[time_column] = pd.to_datetime(df_train[time_column], errors='coerce')
        df_train = df_train.dropna(subset=[time_column])
        df_train = df_train.sort_values(by=time_column)

        freq = pd.infer_freq(df_train[time_column]) or "D"
        logger.info(f"Inferred frequency: {freq}")

        df_train[col_target] = df_train[col_target].replace("None", None).astype(float)

        if df_train[col_target].isna().any():
            logger.error("NaN detected in target")
            raise ValueError("NaN in target")

        df_train = df_train.set_index(time_column)
        df_train = df_train[~df_train.index.duplicated(keep="last")]
        df_train = df_train.asfreq(freq)
        df_train[col_target] = df_train[col_target].ffill()

        series = df_train[col_target]

        sf_train = series.dropna()

        max_samples = 1000
        y_train = sf_train.tail(max_samples)

        season_length = {
            "D": 7,
            "H": 24,
            "M": 12
        }.get(freq, 7)

        logger.info("Starting auto_arima...")

        stepwise_model = auto_arima(
            y_train,
            seasonal=True,
            m=season_length,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            start_P=0, start_Q=0,
            max_P=3, max_Q=3,
            d=None,
            D=None,
            test='adf',
            seasonal_test='ocsb',
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )

        order = stepwise_model.order
        seasonal_order = stepwise_model.seasonal_order

        logger.info(f"Order: {order}")
        logger.info(f"Seasonal: {seasonal_order}")

        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted = model.fit(disp=False, maxiter=100)

        logger.info(f"AIC: {fitted.aic}")

        forecast_values = fitted.forecast(steps=len(df_test))
        forecast_values = np.array(forecast_values)

        df_test_pred = df_test[[time_column, col_target]].copy()
        df_test_pred[col_target] = forecast_values

        return df_test_pred

    except Exception as e:
        logger.error(f"SARIMA failed: {str(e)}")
        raise