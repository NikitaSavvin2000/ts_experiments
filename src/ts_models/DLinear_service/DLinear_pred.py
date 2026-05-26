# import os
# import random
# import numpy as np
# import pandas as pd
# import torch
#
# from darts import TimeSeries
# from darts.models import DLinearModel
#
# SEED = 42
#
# os.environ["PYTHONHASHSEED"] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
#
#
# dlinear_params_easy = {
#     "output_chunk_length": 1,
#     "n_epochs": 10,
#     "batch_size": 16,
#     "optimizer_lr": 1e-3,
#     "hidden_size": 32,
#     "kernel_size": 3
# }
#
#
# def _get_pl_kwargs():
#     if torch.cuda.is_available():
#         return {"accelerator": "gpu", "devices": -1}
#     if torch.backends.mps.is_available():
#         return {"accelerator": "mps", "devices": 1}
#     return {"accelerator": "cpu", "devices": 1}
#
#
# def _infer_freq(df, time_column):
#     df = df.sort_values(time_column)
#     freq = pd.infer_freq(df[time_column])
#     if freq is not None:
#         return freq
#     diffs = df[time_column].diff().dropna()
#     return diffs.mode().iloc[0]
#
#
# def DLinear_forecast(
#         col_target,
#         time_column,
#         df_train,
#         df_test,
#         lag,
#         col_for_train,
#         logger
# ):
#
#     params = dlinear_params_easy
#
#     df_train = df_train.copy()
#     df_test = df_test.copy()
#
#     df_train[time_column] = pd.to_datetime(df_train[time_column])
#     df_test[time_column] = pd.to_datetime(df_test[time_column])
#
#     df_train = df_train.sort_values(time_column).reset_index(drop=True)
#     df_test = df_test.sort_values(time_column).reset_index(drop=True)
#
#     cols = [col_target] + list(col_for_train)
#     exog_cols = list(col_for_train)
#
#     df_train = df_train[cols + [time_column]]
#     df_test = df_test[cols + [time_column]]
#
#     df_train[col_target] = pd.to_numeric(df_train[col_target], errors="coerce")
#
#     freq = _infer_freq(df_train, time_column)
#
#     target_series = TimeSeries.from_dataframe(
#         df_train,
#         time_col=time_column,
#         value_cols=col_target,
#         fill_missing_dates=True,
#         freq=freq
#     ).astype(np.float32)
#
#     past_cov = TimeSeries.from_dataframe(
#         df_train,
#         time_col=time_column,
#         value_cols=exog_cols,
#         fill_missing_dates=True,
#         freq=freq
#     ).astype(np.float32)
#
#     model = DLinearModel(
#         input_chunk_length=lag,
#         output_chunk_length=params["output_chunk_length"],
#         n_epochs=params["n_epochs"],
#         batch_size=params["batch_size"],
#         random_state=SEED,
#         optimizer_kwargs={"lr": params["optimizer_lr"]},
#         pl_trainer_kwargs=_get_pl_kwargs()
#     )
#
#     model.fit(target_series, past_covariates=past_cov)
#
#     preds = []
#
#     history_target = target_series
#     history_cov = past_cov
#
#     last_time = target_series.time_index[-1]
#     step = pd.tseries.frequencies.to_offset(freq)
#
#     for i in range(len(df_test)):
#
#         pred = model.predict(
#             n=1,
#             series=history_target,
#             past_covariates=history_cov
#         )
#
#         val = float(pred.values().ravel()[0])
#         preds.append(val)
#
#         next_time = last_time + step
#         last_time = next_time
#
#         new_target = pd.DataFrame({
#             time_column: [next_time],
#             col_target: [val]
#         })
#
#         new_target_ts = TimeSeries.from_dataframe(
#             new_target,
#             time_col=time_column,
#             value_cols=col_target,
#             fill_missing_dates=True,
#             freq=freq
#         ).astype(np.float32)
#
#         history_target = history_target.append(new_target_ts)
#
#         if past_cov is not None:
#             new_cov = pd.DataFrame({
#                 time_column: [next_time],
#                 **{c: [df_test[c].iloc[i]] for c in exog_cols}
#             })
#
#             new_cov_ts = TimeSeries.from_dataframe(
#                 new_cov,
#                 time_col=time_column,
#                 value_cols=exog_cols,
#                 fill_missing_dates=True,
#                 freq=freq
#             ).astype(np.float32)
#
#             history_cov = history_cov.append(new_cov_ts)
#
#     return pd.DataFrame({
#         time_column: df_test[time_column].values,
#         col_target: preds
#     })


import os
import random
import numpy as np
import pandas as pd
import torch

from darts import TimeSeries
from darts.models import DLinearModel

SEED = 42

def infer_and_reindex(df, time_col):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).drop_duplicates(time_col)

    freq = pd.infer_freq(df[time_col])

    if freq is None:
        diffs = df[time_col].diff().dropna()
        freq = diffs.mode().iloc[0]

    df = df.set_index(time_col)

    if isinstance(freq, pd.Timedelta):
        df = df.asfreq(freq)
    else:
        df = df.asfreq(freq)

    return df, freq

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


DEFAULT_DLINEAR_PARAMS = {
    "output_chunk_length": 1,
    "n_epochs": 10,
    "batch_size": 16,
    "optimizer_lr": 1e-3,
    "hidden_size": 32,
    "kernel_size": 3
}


def _get_pl_kwargs():
    if torch.cuda.is_available():
        return {"accelerator": "gpu", "devices": -1}
    if torch.backends.mps.is_available():
        return {"accelerator": "mps", "devices": 1}
    return {"accelerator": "cpu", "devices": 1}


def _infer_freq(df, time_column):
    df = df.sort_values(time_column)
    freq = pd.infer_freq(df[time_column])
    if freq is not None:
        return freq
    diffs = df[time_column].diff().dropna()
    return diffs.mode().iloc[0]


def DLinear_forecast(
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
        df_train = df_train.drop_duplicates(subset=[time_column])
        df_test = df_test.drop_duplicates(subset=[time_column])

        freq = pd.infer_freq(df_train[time_column])

        if freq is None:
            diffs = df_train[time_column].diff().dropna()
            freq = diffs.mode().iloc[0]

        df_train = df_train.set_index(time_column).asfreq(freq).reset_index()
        df_test = df_test.set_index(time_column).asfreq(freq).reset_index()

        cfg = params or DEFAULT_DLINEAR_PARAMS

        df_train = df_train.copy()
        df_test = df_test.copy()

        df_train[time_column] = pd.to_datetime(df_train[time_column])
        df_test[time_column] = pd.to_datetime(df_test[time_column])

        df_train = df_train.sort_values(time_column).reset_index(drop=True)
        df_test = df_test.sort_values(time_column).reset_index(drop=True)

        cols = [col_target] + list(col_for_train)
        exog_cols = list(col_for_train)

        df_train = df_train[cols + [time_column]]
        df_test = df_test[cols + [time_column]]

        df_train[col_target] = pd.to_numeric(df_train[col_target], errors="coerce")

        freq = _infer_freq(df_train, time_column)

        try:
            target_series = TimeSeries.from_dataframe(
                df_train,
                time_col=time_column,
                value_cols=col_target,
                fill_missing_dates=False
            )
        except Exception as e:
            print(e)

        past_cov = TimeSeries.from_dataframe(
            df_train,
            time_col=time_column,
            value_cols=exog_cols,
            fill_missing_dates=False,
        ).astype(np.float32)

        model = DLinearModel(
            input_chunk_length=lag,
            output_chunk_length=cfg["output_chunk_length"],
            n_epochs=cfg["n_epochs"],
            batch_size=cfg["batch_size"],
            random_state=SEED,
            optimizer_kwargs={"lr": cfg["optimizer_lr"]},
            pl_trainer_kwargs=_get_pl_kwargs()
        )

        model.fit(target_series, past_covariates=past_cov)

        preds = []

        history_target = target_series
        history_cov = past_cov

        last_time = target_series.time_index[-1]
        step = pd.tseries.frequencies.to_offset(freq)

        for i in range(len(df_test)):

            pred = model.predict(
                n=1,
                series=history_target,
                past_covariates=history_cov
            )

            val = float(pred.values().ravel()[0])
            preds.append(val)

            next_time = last_time + step
            last_time = next_time

            new_target = pd.DataFrame({
                time_column: [next_time],
                col_target: [val]
            })

            new_target_ts = TimeSeries.from_dataframe(
                new_target,
                time_col=time_column,
                value_cols=col_target,
                fill_missing_dates=False,
            ).astype(np.float32)

            history_target = history_target.append(new_target_ts)

            if past_cov is not None:
                new_cov = pd.DataFrame({
                    time_column: [next_time],
                    **{c: [df_test[c].iloc[i]] for c in exog_cols}
                })

                new_cov_ts = TimeSeries.from_dataframe(
                    new_cov,
                    time_col=time_column,
                    value_cols=exog_cols,
                    fill_missing_dates=False,
                ).astype(np.float32)

                history_cov = history_cov.append(new_cov_ts)

        return pd.DataFrame({
            time_column: df_test[time_column].values,
            col_target: preds
        })
    except Exception as e:
        print(e)