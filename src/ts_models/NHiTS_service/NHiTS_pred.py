import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from darts import TimeSeries
from darts.models import NHiTSModel

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

torch.set_default_dtype(torch.float32)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

DEFAULT_NHITS_PARAMS = {
    "num_stacks": 3,
    "num_blocks": 1,
    "num_layers": 2,
    "layer_widths": 512,
    "dropout": 0.1,
    "activation": "ReLU",
    "batch_size": 32,
    "n_epochs": 10
}

points_per_call = 1


def _get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "gpu"
        return "cpu"
    except:
        return "cpu"


def _to_float32(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.astype({c: np.float32 for c in cols})


def NHiTS_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger,
        params=None
):
    params = params or DEFAULT_NHITS_PARAMS

    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
    df_train = df_train.sort_values(by=time_column).reset_index(drop=True)

    if col_for_train is None or len(col_for_train) == 0:
        col_for_train = []

    use_features = [col_target] + list(col_for_train)

    df_train = df_train[use_features + [time_column]]
    df_test_pred = df_test[[time_column, col_target]].copy()

    if len(col_for_train) > 0:
        df_test = df_test[use_features + [time_column]]
    else:
        df_test = df_test[[col_target, time_column]]

    df_train = _to_float32(df_train, use_features)
    df_test = _to_float32(df_test, use_features)

    df_train[col_target] = df_train[col_target].replace("None", np.nan)

    if df_train.isna().any().any():
        logger.error(df_train[df_train.isna().any(axis=1)])
        raise ValueError("NaN in train")

    cov_cols = [c for c in use_features if c != col_target]

    train_ts = TimeSeries.from_dataframe(
        df_train,
        time_col=time_column,
        value_cols=col_target
    ).astype(np.float32)

    if len(cov_cols) > 0:
        cov_train_ts = TimeSeries.from_dataframe(
            df_train,
            time_col=time_column,
            value_cols=cov_cols
        ).astype(np.float32)

        cov_full_df = pd.concat([df_train, df_test], axis=0)
        cov_full_df = cov_full_df.sort_values(by=time_column).reset_index(drop=True)

        cov_full_ts = TimeSeries.from_dataframe(
            cov_full_df,
            time_col=time_column,
            value_cols=cov_cols
        ).astype(np.float32)
    else:
        cov_train_ts = None
        cov_full_ts = None

    pl_trainer_kwargs = {
        "accelerator": _get_device(),
        "devices": 1
    }

    model = NHiTSModel(
        input_chunk_length=lag,
        output_chunk_length=points_per_call,
        num_stacks=params["num_stacks"],
        num_blocks=params["num_blocks"],
        num_layers=params["num_layers"],
        layer_widths=params["layer_widths"],
        dropout=params["dropout"],
        activation=params["activation"],
        n_epochs=params["n_epochs"],
        batch_size=params["batch_size"],
        random_state=SEED,
        pl_trainer_kwargs=pl_trainer_kwargs
    )

    if cov_train_ts is not None:
        model.fit(
            series=train_ts,
            past_covariates=cov_train_ts,
            verbose=True
        )

        pred = model.predict(
            n=len(df_test),
            series=train_ts,
            past_covariates=cov_full_ts
        )
    else:
        model.fit(
            series=train_ts,
            verbose=True
        )

        pred = model.predict(
            n=len(df_test),
            series=train_ts
        )

    df_test_pred[col_target] = pred.values().astype(np.float32).flatten()

    return df_test_pred