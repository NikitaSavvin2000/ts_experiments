"""
pdm run src/ts_models/PatchTST_service/runners.py
"""

import os
import random
import numpy as np
import pandas as pd
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def PatchTST_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger
):
    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
    df_test[time_column] = pd.to_datetime(df_test[time_column], errors="coerce")

    df_train = df_train.sort_values(by=time_column).reset_index(drop=True)
    df_test = df_test.sort_values(by=time_column).reset_index(drop=True)

    features = [col_target] + list(col_for_train)

    df_train = df_train[features + [time_column]].copy()
    df_test_pred = df_test[[time_column, col_target]].copy()

    df_train[col_target] = pd.to_numeric(df_train[col_target], errors="coerce")

    if df_train.isna().any().any():
        nan_rows = df_train[df_train.isna().any(axis=1)]
        logger.error(nan_rows)
        raise ValueError("NaN values detected in training data")

    df_nf = pd.DataFrame({
        "unique_id": "series",
        "ds": df_train[time_column],
        "y": df_train[col_target].values
    })

    freq = pd.infer_freq(df_train[time_column])
    if freq is None:
        freq = "D"

    h = len(df_test)

    model = PatchTST(
        h=h,
        input_size=lag,
        max_steps=50,
        batch_size=32,
        random_seed=SEED
    )

    nf = NeuralForecast(
        models=[model],
        freq=freq
    )

    nf.fit(df=df_nf)

    forecast = nf.predict()

    preds = forecast.iloc[:, -1].values

    df_test_pred[col_target] = preds[:len(df_test)]

    return df_test_pred