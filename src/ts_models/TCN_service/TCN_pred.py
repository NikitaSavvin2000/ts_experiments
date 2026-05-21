import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from tcn import TCN

from src.ts_models.ts_utils.timeseries_utils import (
    split_sequence,
    create_x_input,
    make_predictions_lstm
)

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

model_cfg_light = {
    "nb_filters": 4,
    "kernel_size": 3,
    "nb_stacks": 1,
    "dilations": [1, 2],
    "use_layer_norm": False,
    "dropout_rate": 0.0,
    "epochs": 1
}

model_cfg_prod = {
    "nb_filters": 64,
    "kernel_size": 3,
    "nb_stacks": 1,
    "dilations": [1, 2, 4, 8],
    "use_layer_norm": True,
    "dropout_rate": 0.02,
    "epochs": 10
}

points_per_call = 1


def _log_epoch(epoch, logs):
    print(f"epoch={epoch} loss={logs.get('loss')} mae={logs.get('mae')}", flush=True)

import time
from tensorflow.keras.callbacks import Callback

class EpochProgress(Callback):
    def on_train_begin(self, logs=None):
        print("[TRAIN] start", flush=True)

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()
        print(f"[EPOCH {epoch}] start", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        dt = time.time() - self.t0
        loss = logs.get("loss")
        mae = logs.get("mae")
        print(f"[EPOCH {epoch}] end | loss={loss:.5f} mae={mae:.5f} time={dt:.2f}s", flush=True)


def TCN_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger,
        light=True
):
    print("START PIPELINE", flush=True)

    cfg = model_cfg_light if light else model_cfg_prod

    print("STEP 1: preprocessing", flush=True)

    df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
    df_train = df_train.sort_values(by=time_column).reset_index(drop=True)

    col_for_train = [col_target] + list(col_for_train)

    df_train = df_train[col_for_train].copy()
    df_test_pred = df_test[[time_column, col_target]].copy()
    df_test = df_test[col_for_train].copy()

    df_train[col_target] = df_train[col_target].replace("None", None).astype(float)

    if df_train.isna().any().any():
        logger.error("NaN detected")
        raise ValueError("NaN in train")

    values = df_train[col_for_train].astype(np.float32).values

    print("STEP 2: building sequences", flush=True)

    x_input = create_x_input(
        df_train[col_for_train].astype(np.float32),
        lag
    ).astype(np.float32)

    X, y = split_sequence(values, lag)

    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    n_features = values.shape[1]
    X = X.reshape((X.shape[0], lag, n_features))

    print(f"DATA READY: X={X.shape}, y={y.shape}", flush=True)

    print("STEP 3: build model", flush=True)

    model = Sequential()
    model.add(Input(shape=(lag, n_features)))

    model.add(
        TCN(
            nb_filters=cfg["nb_filters"],
            kernel_size=cfg["kernel_size"],
            nb_stacks=cfg["nb_stacks"],
            dilations=cfg["dilations"],
            use_layer_norm=cfg["use_layer_norm"],
            dropout_rate=cfg["dropout_rate"],
            kernel_initializer="glorot_uniform"
        )
    )

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    callbacks = [
        LambdaCallback(on_epoch_end=_log_epoch),
        EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    ]

    print("STEP 4: training", flush=True)

    model.fit(
        X,
        y,
        epochs=cfg["epochs"],
        batch_size=32,
        shuffle=False,
        verbose=0,
        callbacks=[
            EpochProgress(),
            EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
        ]
    )

    print("STEP 5: prediction init", flush=True)

    x_input = x_input.reshape((1, lag, n_features))

    predict_values = make_predictions_lstm(
        x_input=x_input,
        x_future=df_test[col_for_train].values,
        model=model,
        points_per_call=points_per_call
    )

    predict_values = np.array(predict_values).flatten()

    df_test_pred[col_target] = predict_values

    print("PIPELINE DONE", flush=True)

    return df_test_pred