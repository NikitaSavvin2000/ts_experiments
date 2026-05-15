import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import regularizers

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

model_architecture_params = {
    "lstm0_units": 64,
    "lstm1_units": 64,
    "lstm2_units": 32,
    "activation": "swish",
    "recurrent_dropout_rate": 0.0,
    "regularizers_l2": 0.001,
    "optimizer": "adam"
}

epochs = 10
points_per_call = 1


def LSTM3_forecast(
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

    col_for_train = [col_target] + list(col_for_train)

    df_train = df_train[col_for_train].copy()
    df_test_pred = df_test[[time_column, col_target]].copy()
    df_test = df_test[col_for_train].copy()

    df_train[col_target] = df_train[col_target].replace("None", None).astype(float)

    nan_locations = df_train.isna()

    if nan_locations.any().any():
        logger.error("NaN values found in df_train")
        nan_rows = df_train[nan_locations.any(axis=1)]
        logger.error(nan_rows)
        raise ValueError("NaN values detected in training data")

    values = df_train[col_for_train].astype(np.float32).values

    x_input = create_x_input(
        df_train[col_for_train].astype(np.float32),
        lag
    ).astype(np.float32)

    X, y = split_sequence(values, lag)

    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    n_features = values.shape[1]
    X = X.reshape((X.shape[0], lag, n_features))

    model = Sequential()

    model.add(
        LSTM(
            model_architecture_params["lstm0_units"],
            activation=model_architecture_params["activation"],
            return_sequences=True,
            recurrent_dropout=model_architecture_params["recurrent_dropout_rate"],
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)
        )
    )

    model.add(
        LSTM(
            model_architecture_params["lstm1_units"],
            activation=model_architecture_params["activation"],
            return_sequences=True,
            recurrent_dropout=model_architecture_params["recurrent_dropout_rate"],
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)
        )
    )

    model.add(
        LSTM(
            model_architecture_params["lstm2_units"],
            activation=model_architecture_params["activation"],
            return_sequences=False,
            recurrent_dropout=model_architecture_params["recurrent_dropout_rate"],
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)
        )
    )

    model.add(
        Dense(
            points_per_call,
            activation="linear",
            kernel_regularizer=regularizers.l2(
                model_architecture_params["regularizers_l2"]
            ),
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)
        )
    )

    model.compile(
        optimizer=model_architecture_params["optimizer"],
        loss="mean_squared_error",
        metrics=["mae"]
    )

    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=32,
        shuffle=False,
        verbose=1
    )

    x_input = x_input.reshape((1, lag, n_features))

    predict_values = make_predictions_lstm(
        x_input=x_input,
        x_future=df_test[col_for_train].values,
        model=model,
        points_per_call=points_per_call
    )

    predict_values = np.array(predict_values).flatten()

    df_test_pred[col_target] = predict_values

    return df_test_pred