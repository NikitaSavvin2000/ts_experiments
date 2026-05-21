import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, Model

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

tf.keras.backend.clear_session()

tf.config.optimizer.set_jit(False)

try:
    tf.config.set_visible_devices([], "GPU")
    print("[DEVICE] CPU forced")
except Exception:
    print("[DEVICE] GPU config skipped")

model_architecture_params = {
    "embed_dim": 8,
    "num_heads": 1,
    "ff_dim": 16,
    "dropout": 0.0,
    "dense_units": 16,
    "learning_rate": 0.001,
    "epochs": 3,
    "batch_size": 64
}

points_per_call = 1


class TransformerBlock(layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()

        print("[MODEL] init TransformerBlock")

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, embed_dim // num_heads)
        )

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])

        self.dropout = layers.Dropout(dropout)

    def call(self, x):

        attn = self.att(x, x)
        x = self.norm1(x + attn)

        ffn = self.ffn(x)
        ffn = self.dropout(ffn)

        x = self.norm2(x + ffn)

        return x


def build_transformer_model(lag, n_features):

    print("[MODEL] building started")

    inputs = layers.Input(shape=(lag, n_features))

    x = layers.Dense(model_architecture_params["embed_dim"])(inputs)

    print("[MODEL] embedding done")

    x = TransformerBlock(
        embed_dim=model_architecture_params["embed_dim"],
        num_heads=model_architecture_params["num_heads"],
        ff_dim=model_architecture_params["ff_dim"],
        dropout=model_architecture_params["dropout"]
    )(x)

    print("[MODEL] transformer block done")

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(
        model_architecture_params["dense_units"],
        activation="relu"
    )(x)

    outputs = layers.Dense(points_per_call)(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=model_architecture_params["learning_rate"]
        ),
        loss="mse",
        metrics=["mae"]
    )

    print("[MODEL] compiled")

    return model


def Transformer_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger
):

    print("[PIPELINE] START")

    print("[PIPELINE] step 1 - copy data")

    df_train = df_train.copy()
    df_test = df_test.copy()

    print("[PIPELINE] step 2 - datetime parse")

    df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
    df_train = df_train.sort_values(time_column).reset_index(drop=True)

    print("[PIPELINE] step 3 - feature selection")

    cols = [col_target] + list(col_for_train)
    cols = list(dict.fromkeys(cols))

    df_train = df_train[cols]
    df_test = df_test[cols]

    print("[PIPELINE] step 4 - cleaning")

    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)

    df_train = df_train.fillna(df_train.median(numeric_only=True))
    df_test = df_test.fillna(df_train.median(numeric_only=True))

    print("[PIPELINE] step 5 - numpy conversion")

    values = df_train.astype(np.float32).values

    print("[PIPELINE] step 6 - sequence split")

    X, y = split_sequence(values, lag)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    X = X.reshape(X.shape[0], lag, values.shape[1])
    y = y.reshape(-1, 1)

    print("[PIPELINE] data shape:", X.shape, y.shape)

    print("[PIPELINE] step 7 - model build")

    model = build_transformer_model(lag, values.shape[1])

    model.summary()

    print("[PIPELINE] step 8 - training start")

    history = model.fit(
        X,
        y,
        epochs=model_architecture_params["epochs"],
        batch_size=model_architecture_params["batch_size"],
        shuffle=False,
        verbose=1
    )

    print("[PIPELINE] step 9 - training done")
    print("[PIPELINE] final loss:", history.history["loss"][-1])

    print("[PIPELINE] step 10 - inference prep")

    x_input = create_x_input(df_train.astype(np.float32), lag)
    x_input = x_input.reshape(1, lag, values.shape[1])

    print("[PIPELINE] step 11 - forecasting")

    preds = make_predictions_lstm(
        x_input=x_input,
        x_future=df_test[cols].values.astype(np.float32),
        points_per_call=points_per_call,
        model=model
    )

    preds = np.array(preds).flatten()

    print("[PIPELINE] step 12 - result build")

    result = df_test[[time_column, col_target]].copy()
    result[col_target] = preds

    print("[PIPELINE] DONE")

    return result