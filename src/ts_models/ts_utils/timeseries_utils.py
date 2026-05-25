import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd

import os
import matplotlib.pyplot as plt
import pandas as pd


def split_sequence(sequence, n_steps):
    """
    Split a univariate sequence into samples for supervised learning.

    Parameters:
        sequence (np.ndarray): Input sequence.
        n_steps (int): Number of steps to look back.

    Returns:
        tuple: Arrays of input samples (X) and targets (y).
    """
    X, y = [], []


    for i in range(len(sequence) - n_steps):
        seq_x, seq_y = sequence[i:i + n_steps, :], sequence[i + n_steps, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def create_x_input(df_train, n_steps):
    """
    Create the input array for predictions from the training DataFrame.

    Parameters:
        df_train (pd.DataFrame): Training data.
        n_steps (int): Number of steps to look back.

    Returns:
        np.ndarray: Input array for predictions.
    """

    return df_train.iloc[-n_steps:].values



def make_predictions(x_input, x_future, n_features, model, lag, count_pred_points):
    """
    Generate predictions for a future horizon using an iterative approach.

    Parameters:
        x_input (np.ndarray): Initial input data.
        x_future (np.ndarray): Future data.
        n_features (int): Number of features in the data.
        model (tf.keras.Model): Trained prediction model.
        lag (int): Number of time steps used for predictions.

    Returns:
        list: Predicted values.
    """

    predict_values = []
    for _ in range(count_pred_points):
        x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, -1)), dtype=tf.float32)
        y_predict = model.predict(x_input_tensor)
        predict_values.append(y_predict)

        x_input = np.delete(x_input, 0, axis=1)
        future_lag = x_future[0]
        x_future = np.delete(x_future, 0, axis=0)
        future_lag[0] = y_predict
        x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)
        x_input = x_input.reshape((1, lag, n_features))

    return predict_values



def make_predictions_lstm(x_input, x_future, points_per_call, model):
    predict_values = []
    x_future_len = len(x_future)
    remaining_horizon = x_future_len

    while remaining_horizon > 0:
        current_points_to_predict = min(remaining_horizon, points_per_call)
        x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, x_input.shape[1], x_input.shape[2])), dtype=tf.float32)
        y_predict = model.predict(x_input_tensor, verbose=0)

        if len(y_predict.shape) == 2 and y_predict.shape[0] == 1:
            y_predict = y_predict[0]

        y_predict = y_predict[:current_points_to_predict]
        predict_values.extend(y_predict)

        for i in range(current_points_to_predict):
            cur_val = y_predict[i]
            x_input = np.delete(x_input, (0), axis=1)
            future_lag = x_future[0]
            x_future = np.delete(x_future, 0, axis=0)
            future_lag[0] = cur_val
            x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)

        remaining_horizon -= current_points_to_predict

    return predict_values


def make_predictions_np_input(x_input, x_future, n_features, model, lag, count_pred_points):
    """
    Generate predictions for a future horizon using an iterative approach.

    Parameters:_
        x_input (np.ndarray): Initial input data.
        x_future (np.ndarray): Future data.
        n_features (int): Number of features in the data.
        model (tf.keras.Model): Trained prediction model.
        lag (int): Number of time steps used for predictions.

    Returns:
        list: Predicted values.
    """
    predict_values = []
    for _ in range(count_pred_points):
        x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, -1)), dtype=tf.float32)
        if hasattr(x_input_tensor, "numpy"):
            x_input_tensor = x_input_tensor.numpy()
        x_input_tensor = np.asarray(x_input_tensor)
        y_predict = model.predict(x_input_tensor)
        predict_values.append(y_predict)

        x_input = np.delete(x_input, 0, axis=1)
        future_lag = x_future[0]
        x_future = np.delete(x_future, 0, axis=0)
        future_lag[0] = y_predict
        x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)
        x_input = x_input.reshape((1, lag, n_features))

    return predict_values


def regression_metrics(true, pred):
    true = np.array(true)
    pred = np.array(pred)

    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    mape = np.mean(np.abs((true - pred) / np.clip(np.abs(true), 1e-8, None))) * 100

    round_int = 3

    return {
        "r2": float(round(r2, round_int)),
        "mae": float(round(mae, round_int)),
        "mape": float(round(mape, round_int)),
        "rmse": float(round(rmse, round_int))
    }


def calculate_discreteness_interval(df: pd.DataFrame, time_column: str) -> int:
    """
    Вычисляет средний временной интервал в минутах между записями.

    :param df: DataFrame с временными метками
    :param time_column: Название колонки с временными метками
    :return: Средний временной интервал в минутах
    """
    df[time_column] = pd.to_datetime(df[time_column])
    time_interval = df[time_column].diff().dt.total_seconds().mean()
    return round(time_interval)


def calculate_test_points_predict(start_test_date: str, end_test_date: str, discreteness: int) -> int:
    start = pd.to_datetime(start_test_date)
    end = pd.to_datetime(end_test_date)

    total_seconds = (end - start).total_seconds()

    return int(total_seconds // discreteness) + 1


def generate_time_series_df(start_date: str, n_rows: int, freq_seconds: int, col_time: str, col_target: str):
    start = pd.to_datetime(start_date)

    times = pd.date_range(
        start=start + pd.Timedelta(seconds=freq_seconds),
        periods=n_rows,
        freq=f"{freq_seconds}s"
    )

    df = pd.DataFrame({
        col_time: times.strftime("%Y-%m-%d %H:%M:%S"),
        col_target: [None] * n_rows
    })

    return df



def test_method_visualize(df_eval, df_test_pred, df_real_pred, col_time, col_target, metrix_dict):
    df_eval = df_eval.copy()
    df_test_pred = df_test_pred.copy()
    df_real_pred = df_real_pred.copy()

    df_eval[col_time] = pd.to_datetime(df_eval[col_time])
    df_test_pred[col_time] = pd.to_datetime(df_test_pred[col_time])
    df_real_pred[col_time] = pd.to_datetime(df_real_pred[col_time])

    df_eval = df_eval.sort_values(col_time)
    df_test_pred = df_test_pred.sort_values(col_time)
    df_real_pred = df_real_pred.sort_values(col_time)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_eval[col_time],
        y=df_eval[col_target],
        mode="lines",
        name="real",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=df_test_pred[col_time],
        y=df_test_pred[col_target],
        mode="lines",
        name="test_pred",
        line=dict(color="orange")
    ))

    fig.add_trace(go.Scatter(
        x=df_real_pred[col_time],
        y=df_real_pred[col_target],
        mode="lines",
        name="real_pred",
        line=dict(color="red")
    ))

    metrics_text = " | ".join([f"{k}: {v}" for k, v in metrix_dict.items()])

    fig.update_layout(
        title=metrics_text,
        xaxis_title=col_time,
        yaxis_title=col_target
    )

    fig.show()

def test_method_visualize_m(
        df_eval,
        df_test_pred,
        df_real_pred,
        col_time,
        col_target,
        metrix_dict
):

    print("[VIS] start")

    df_eval = df_eval.copy()
    df_test_pred = df_test_pred.copy()
    df_real_pred = df_real_pred.copy()

    df_eval[col_time] = pd.to_datetime(df_eval[col_time])
    df_test_pred[col_time] = pd.to_datetime(df_test_pred[col_time])
    df_real_pred[col_time] = pd.to_datetime(df_real_pred[col_time])

    df_eval = df_eval.sort_values(col_time)
    df_test_pred = df_test_pred.sort_values(col_time)
    df_real_pred = df_real_pred.sort_values(col_time)

    print(f"[VIS] eval points: {len(df_eval)}")
    print(f"[VIS] test_pred points: {len(df_test_pred)}")
    print(f"[VIS] real_pred points: {len(df_real_pred)}")

    plt.figure(figsize=(14, 6))

    plt.plot(
        df_eval[col_time],
        df_eval[col_target],
        label="real",
        linewidth=2
    )

    plt.plot(
        df_test_pred[col_time],
        df_test_pred[col_target],
        label="test_pred",
        linewidth=2
    )

    plt.plot(
        df_real_pred[col_time],
        df_real_pred[col_target],
        label="real_pred",
        linewidth=2
    )

    title = " | ".join([f"{k}: {v}" for k, v in metrix_dict.items()])

    plt.title(title)
    plt.xlabel(col_time)
    plt.ylabel(col_target)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("IMAGE_ARTICLE", exist_ok=True)

    save_path = os.path.join("IMAGE_ARTICLE", "forecast_plot_Transformer.png")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")

    print(f"[VIS] saved: {save_path}")

    plt.show()

    print("[VIS] done")