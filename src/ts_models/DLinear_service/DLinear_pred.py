import random
import numpy as np
import pandas as pd
import torch

from darts import TimeSeries
from darts.models import DLinearModel


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


DEFAULT_DLINEAR_PARAMS = {
    "output_chunk_length": 1,
    "n_epochs": 10,
    "batch_size": 16,
    "optimizer_lr": 1e-3,
}


def _get_pl_kwargs():
    if torch.cuda.is_available():
        return {"accelerator": "gpu", "devices": -1}
    if torch.backends.mps.is_available():
        return {"accelerator": "mps", "devices": 1}
    return {"accelerator": "cpu", "devices": 1}


def _infer_freq(df, time_col):
    t = df[time_col].sort_values()
    freq = pd.infer_freq(t)

    print("[DEBUG] infer_freq:", freq)

    if freq is not None:
        return freq

    diffs = t.diff().dropna()
    mode = diffs.mode().iloc[0]

    print("[DEBUG] fallback freq:", mode)

    return mode


def _prepare(df, time_col, freq):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).drop_duplicates(time_col)

    print("[DEBUG] PREPARE start rows:", len(df))

    df = df.set_index(time_col)

    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)

    print("[DEBUG] full index size:", len(full_index))

    df = df.reindex(full_index)

    print("[DEBUG] NaN after reindex target:",
          df.iloc[:, 0].isna().sum())

    df.index.name = time_col
    df = df.reset_index()

    return df


def DLinear_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger=None,
        params=None
):
    cfg = params or DEFAULT_DLINEAR_PARAMS

    print("\n========== START DEBUG ==========\n")

    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train[time_column] = pd.to_datetime(df_train[time_column])
    df_test[time_column] = pd.to_datetime(df_test[time_column])

    print("[DEBUG] RAW TRAIN ROWS:", len(df_train))
    print("[DEBUG] RAW TEST ROWS:", len(df_test))

    df_train = df_train.sort_values(time_column).drop_duplicates(time_column)
    df_test = df_test.sort_values(time_column).drop_duplicates(time_column)

    print("[DEBUG] AFTER DROP DUP TRAIN:", len(df_train))
    print("[DEBUG] AFTER DROP DUP TEST:", len(df_test))

    freq = _infer_freq(df_train, time_column)

    df_train = _prepare(df_train, time_column, freq)
    df_test = _prepare(df_test, time_column, freq)

    exog_cols = list(col_for_train)

    df_train[col_target] = pd.to_numeric(df_train[col_target], errors="coerce")
    df_train[exog_cols] = df_train[exog_cols].astype(float)

    print("[DEBUG] NaN target before drop:", df_train[col_target].isna().sum())

    df_train = df_train.dropna(subset=[col_target])

    print("[DEBUG] AFTER DROPNA TRAIN:", len(df_train))

    # =========================
    # TARGET SERIES
    # =========================
    target_series = TimeSeries.from_dataframe(
        df_train,
        time_col=time_column,
        value_cols=col_target,
        fill_missing_dates=True,
        freq=freq
    ).astype(np.float32)

    # =========================
    # COVARIATES
    # =========================
    past_cov = TimeSeries.from_dataframe(
        df_train,
        time_col=time_column,
        value_cols=exog_cols,
        fill_missing_dates=True,
        freq=freq
    ).astype(np.float32)

    print("[DEBUG] TARGET SERIES LENGTH:", len(target_series))
    print("[DEBUG] COV SERIES LENGTH:", len(past_cov))

    print("[DEBUG] TARGET NAN:", np.isnan(target_series.values()).sum())
    print("[DEBUG] COV NAN:", np.isnan(past_cov.values()).sum())

    print("[DEBUG] INPUT CHUNK:", lag)

    assert len(target_series) > lag, "Too short series for lag"
    assert len(target_series) == len(past_cov), "Target/Cov mismatch"
    assert np.isnan(target_series.values()).sum() == 0, "NaN in target"
    assert np.isnan(past_cov.values()).sum() == 0, "NaN in cov"

    # =========================
    # MODEL
    # =========================
    model = DLinearModel(
        input_chunk_length=lag,
        output_chunk_length=cfg["output_chunk_length"],
        n_epochs=cfg["n_epochs"],
        batch_size=cfg["batch_size"],
        random_state=SEED,
        optimizer_kwargs={"lr": cfg["optimizer_lr"]},
        pl_trainer_kwargs=_get_pl_kwargs()
    )

    print("\n[DEBUG] START FIT\n")

    model.fit(target_series, past_covariates=past_cov)

    print("\n[DEBUG] FIT DONE\n")

    preds = []

    history_target = target_series
    history_cov = past_cov

    for i in range(len(df_test)):

        pred = model.predict(
            n=1,
            series=history_target,
            past_covariates=history_cov
        )

        val = float(pred.values().ravel()[0])

        print(f"[DEBUG] step={i}, pred={val}")

        preds.append(val)

        next_time = df_test[time_column].iloc[i]

        new_target = pd.DataFrame({
            time_column: [next_time],
            col_target: [val]
        })

        new_cov = pd.DataFrame({
            time_column: [next_time],
            **{c: [df_test[c].iloc[i]] for c in exog_cols}
        })

        new_target_ts = TimeSeries.from_dataframe(
            new_target,
            time_col=time_column,
            value_cols=col_target,
            fill_missing_dates=True,
            freq=freq
        ).astype(np.float32)

        new_cov_ts = TimeSeries.from_dataframe(
            new_cov,
            time_col=time_column,
            value_cols=exog_cols,
            fill_missing_dates=True,
            freq=freq
        ).astype(np.float32)

        history_target = history_target.append(new_target_ts)
        history_cov = history_cov.append(new_cov_ts)

    print("\n========== END DEBUG ==========\n")

    return pd.DataFrame({
        time_column: df_test[time_column].values,
        col_target: preds
    })