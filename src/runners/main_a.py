import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
import ast
import sys
import gc
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from src.configs.experiment_conf import init_experiment_config
from src.setups.experiment_setup import (
    init_experiment_setup,
    load_and_prepare_progress,
    get_pending_experiments,
    append_experiment_to_csv,
    not_exogenous_models
)

from config import logger_language
from src.pipelines.ts_pipeline import TSExperimentPipeline


import torch

torch.set_num_threads(1)


home_path, export_path, experiment_path, logger = init_experiment_config()

df_experiment_design = init_experiment_setup()

experiment_design_path = os.path.join(experiment_path, "experiment_design.csv")
progress_csv_path = os.path.join(experiment_path, "progress.csv")
trace_csv_path = os.path.join(experiment_path, "traces.csv")

results_path = os.path.join(experiment_path, "results")

os.makedirs(results_path, exist_ok=True)

df_experiment_design.to_csv(experiment_design_path)

df_ready_progress = load_and_prepare_progress(
    progress_csv_path=progress_csv_path,
    columns=df_experiment_design.columns
)

df_to_experiment = get_pending_experiments(
    df_experiment_design=df_experiment_design,
    df_ready_progress=df_ready_progress
)

if df_to_experiment is None or df_to_experiment.empty:
    sys.exit(0)

setups_lags_csv = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR2ObHgVx2M6a7rvS5TcIhkxCjGQh3891WcpV8EYUV3vG-FsQAbInhA3xvqCbaPD0slfot2MkBL7ZKL/pub?gid=360674600&single=true&output=csv"
setups_params_csv = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR2ObHgVx2M6a7rvS5TcIhkxCjGQh3891WcpV8EYUV3vG-FsQAbInhA3xvqCbaPD0slfot2MkBL7ZKL/pub?gid=86420863&single=true&output=csv"

df_setups_lags = pd.read_csv(setups_lags_csv)
df_setups_params = pd.read_csv(setups_params_csv)

CPU = os.cpu_count() or 1
MAX_WORKERS = min(6, max(3, CPU // 4))
MAX_IN_FLIGHT = MAX_WORKERS * 2


def worker_init():
    gc.enable()
    gc.collect()


def run_experiment(experiment):
    try:
        gc.collect()

        model = experiment["model"]
        dataset_name = experiment["dataset_name"]

        iteration_results_path = os.path.join(results_path, experiment["result_dir_name"])
        os.makedirs(iteration_results_path, exist_ok=True)

        if model in not_exogenous_models and experiment["trajectory_cols"] != "baseline":
            exp_result = experiment.copy()

            exp_result.update({
                "pacf_lag": None,
                "col_for_train": None,
                "discreteness": None,
                "r2": None,
                "mae": None,
                "rmse": None,
                "mape": None,
                "smape": None,
                "wape": None,
                "bias": None,
                "medae": None,
                "nrmse": None,
                "elapsed_seconds": None
            })

            pd.DataFrame([exp_result]).to_csv(
                os.path.join(iteration_results_path, "line_result_table.csv"),
                index=False
            )

            append_experiment_to_csv(
                experiment=experiment,
                progress_csv_path=progress_csv_path
            )

            gc.collect()
            return True

        lag_row = df_setups_lags[
            (df_setups_lags["model"] == model) &
            (df_setups_lags["dataset_name"] == dataset_name)
            ]["best_lag"]

        lag = lag_row.iloc[0] if not lag_row.empty else None

        params_row = df_setups_params[
            (df_setups_params["model"] == model) &
            (df_setups_params["dataset_name"] == dataset_name)
            ]["best_params"]

        params = params_row.iloc[0] if not params_row.empty else None

        if params:
            params = ast.literal_eval(params)

        already_selected_cols = None
        test_points_list = sorted(experiment["list_predict_points_to_test"], reverse=True)

        for points_to_pred in test_points_list:

            ts_pipeline = TSExperimentPipeline(
                experiment=experiment,
                home_path=home_path,
                export_path=export_path,
                experiment_path=experiment_path,
                logger=logger,
                messages=None,
                lag=lag,
                params=params,
                trace_csv_path=trace_csv_path,
                logger_language=logger_language,
                already_selected_cols=already_selected_cols,
                test_points=points_to_pred
            )

            ts_pipeline.init_design(df_experiment_design)
            ts_pipeline.load_dataset()

            start = time.perf_counter()

            ts_pipeline.prepare_future_dataframe()
            ts_pipeline.run_time2vec()
            ts_pipeline.run_split()
            ts_pipeline.run_test_predict()

            end = time.perf_counter()

            exp_result = experiment.copy()

            exp_result["pacf_lag"] = lag
            exp_result["col_for_train"] = ts_pipeline.col_for_train
            exp_result["discreteness"] = ts_pipeline.discreteness_sec

            for k, v in ts_pipeline.metrix_dict.items():
                exp_result[k] = float(round(v, 3))

            exp_result["elapsed_seconds"] = end - start
            exp_result["points_to_pred"] = points_to_pred

            exp_result["start_train_date"] = ts_pipeline.start_train_date
            exp_result["end_train_date"] = ts_pipeline.end_train_date
            exp_result["start_test_date"] = ts_pipeline.start_test_date
            exp_result["end_test_date"] = ts_pipeline.end_test_date

            append_experiment_to_csv(
                experiment=exp_result,
                progress_csv_path=os.path.join(iteration_results_path, "line_result_table.csv")
            )

            if already_selected_cols is None:
                already_selected_cols = ts_pipeline.col_for_train

            del ts_pipeline
            del exp_result
            gc.collect()

        gc.collect()
        return True

    except Exception:
        gc.collect()
        return False


if __name__ == "__main__":

    experiments = [exp for _, exp in df_to_experiment.iterrows()]

    in_flight = set()

    with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=worker_init
    ) as executor:

        idx = 0

        while idx < len(experiments) or in_flight:

            while idx < len(experiments) and len(in_flight) < MAX_IN_FLIGHT:
                future = executor.submit(run_experiment, experiments[idx])
                in_flight.add(future)
                idx += 1

            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)

            for f in done:
                try:
                    f.result()
                except Exception:
                    pass

            gc.collect()

        gc.collect()