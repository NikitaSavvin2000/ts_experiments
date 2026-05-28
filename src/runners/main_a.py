
"""
pdm run src/runners/main_a.py
"""
import os
import time
import ast

import sys
import pandas as pd
import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor, as_completed

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
from tqdm import tqdm


MAX_WORKERS = 8


def init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"


def scip_not_exogenous_models(experiment, results_path, progress_csv_path):
    iteration_results_path = os.path.join(results_path, experiment["result_dir_name"])
    os.makedirs(iteration_results_path, exist_ok=True)

    exp_result = experiment.copy()

    exp_result["pacf_lag"] = None
    exp_result["col_for_train"] = None
    exp_result["discreteness"] = None
    exp_result["r2"] = None
    exp_result["mae"] = None
    exp_result["mape"] = None
    exp_result["rmse"] = None
    exp_result["elapsed_seconds"] = None

    df_result = pd.DataFrame([exp_result])

    df_result.to_csv(os.path.join(iteration_results_path, "line_result_table.csv"))

    append_experiment_to_csv(
        experiment=experiment,
        progress_csv_path=progress_csv_path
    )


def run_experiment(
        experiment,
        home_path,
        export_path,
        experiment_path,
        results_path,
        progress_csv_path,
        df_experiment_design,
        logger,
        MESSAGES,
        logger_language,
        df_setups_lags,
        df_setups_params,
):

    model = experiment["model"]
    dataset_name = experiment["dataset_name"]


    lag_filtered = df_setups_lags.loc[
        (df_setups_lags["model"] == model) &
        (df_setups_lags["dataset_name"] == dataset_name),
        "best_lag"
    ]

    lag = lag_filtered.iloc[0] if not lag_filtered.empty else None


    params_filtered = df_setups_params.loc[
        (df_setups_params["model"] == model) &
        (df_setups_params["dataset_name"] == dataset_name),
        "best_params"
    ]

    params = params_filtered.iloc[0] if not params_filtered.empty else None

    if params is not None:
        print("="*100)
        print(f"params = {params}")
        print(f"params type = {type(params)}")

        params = ast.literal_eval(params)

    if model in not_exogenous_models:
        if experiment["trajectory_cols"] != "baseline":
            scip_not_exogenous_models(experiment, results_path, progress_csv_path)
            return None


    ts_pipeline = TSExperimentPipeline(
        experiment=experiment,
        home_path=home_path,
        export_path=export_path,
        experiment_path=experiment_path,
        logger=logger,
        messages=MESSAGES,
        lag=lag,
        params=params,
        trace_csv_path=trace_csv_path,
        logger_language=logger_language
    )

    ts_pipeline.init_design(df_experiment_design)
    ts_pipeline.load_dataset()

    start = time.perf_counter()

    ts_pipeline.prepare_future_dataframe()
    ts_pipeline.run_time2vec()
    ts_pipeline.run_split()
    ts_pipeline.run_test_predict()

    end = time.perf_counter()

    iteration_results_path = os.path.join(results_path, experiment["result_dir_name"])
    os.makedirs(iteration_results_path, exist_ok=True)

    exp_result = experiment.copy()

    exp_result["pacf_lag"] = lag
    exp_result["col_for_train"] = ts_pipeline.col_for_train
    exp_result["discreteness"] = ts_pipeline.discreteness_sec
    exp_result["r2"] = ts_pipeline.metrix_dict["r2"]
    exp_result["mae"] = ts_pipeline.metrix_dict["mae"]
    exp_result["mape"] = ts_pipeline.metrix_dict["mape"]
    exp_result["rmse"] = ts_pipeline.metrix_dict["rmse"]
    exp_result["elapsed_seconds"] = end - start

    pd.DataFrame([exp_result]).to_csv(
        os.path.join(iteration_results_path, "line_result_table.csv")
    )

    ts_pipeline.df_test_pred.to_csv(
        os.path.join(iteration_results_path, "test_pred_norm.csv")
    )

    ts_pipeline.df_test_pred_not_norm.to_csv(
        os.path.join(iteration_results_path, "test_pred_not_norm.csv")
    )

    append_experiment_to_csv(
        experiment=experiment,
        progress_csv_path=progress_csv_path
    )

    return exp_result


MESSAGES = {
    "en": {
        "experiment_created": "Experiment design created. Available at {}",
    },
    "ru": {
        "experiment_created": "Дизайн экспериментов создан. Доступен по пути {}",
    },
    "zh": {
        "experiment_created": "实验设计已创建，路径：{}",
    }
}

msg = MESSAGES[logger_language]


home_path, export_path, experiment_path, logger = init_experiment_config()

df_experiment_design = init_experiment_setup()

experiment_design_path = os.path.join(experiment_path, "experiment_design.csv")
progress_csv_path = os.path.join(experiment_path, "progress.csv")
results_path = os.path.join(experiment_path, "results")
trace_csv_path = os.path.join(experiment_path, "traces.csv")

setups_path_csv = os.path.join(results_path, "setups_lag.csv")
setups_params_csv = os.path.join(results_path, "setups_params.csv")

# df_setups_lags = pd.read_csv(setups_path_csv)
# df_setups_params = pd.read_csv(setups_params_csv)

os.makedirs(results_path, exist_ok=True)

df_experiment_design.to_csv(experiment_design_path)

logger.info(msg["experiment_created"].format(experiment_design_path))

df_ready_progress = load_and_prepare_progress(
    progress_csv_path=progress_csv_path,
    columns=df_experiment_design.columns
)

df_to_experiment = get_pending_experiments(
    df_experiment_design=df_experiment_design,
    df_ready_progress=df_ready_progress
)

if df_to_experiment is None or df_to_experiment.empty:
    logger.info("All experiments completed")
    sys.exit(0)


setups_lags_csv = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR2ObHgVx2M6a7rvS5TcIhkxCjGQh3891WcpV8EYUV3vG-FsQAbInhA3xvqCbaPD0slfot2MkBL7ZKL/pub?gid=360674600&single=true&output=csv"
setups_params_csv = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR2ObHgVx2M6a7rvS5TcIhkxCjGQh3891WcpV8EYUV3vG-FsQAbInhA3xvqCbaPD0slfot2MkBL7ZKL/pub?gid=86420863&single=true&output=csv"
df_setups_lags = pd.read_csv(setups_lags_csv)
df_setups_params = pd.read_csv(setups_params_csv)
def main():
    with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=init_worker
    ) as executor:

        futures = []

        for dataset_name, df_batch in tqdm(df_to_experiment.groupby("dataset_name")):
            for _, experiment in df_batch.iterrows():
                experiment = experiment.to_dict()

                futures.append(
                    executor.submit(
                        run_experiment,
                        experiment,
                        home_path,
                        export_path,
                        experiment_path,
                        results_path,
                        progress_csv_path,
                        df_experiment_design,
                        logger,
                        MESSAGES,
                        logger_language,
                        df_setups_lags,
                        df_setups_params,
                    )
                )

        for f in tqdm(as_completed(futures), total=len(futures)):
            _ = f.result()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()