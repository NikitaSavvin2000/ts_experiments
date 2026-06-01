import os
import time
import ast
import sys
import gc
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.configs.experiment_conf import init_experiment_config
from src.setups.experiment_setup import (
    init_experiment_setup,
    load_and_prepare_progress,
    get_pending_experiments,
    append_experiment_to_csv,
    not_exogenous_models
)

from src.ts_models.ts_utils.timeseries_utils import plot_predictions
from config import logger_language
from src.pipelines.ts_pipeline import TSExperimentPipeline


MAX_WORKERS = 12


def clear_memory():
    gc.collect()


MESSAGES = {
    "en": {"experiment_created": "Experiment design created. Available at {}"},
    "ru": {"experiment_created": "Дизайн экспериментов создан. Доступен по пути {}"},
    "zh": {"experiment_created": "实验设计已创建，路径：{}"}
}

home_path, export_path, experiment_path, logger = init_experiment_config()

df_experiment_design = init_experiment_setup()

experiment_design_path = os.path.join(experiment_path, "experiment_design.csv")
progress_csv_path = os.path.join(experiment_path, "progress.csv")
trace_csv_path = os.path.join(experiment_path, "traces.csv")

charts_dir = os.path.join(experiment_path, "charts")
charts_dir_ru = os.path.join(charts_dir, "ru")
charts_dir_en = os.path.join(charts_dir, "en")

results_path = os.path.join(experiment_path, "results")

os.makedirs(results_path, exist_ok=True)
os.makedirs(charts_dir_ru, exist_ok=True)
os.makedirs(charts_dir_en, exist_ok=True)

df_experiment_design.to_csv(experiment_design_path)

msg = MESSAGES[logger_language]
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


def run_experiment(experiment):

    try:

        if experiment["model"] in not_exogenous_models and experiment["trajectory_cols"] != "baseline":
            exp_result = experiment.copy()

            iteration_results_path = os.path.join(results_path, experiment["result_dir_name"])
            os.makedirs(iteration_results_path, exist_ok=True)

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
                os.path.join(iteration_results_path, "line_result_table.csv")
            )

            append_experiment_to_csv(experiment=experiment, progress_csv_path=progress_csv_path)
            return

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
            params = ast.literal_eval(params)

        already_selected_cols = None
        list_predict_points_to_test = sorted(experiment["list_predict_points_to_test"], reverse=True)

        iteration_results_path = os.path.join(results_path, experiment["result_dir_name"])
        os.makedirs(iteration_results_path, exist_ok=True)

        for points_to_pred in list_predict_points_to_test:

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

            exp_result["r2"] = float(round(ts_pipeline.metrix_dict["r2"], 3))
            exp_result["mae"] = float(round(ts_pipeline.metrix_dict["mae"], 3))
            exp_result["rmse"] = float(round(ts_pipeline.metrix_dict["rmse"], 3))
            exp_result["mape"] = float(round(ts_pipeline.metrix_dict["mape"], 3))
            exp_result["smape"] = float(round(ts_pipeline.metrix_dict["smape"], 3))
            exp_result["wape"] = float(round(ts_pipeline.metrix_dict["wape"], 3))
            exp_result["bias"] = float(round(ts_pipeline.metrix_dict["bias"], 3))
            exp_result["medae"] = float(round(ts_pipeline.metrix_dict["medae"], 3))
            exp_result["nrmse"] = float(round(ts_pipeline.metrix_dict["nrmse"], 3))

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

            ts_pipeline.df_test_pred.to_csv(
                os.path.join(iteration_results_path, f"test_pred_norm_ptp_{points_to_pred}.csv")
            )

            ts_pipeline.df_test_pred_not_norm.to_csv(
                os.path.join(iteration_results_path, f"test_pred_not_norm_ptp_{points_to_pred}.csv")
            )

            if already_selected_cols is None:
                already_selected_cols = ts_pipeline.col_for_train

            clear_memory()

        append_experiment_to_csv(experiment=experiment, progress_csv_path=progress_csv_path)

        clear_memory()

    except Exception as e:
        logger.error(str(e))
        clear_memory()


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(run_experiment, exp) for _, exp in df_to_experiment.iterrows()]

    for f in as_completed(futures):
        f.result()