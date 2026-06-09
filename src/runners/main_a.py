"""
pdm run src/runners/main_a.py
"""
import os
import time
import ast
import sys
import pandas as pd
import gc

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
from tqdm import tqdm

WORKERS = 6

import os
import random
import numpy as np
import torch
import tensorflow as tf

def init_deterministic(seed: int = 42):

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    import random
    import numpy as np
    import torch
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

init_deterministic(42)

def scip_not_exogenous_models(experiment):
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
    exp_result["smape"] = None
    exp_result["wape"] = None
    exp_result["bias"] = None
    exp_result["medae"] = None
    exp_result["nrmse"] = None
    exp_result["elapsed_seconds"] = None

    df_result = pd.DataFrame([exp_result])
    df_result.to_csv(os.path.join(iteration_results_path, "line_result_table.csv"), index=False)

    append_experiment_to_csv(experiment=experiment, progress_csv_path=progress_csv_path)


MESSAGES = {
    "en": {"experiment_created": "Experiment design created. Available at {}"},
    "ru": {"experiment_created": "Дизайн экспериментов создан. Доступен по пути {}"},
    "zh": {"experiment_created": "实验设计已创建，路径：{}"}
}

msg = MESSAGES[logger_language]

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
os.makedirs(charts_dir, exist_ok=True)
os.makedirs(charts_dir_ru, exist_ok=True)
os.makedirs(charts_dir_en, exist_ok=True)

df_experiment_design.to_csv(experiment_design_path, index=False)
logger.info(msg["experiment_created"].format(experiment_design_path))

df_ready_progress = load_and_prepare_progress(
    progress_csv_path=progress_csv_path,
    columns=df_experiment_design.columns
)

df_to_experiment = get_pending_experiments(
    df_experiment_design=df_experiment_design,
    df_ready_progress=df_ready_progress
)


df_to_experiment = df_to_experiment[df_to_experiment["dataset_name"] == "morocco_zone_1"]
df_to_experiment = df_to_experiment[df_to_experiment["model"] == "NHiTS"]
#
#
# # test_trajectory_cols = ["optuna_features", "horizon_selected_features"]
test_trajectory_cols = ["optuna_features"]
#
#
df_to_experiment = df_to_experiment[
    df_to_experiment["trajectory_cols"].isin(test_trajectory_cols)
]


if df_to_experiment is None or df_to_experiment.empty:
    logger.info("All experiments completed")
    sys.exit(0)

setups_lags_csv = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR2ObHgVx2M6a7rvS5TcIhkxCjGQh3891WcpV8EYUV3vG-FsQAbInhA3xvqCbaPD0slfot2MkBL7ZKL/pub?gid=360674600&single=true&output=csv"
setups_params_csv = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR2ObHgVx2M6a7rvS5TcIhkxCjGQh3891WcpV8EYUV3vG-FsQAbInhA3xvqCbaPD0slfot2MkBL7ZKL/pub?gid=86420863&single=true&output=csv"

df_setups_lags = pd.read_csv(setups_lags_csv)
df_setups_params = pd.read_csv(setups_params_csv)


def run_experiment(experiment):
    if experiment["model"] in not_exogenous_models:
        if experiment["trajectory_cols"] != "baseline":
            scip_not_exogenous_models(experiment)
            return

    model = experiment["model"]
    dataset_name = experiment["dataset_name"]
    trajectory = experiment["trajectory_cols"]

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
        elapsed_seconds = end - start

        iteration_results_path = os.path.join(results_path, experiment["result_dir_name"])
        os.makedirs(iteration_results_path, exist_ok=True)

        exp_result = experiment.copy()

        exp_result["pacf_lag"] = lag
        exp_result["col_for_train"] = ts_pipeline.col_for_train
        exp_result["discreteness"] = ts_pipeline.discreteness_sec

        for k in ts_pipeline.metrix_dict:
            exp_result[k] = float(round(ts_pipeline.metrix_dict[k], 3))

        exp_result["elapsed_seconds"] = elapsed_seconds
        exp_result["points_to_pred"] = points_to_pred

        exp_result["start_train_date"] = ts_pipeline.start_train_date
        exp_result["end_train_date"] = ts_pipeline.end_train_date
        exp_result["start_test_date"] = ts_pipeline.start_test_date
        exp_result["end_test_date"] = ts_pipeline.end_test_date

        line_result_table_path = os.path.join(iteration_results_path, "line_result_table.csv")

        append_experiment_to_csv(
            experiment=exp_result,
            progress_csv_path=line_result_table_path
        )

        ts_pipeline.df_test_pred.to_csv(
            os.path.join(iteration_results_path, f"test_pred_norm_ptp_{points_to_pred}.csv"),
            index=False
        )

        ts_pipeline.df_test_pred_not_norm.to_csv(
            os.path.join(iteration_results_path, f"test_pred_not_norm_ptp_{points_to_pred}.csv"),
            index=False
        )

        if already_selected_cols is None:
            already_selected_cols = ts_pipeline.col_for_train

        ru_dir = os.path.join(charts_dir_ru, f"{points_to_pred}_predict_points")
        en_dir = os.path.join(charts_dir_en, f"{points_to_pred}_predict_points")

        os.makedirs(ru_dir, exist_ok=True)
        os.makedirs(en_dir, exist_ok=True)

        charts_path_ru = os.path.join(ru_dir, f"{dataset_name}_{model}_{trajectory}.png")
        charts_path_en = os.path.join(en_dir, f"{dataset_name}_{model}_{trajectory}.png")

        # plot_predictions(
        #     df=ts_pipeline.df_test_pred,
        #     time_col="datetime",
        #     pred_col="pred",
        #     real_col="true",
        #     title=f"Dataset - {dataset_name} | Model - {model} | Trajectory - {trajectory}",
        #     xlabel="Datetime",
        #     ylabel="Value",
        #     title_pred="Prediction",
        #     title_real="Real",
        #     metrix_dict=ts_pipeline.metrix_dict,
        #     save_filename=charts_path_en,
        #     figsize=(16, 6)
        # )
        #
        # plot_predictions(
        #     df=ts_pipeline.df_test_pred,
        #     time_col="datetime",
        #     pred_col="pred",
        #     real_col="true",
        #     title=f"Датасет - {dataset_name} | Модель - {model} | Траектория - {trajectory}",
        #     xlabel="Время",
        #     ylabel="Значение",
        #     title_pred="Предсказанное",
        #     title_real="Действительное",
        #     metrix_dict=ts_pipeline.metrix_dict,
        #     save_filename=charts_path_ru,
        #     figsize=(16, 6)
        # )
        del ts_pipeline
        gc.collect()

    append_experiment_to_csv(
        experiment=experiment,
        progress_csv_path=progress_csv_path
    )


for dataset_name, df_batch in df_to_experiment.groupby("dataset_name"):
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(run_experiment, exp) for _, exp in df_batch.iterrows()]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass