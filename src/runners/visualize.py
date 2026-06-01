"""
pdm run src/runners/visualize.py
"""
import os
import time
import ast
import sys
import pandas as pd

from src.configs.experiment_conf import init_experiment_config
from src.setups.experiment_setup import (init_experiment_setup,
                                         load_and_prepare_progress,
                                         get_pending_experiments,
                                         append_experiment_to_csv,
                                         not_exogenous_models)

from src.ts_models.ts_utils.timeseries_utils import plot_predictions


from config import logger_language
from src.pipelines.ts_pipeline import TSExperimentPipeline
from tqdm import tqdm



import pandas as pd
import matplotlib.pyplot as plt

def plot_last_n_points(dataset_csv, col_time, col_target, n, title="", figsize=(16, 6)):
    df = pd.read_csv(dataset_csv)
    df = df.sort_values(col_time).tail(n)

    plt.figure(figsize=figsize)

    plt.plot(
        df[col_time],
        df[col_target],
        color="blue"
    )

    plt.title(title)
    plt.xlabel(col_time)
    plt.ylabel(col_target)

    plt.grid(False)

    plt.xticks([])

    plt.tight_layout()
    plt.show()

def scip_not_exogenous_models(experiment):

    iteration_results_path = os.path.join(results_path, experiment["result_dir_name"])
    os.makedirs(iteration_results_path, exist_ok=True)

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
    append_experiment_to_csv(experiment=experiment, progress_csv_path=progress_csv_path)


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

experiment_design_path = os.path.join(experiment_path,"experiment_design.csv")
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

df_experiment_design.to_csv(experiment_design_path)
logger.info(msg["experiment_created"].format(experiment_design_path))


df_ready_progress = load_and_prepare_progress(progress_csv_path=progress_csv_path, columns=df_experiment_design.columns)
df_to_experiment = get_pending_experiments(df_experiment_design=df_experiment_design, df_ready_progress=df_ready_progress)

if df_to_experiment is None or df_to_experiment.empty:
    logger.info("All experiments completed")
    sys.exit(0)


setups_lags_csv = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR2ObHgVx2M6a7rvS5TcIhkxCjGQh3891WcpV8EYUV3vG-FsQAbInhA3xvqCbaPD0slfot2MkBL7ZKL/pub?gid=360674600&single=true&output=csv"
setups_params_csv = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR2ObHgVx2M6a7rvS5TcIhkxCjGQh3891WcpV8EYUV3vG-FsQAbInhA3xvqCbaPD0slfot2MkBL7ZKL/pub?gid=86420863&single=true&output=csv"
df_setups_lags = pd.read_csv(setups_lags_csv)
df_setups_params = pd.read_csv(setups_params_csv)

df_unique = df_to_experiment[
    ["dataset_name", "dataset_csv", "col_time", "col_target"]
].drop_duplicates()

for _, experiment in tqdm(df_unique.iterrows()):

    dataset_name = experiment["dataset_name"]
    dataset_csv = experiment["dataset_csv"]
    col_time = experiment["col_time"]
    col_target = experiment["col_target"]

    df = pd.read_csv(dataset_csv)

    plot_last_n_points(dataset_csv, col_time, col_target, 300, dataset_name)



