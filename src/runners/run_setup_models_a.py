"""
pdm run src/runners/run_setup_models_a.py
"""

import os
import time
import ast
import sys
import asyncio
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from src.configs.experiment_conf import init_experiment_config
from src.setups.experiment_setup import (
    init_experiment_setup,
    load_and_prepare_progress,
    get_pending_experiments,
    append_experiment_to_csv,
    append_progress_to_csv
)

from config import logger_language
from src.pipelines.setup_pipeline import SetupModel

MAX_WORKERS = 8


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

results_path = os.path.join(experiment_path, "results_setups")
os.makedirs(results_path, exist_ok=True)

df_experiment_design = init_experiment_setup()

experiment_design_path = os.path.join(results_path, "experiment_design_setups.csv")
progress_csv_path = os.path.join(results_path, "progress_setups.csv")

df_experiment_design.to_csv(experiment_design_path)
logger.info(msg["experiment_created"].format(experiment_design_path))

df_experiment_design = df_experiment_design[
    (df_experiment_design["trajectory_cols"] == "baseline")
]

df_experiment_design_new_ds = df_experiment_design.copy()
df_experiment_design_new_method = df_experiment_design.copy()

df_experiment_design = df_experiment_design[df_experiment_design["dataset_name"] == "Metro_Traffic"]

# df_experiment_design = df_experiment_design[df_experiment_design["model"] == "NHiTS"]

# print(df_to_experiment_new_ds)
# print(df_to_experiment_new_method)
#
# def make_hashable(df):
#     df = df.copy()
#     for c in df.columns:
#         df[c] = df[c].apply(lambda x: tuple(x) if isinstance(x, list) else x)
#     return df
#
#
# def restore_lists(df):
#     df = df.copy()
#     for c in df.columns:
#         df[c] = df[c].apply(lambda x: list(x) if isinstance(x, tuple) else x)
#     return df
#
#
# df_experiment_design = pd.concat(
#     [df_to_experiment_new_ds, df_to_experiment_new_method],
#     ignore_index=True
# )

# df_experiment_design = make_hashable(df_experiment_design)
# df_experiment_design = df_experiment_design.drop_duplicates().reset_index(drop=True)
# df_experiment_design = restore_lists(df_experiment_design)

print(df_experiment_design)

df_ready_progress = load_and_prepare_progress(
    progress_csv_path=progress_csv_path,
    columns=df_experiment_design.columns
)
print("Уже готово")
print(df_ready_progress)

df_to_experiment = get_pending_experiments(
    df_experiment_design=df_experiment_design,
    df_ready_progress=df_ready_progress
)

df_to_experiment = get_pending_experiments(df_experiment_design=df_experiment_design, df_ready_progress=df_ready_progress)


def to_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x) or x == "":
        return []
    if isinstance(x, str):
        return ast.literal_eval(x)
    return []

df_to_experiment["additional_cols"] = df_to_experiment["additional_cols"].apply(to_list)



print(df_to_experiment)

print(f"К расчетам = {len(df_to_experiment)} строк")
print(df_to_experiment)


if df_to_experiment is None or df_to_experiment.empty:
    logger.info("All experiments completed")
    sys.exit(0)

model_not_support_lags = ["Prophet"]

def run_experiment(experiment):
    model = experiment["model"]
    dataset_name = experiment["dataset_name"]

    list_predict_points_to_test = experiment["list_predict_points_to_test"]
    points_to_pred = max(list_predict_points_to_test)

    setups_pipeline = SetupModel(
        experiment=experiment,
        home_path=home_path,
        export_path=export_path,
        experiment_path=experiment_path,
        logger=logger,
        messages=MESSAGES,
        logger_language=logger_language,
        test_points=points_to_pred

    )

    setups_pipeline.init_design(df_experiment_design)
    setups_pipeline.load_dataset()
    setups_pipeline.prepare_future_dataframe()
    setups_pipeline.run_time2vec()
    setups_pipeline.run_split()

    if model in model_not_support_lags:
        best_lag = 1
    else:
        setups_pipeline.run_setup_lag_by_model()
        best_lag = setups_pipeline.best_lag

    append_progress_to_csv(
        progress_row={
            "model": model,
            "dataset_name": dataset_name,
            "best_lag": best_lag
        },
        progress_csv_path=os.path.join(results_path, "setups_lag.csv")
    )

    setups_pipeline.run_setup_models_params()

    append_progress_to_csv(
        progress_row={
            "model": model,
            "dataset_name": dataset_name,
            "best_params": setups_pipeline.best_params,
            "best_metrics": setups_pipeline.best_metrics
        },
        progress_csv_path=os.path.join(results_path, "setups_params.csv")
    )

    append_experiment_to_csv(
        experiment=experiment,
        progress_csv_path=progress_csv_path
    )

    return dataset_name


async def main():
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [
            loop.run_in_executor(executor, run_experiment, row)
            for _, row in df_to_experiment.iterrows()
        ]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())