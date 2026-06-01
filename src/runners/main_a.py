import os
import time
import ast
import gc
import sys
import pandas as pd
import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.configs.experiment_conf import init_experiment_config
from src.setups.experiment_setup import (
    init_experiment_setup,
    load_and_prepare_progress,
    get_pending_experiments,
    append_experiment_to_csv,
    not_exogenous_models
)

from src.pipelines.ts_pipeline import TSExperimentPipeline
from src.ts_models.ts_utils.timeseries_utils import plot_predictions
from config import logger_language

MAX_WORKERS = 12


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
    exp_result["smape"] = None
    exp_result["wape"] = None
    exp_result["bias"] = None
    exp_result["medae"] = None
    exp_result["nrmse"] = None
    exp_result["elapsed_seconds"] = None

    pd.DataFrame([exp_result]).to_csv(
        os.path.join(iteration_results_path, "line_result_table.csv"),
        index=False
    )

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
        charts_dir_ru,
        charts_dir_en,
        progress_csv_path,
        df_experiment_design,
        logger,
        MESSAGES,
        logger_language,
        df_setups_lags,
        df_setups_params,
        trace_csv_path
):
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

    if model in not_exogenous_models and experiment["trajectory_cols"] != "baseline":
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

    m = ts_pipeline.metrix_dict
    exp_result["r2"] = float(round(m["r2"], 3))
    exp_result["mae"] = float(round(m["mae"], 3))
    exp_result["rmse"] = float(round(m["rmse"], 3))
    exp_result["mape"] = float(round(m["mape"], 3))
    exp_result["smape"] = float(round(m["smape"], 3))
    exp_result["wape"] = float(round(m["wape"], 3))
    exp_result["bias"] = float(round(m["bias"], 3))
    exp_result["medae"] = float(round(m["medae"], 3))
    exp_result["nrmse"] = float(round(m["nrmse"], 3))

    exp_result["elapsed_seconds"] = end - start

    exp_result["start_train_date"] = ts_pipeline.start_train_date
    exp_result["end_train_date"] = ts_pipeline.end_train_date
    exp_result["start_test_date"] = ts_pipeline.start_test_date
    exp_result["end_test_date"] = ts_pipeline.end_test_date

    exp_result["points_to_pred"] = experiment.get("points_to_pred", None)

    pd.DataFrame([exp_result]).to_csv(
        os.path.join(iteration_results_path, "line_result_table.csv"),
        index=False
    )

    ts_pipeline.df_test_pred.to_csv(
        os.path.join(iteration_results_path, "test_pred_norm.csv"),
        index=False
    )

    ts_pipeline.df_test_pred_not_norm.to_csv(
        os.path.join(iteration_results_path, "test_pred_not_norm.csv"),
        index=False
    )

    charts_path_ru = os.path.join(
        charts_dir_ru,
        f"{dataset_name}_{model}_{trajectory}.png"
    )

    charts_path_en = os.path.join(
        charts_dir_en,
        f"{dataset_name}_{model}_{trajectory}.png"
    )

    title_en = f"Dataset - {dataset_name} | Model - {model} | Trajectory - {trajectory}"
    title_ru = f"Датасет - {dataset_name} | Модель - {model} | Траектория - {trajectory}"

    plot_predictions(
        df=ts_pipeline.df_test_pred,
        time_col="datetime",
        pred_col="pred",
        real_col="true",
        title=title_en,
        xlabel="Datetime",
        ylabel="Value",
        title_pred="Prediction",
        title_real="Real",
        metrix_dict=m,
        save_filename=charts_path_en,
        figsize=(16, 6)
    )

    plot_predictions(
        df=ts_pipeline.df_test_pred,
        time_col="datetime",
        pred_col="pred",
        real_col="true",
        title=title_ru,
        xlabel="Время",
        ylabel="Значение",
        title_pred="Предсказанное",
        title_real="Действительное",
        metrix_dict=m,
        save_filename=charts_path_ru,
        figsize=(16, 6)
    )

    append_experiment_to_csv(
        experiment=experiment,
        progress_csv_path=progress_csv_path
    )

    return exp_result


def clear_memory():
    gc.collect()


def process_batch(df_batch, ctx):
    futures = []

    with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=init_worker
    ) as executor:

        for _, row in df_batch.iterrows():
            experiment = row.to_dict()

            futures.append(
                executor.submit(
                    run_experiment,
                    experiment,
                    ctx["home_path"],
                    ctx["export_path"],
                    ctx["experiment_path"],
                    ctx["results_path"],
                    ctx["charts_dir_ru"],
                    ctx["charts_dir_en"],
                    ctx["progress_csv_path"],
                    ctx["df_experiment_design"],
                    ctx["logger"],
                    ctx["MESSAGES"],
                    ctx["logger_language"],
                    ctx["df_setups_lags"],
                    ctx["df_setups_params"],
                    ctx["trace_csv_path"]
                )
            )

        for f in tqdm(as_completed(futures), total=len(futures)):
            _ = f.result()

    clear_memory()


def main():
    home_path, export_path, experiment_path, logger = init_experiment_config()

    df_experiment_design = init_experiment_setup()

    progress_csv_path = os.path.join(experiment_path, "progress.csv")
    results_path = os.path.join(experiment_path, "results")

    charts_dir = os.path.join(experiment_path, "charts")
    charts_dir_ru = os.path.join(charts_dir, "ru")
    charts_dir_en = os.path.join(charts_dir, "en")

    trace_csv_path = os.path.join(experiment_path, "traces.csv")

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(charts_dir_ru, exist_ok=True)
    os.makedirs(charts_dir_en, exist_ok=True)

    df_experiment_design.to_csv(
        os.path.join(experiment_path, "experiment_design.csv"),
        index=False
    )

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

    ctx = {
        "home_path": home_path,
        "export_path": export_path,
        "experiment_path": experiment_path,
        "results_path": results_path,
        "charts_dir_ru": charts_dir_ru,
        "charts_dir_en": charts_dir_en,
        "progress_csv_path": progress_csv_path,
        "df_experiment_design": df_experiment_design,
        "logger": logger,
        "MESSAGES": {
            "en": {"experiment_created": "Experiment design created. Available at {}"},
            "ru": {"experiment_created": "Дизайн экспериментов создан. Доступен по пути {}"},
            "zh": {"experiment_created": "实验设计已创建，路径：{}"}
        },
        "logger_language": logger_language,
        "df_setups_lags": df_setups_lags,
        "df_setups_params": df_setups_params,
        "trace_csv_path": trace_csv_path
    }

    grouped_batches = list(df_to_experiment.groupby("dataset_name"))

    for _, df_batch in tqdm(grouped_batches):
        process_batch(df_batch, ctx)
        del df_batch
        clear_memory()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()