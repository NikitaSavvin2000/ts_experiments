# """
# pdm run src/runners/main.py
# """
# import os
# import time
# import sys
# import pandas as pd

# from src.configs.experiment_conf import init_experiment_config
# from src.setups.experiment_setup import (init_experiment_setup,
#                                          load_and_prepare_progress,
#                                          get_pending_experiments,
#                                          append_experiment_to_csv,
#                                          not_exogenous_models)

# from config import logger_language
# from src.pipelines.ts_pipeline import TSExperimentPipeline
# from tqdm import tqdm


# def scip_not_exogenous_models(experiment):

#     iteration_results_path = os.path.join(results_path, experiment["result_dir_name"])
#     os.makedirs(iteration_results_path, exist_ok=True)

#     exp_result["pacf_lag"] = None
#     exp_result["col_for_train"] = None
#     exp_result["discreteness"] = None
#     exp_result["r2"] = None
#     exp_result["mae"] = None
#     exp_result["mape"] = None
#     exp_result["rmse"] = None
#     exp_result["elapsed_seconds"] = None


#     df_result = pd.DataFrame([exp_result])

#     df_result.to_csv(os.path.join(iteration_results_path, "line_result_table.csv"))
#     append_experiment_to_csv(experiment=experiment, progress_csv_path=progress_csv_path)


# # ============================================
# # en: Multilingual logging messages for experiment pipeline
# # ru: Мультиязычные сообщения логирования для экспериментального пайплайна
# # zh: 实验流水线的多语言日志消息
# # ============================================
# # Define in src/config.py
# # ============================================
# MESSAGES = {
#     "en": {
#         "experiment_created": "Experiment design created. Available at {}",
#     },
#     "ru": {
#         "experiment_created": "Дизайн экспериментов создан. Доступен по пути {}",
#     },
#     "zh": {
#         "experiment_created": "实验设计已创建，路径：{}",
#     }
# }
# msg = MESSAGES[logger_language]


# # ============================================
# # en: Experiment initialization (folders, logger, snapshot)
# # ru: Инициализация эксперимента (папки, логгер, snapshot)
# # zh: 实验初始化（文件夹、日志器、快照）
# # ============================================
# home_path, export_path, experiment_path, logger = init_experiment_config()

# # ============================================
# # en: Experiment design configuration initialization
# # ru: Инициализация конфигурации дизайна экспериментов
# # zh: 实验设计配置初始化
# # ============================================
# df_experiment_design = init_experiment_setup()

# experiment_design_path = os.path.join(experiment_path,"experiment_design.csv")
# progress_csv_path = os.path.join(experiment_path, "progress.csv")
# results_path = os.path.join(experiment_path, "results")
# os.makedirs(results_path, exist_ok=True)

# df_experiment_design.to_csv(experiment_design_path)
# logger.info(msg["experiment_created"].format(experiment_design_path))


# df_ready_progress = load_and_prepare_progress(progress_csv_path=progress_csv_path, columns=df_experiment_design.columns)
# df_to_experiment = get_pending_experiments(df_experiment_design=df_experiment_design, df_ready_progress=df_ready_progress)

# if df_to_experiment is None or df_to_experiment.empty:
#     logger.info("All experiments completed")
#     sys.exit(0)

# # ============================================
# # en: Experiment grid generation for models
# # ru: Генерация сетки экспериментов для моделей
# # zh: 模型实验网格生成
# # ============================================
# for _, experiment in tqdm(df_to_experiment.iterrows()):

#     if experiment["model"] in not_exogenous_models:
#         if experiment["trajectory_cols"] != "baseline":
#             scip_not_exogenous_models(experiment)
#             continue
#     # ============================================
#     # en: Initialize experiment pipeline instance
#     # ru: Инициализация экземпляра пайплайна эксперимента
#     # zh: 初始化实验管道实例
#     # ============================================
#     ts_pipeline = TSExperimentPipeline(
#         experiment=experiment,
#         home_path=home_path,
#         export_path=export_path,
#         experiment_path=experiment_path,
#         logger=logger,
#         messages=MESSAGES,
#         logger_language=logger_language
#     )

#     # ============================================
#     # en: Save experiment grid configuration
#     # ru: Сохранение конфигурации сетки экспериментов
#     # zh: 保存实验网格配置
#     # ============================================
#     ts_pipeline.init_design(df_experiment_design)

#     # ============================================
#     # en: Load dataset for current experiment
#     # ru: Загрузка датасета для текущего эксперимента
#     # zh: 加载当前实验的数据集
#     # ============================================
#     ts_pipeline.load_dataset()


#     start = time.perf_counter()

#     ts_pipeline.prepare_future_dataframe()

#     # ============================================
#     # en: Apply Time2Vec temporal encoding
#     # ru: Применение временного кодирования Time2Vec
#     # zh: 应用 Time2Vec 时间编码
#     # ============================================
#     ts_pipeline.run_time2vec()

#     # ============================================
#     # en: Select optimal lag using PACF
#     # ru: Выбор оптимального лага с помощью PACF
#     # zh: 使用 PACF 选择最优滞后
#     # ============================================
#     # ts_pipeline.run_lag_pacf()

#     # ============================================
#     # en: Split dataset into train and test sets
#     # ru: Разделение данных на train и test выборки
#     # zh: 将数据划分为训练集和测试集
#     # ============================================
#     ts_pipeline.run_split()

#     # ============================================
#     # en: Statistical feature selection for forecasting
#     # ru: Статистический отбор признаков для прогнозирования
#     # zh: 预测任务的统计特征选择
#     # ============================================
#     # ts_pipeline.fetch_stat_select_features()

#     ts_pipeline.run_test_predict()


#     iteration_results_path = os.path.join(results_path, experiment["result_dir_name"])
#     os.makedirs(iteration_results_path, exist_ok=True)

#     end = time.perf_counter()

#     elapsed_seconds = end - start

#     exp_result = experiment.copy()

#     exp_result["pacf_lag"] = ts_pipeline.pacf_lag
#     exp_result["col_for_train"] = ts_pipeline.col_for_train
#     exp_result["discreteness"] = ts_pipeline.discreteness_sec
#     exp_result["r2"] = ts_pipeline.metrix_dict["r2"]
#     exp_result["mae"] = ts_pipeline.metrix_dict["mae"]
#     exp_result["mape"] = ts_pipeline.metrix_dict["mape"]
#     exp_result["rmse"] = ts_pipeline.metrix_dict["rmse"]
#     exp_result["elapsed_seconds"] = elapsed_seconds


#     df_result = pd.DataFrame([exp_result])

#     df_result.to_csv(os.path.join(iteration_results_path, "line_result_table.csv"))

#     ts_pipeline.df_test_pred.to_csv(os.path.join(iteration_results_path, "test_pred_norm.csv"))
#     ts_pipeline.df_test_pred_not_norm.to_csv(os.path.join(iteration_results_path, "test_pred_not_norm.csv"))

#     append_experiment_to_csv(experiment=experiment, progress_csv_path=progress_csv_path)



"""
pdm run src/runners/main.py
"""
import os
import time
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


MAX_WORKERS = 5


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
        # df_setups_lags,
        # df_setups_params,
):

    model = experiment["model"]
    dataset_name = experiment["dataset_name"]


    # lag = df_setups_lags.loc[
    #     (df_setups_lags["model"] == model) &
    #     (df_setups_lags["dataset_name"] == dataset_name),
    #     "best_lag"
    # ].iloc[0]



    # params = df_setups_params.loc[
    #     (df_setups_lags["model"] == model) &
    #     (df_setups_lags["dataset_name"] == dataset_name),
    #     "best_params"
    # ].iloc[0]

    if model in not_exogenous_models:
        if experiment["trajectory_cols"] != "baseline":
            scip_not_exogenous_models(experiment, results_path, progress_csv_path)
            return None

    params = None
    lag = 22

    ts_pipeline = TSExperimentPipeline(
        experiment=experiment,
        home_path=home_path,
        export_path=export_path,
        experiment_path=experiment_path,
        logger=logger,
        messages=MESSAGES,
        lag=lag,
        params=params,
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

    exp_result["pacf_lag"] = ts_pipeline.pacf_lag
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

df_to_experiment = df_to_experiment[df_to_experiment["dataset_name"] == "morocco_zone_1"]
df_to_experiment = df_to_experiment[df_to_experiment["model"] == "XGBoost"]

if df_to_experiment is None or df_to_experiment.empty:
    logger.info("All experiments completed")
    sys.exit(0)


def main():
    with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=init_worker
    ) as executor:

        futures = []

        for _, experiment in tqdm(df_to_experiment.iterrows()):
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
                    # df_setups_lags,
                    # df_setups_params,
                )
            )

        for f in tqdm(as_completed(futures), total=len(futures)):
            _ = f.result()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()