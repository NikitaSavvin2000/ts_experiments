"""
pdm run src/runners/main.py
"""
import os
import time
import sys
import pandas as pd

from src.configs.experiment_conf import init_experiment_config
from src.setups.experiment_setup import (init_experiment_setup,
                                         load_and_prepare_progress,
                                         get_pending_experiments,
                                         append_experiment_to_csv,
                                         not_exogenous_models,
                                         append_progress_to_csv)

from config import logger_language
from src.pipelines.setup_pipeline import SetupModel
from tqdm import tqdm



# ============================================
# en: Multilingual logging messages for experiment pipeline
# ru: Мультиязычные сообщения логирования для экспериментального пайплайна
# zh: 实验流水线的多语言日志消息
# ============================================
# Define in src/config.py
# ============================================
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


# ============================================
# en: Experiment initialization (folders, logger, snapshot)
# ru: Инициализация эксперимента (папки, логгер, snapshot)
# zh: 实验初始化（文件夹、日志器、快照）
# ============================================
home_path, export_path, experiment_path, logger = init_experiment_config()

results_path = os.path.join(experiment_path, "results_setups")
os.makedirs(results_path, exist_ok=True)

# ============================================
# en: Experiment design configuration initialization
# ru: Инициализация конфигурации дизайна экспериментов
# zh: 实验设计配置初始化
# ============================================
df_experiment_design = init_experiment_setup()

experiment_design_path = os.path.join(results_path,"experiment_design_setups.csv")
progress_csv_path = os.path.join(results_path, "progress_setups.csv")


df_experiment_design.to_csv(experiment_design_path)
logger.info(msg["experiment_created"].format(experiment_design_path))


df_experiment_design = df_experiment_design[df_experiment_design["trajectory_cols"] == "baseline"]
df_experiment_design = df_experiment_design[df_experiment_design["model"] == "LSTM"]


print(df_experiment_design)


df_ready_progress = load_and_prepare_progress(progress_csv_path=progress_csv_path, columns=df_experiment_design.columns)
df_to_experiment = get_pending_experiments(df_experiment_design=df_experiment_design, df_ready_progress=df_ready_progress)

if df_to_experiment is None or df_to_experiment.empty:
    logger.info("All experiments completed")
    sys.exit(0)


# model_not_support_lags = ["Prophet", "ARIMA", "SARIMA"]

model_not_support_lags = ["Prophet",]


# ============================================
# en: Experiment grid generation for models
# ru: Генерация сетки экспериментов для моделей
# zh: 模型实验网格生成
# ============================================
for _, experiment in tqdm(df_to_experiment.iterrows()):

    model = experiment["model"]
    dataset_name = experiment["dataset_name"]

    # ============================================
    # en: Initialize experiment pipeline instance
    # ru: Инициализация экземпляра пайплайна эксперимента
    # zh: 初始化实验管道实例
    # ============================================
    setups_pipeline = SetupModel(
        experiment=experiment,
        home_path=home_path,
        export_path=export_path,
        experiment_path=experiment_path,
        logger=logger,
        messages=MESSAGES,
        logger_language=logger_language
    )

    # ============================================
    # en: Save experiment grid configuration
    # ru: Сохранение конфигурации сетки экспериментов
    # zh: 保存实验网格配置
    # ============================================
    setups_pipeline.init_design(df_experiment_design)

    # ============================================
    # en: Load dataset for current experiment
    # ru: Загрузка датасета для текущего эксперимента
    # zh: 加载当前实验的数据集
    # ============================================
    setups_pipeline.load_dataset()


    start = time.perf_counter()

    setups_pipeline.prepare_future_dataframe()

    # ============================================
    # en: Apply Time2Vec temporal encoding
    # ru: Применение временного кодирования Time2Vec
    # zh: 应用 Time2Vec 时间编码
    # ============================================
    setups_pipeline.run_time2vec()

    # ============================================
    # en: Select optimal lag using PACF
    # ru: Выбор оптимального лага с помощью PACF
    # zh: 使用 PACF 选择最优滞后
    # ============================================
    # ts_pipeline.run_lag_pacf()

    # ============================================
    # en: Split dataset into train and test sets
    # ru: Разделение данных на train и test выборки
    # zh: 将数据划分为训练集和测试集
    # ============================================
    setups_pipeline.run_split()

    # ============================================
    # en: Statistical feature selection for forecasting
    # ru: Статистический отбор признаков для прогнозирования
    # zh: 预测任务的统计特征选择
    # ============================================
    # ts_pipeline.fetch_stat_select_features()

    model = experiment["model"]
    dataset_name = experiment["dataset_name"]

    if model in model_not_support_lags:
        best_lag = 1
        best_score = 1
        best_pred = None
    else:
        setups_pipeline.run_setup_lag_by_model()

        best_lag = setups_pipeline.best_lag
        # best_score = setups_pipeline.best_score
        # best_pred = setups_pipeline.best_pred

    row_lag = {
        "model": model,
        "dataset_name": dataset_name,
        "best_lag": best_lag,
        # "best_score": best_score
    }

    append_progress_to_csv(progress_row=row_lag, progress_csv_path=os.path.join(results_path, "setups_lag.csv"))


    # setups_pipeline.pacf_lag = 25

    setups_pipeline.run_setup_models_params()

    best_params = setups_pipeline.best_params
    best_metrics_params = setups_pipeline.best_metrics

    row_params = {
        "model": experiment["model"],
        "dataset_name": experiment["dataset_name"],
        "best_params": best_params,
        "best_metrics": best_metrics_params
    }


    append_progress_to_csv(progress_row=row_params, progress_csv_path=os.path.join(results_path, "setups_params.csv"))


    append_experiment_to_csv(experiment=experiment, progress_csv_path=progress_csv_path)


