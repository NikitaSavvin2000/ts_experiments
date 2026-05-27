import os
import logging
import pandas as pd

from src.configs.data_config import datasets_csv_dict
from config import logger_language

from src.ts_models.arimax_service.arimax_pred import ARIMAX_forecast
from src.ts_models.catboost_service.catboost_pred import CatBoost_forecast
from src.ts_models.lightgbm_service.lightgbm_pred import LightGBM_forecast
from src.ts_models.xgboost_service.xgboost_pred import XGBoost_forecast
from src.ts_models.lr_service.lr_pred import LinearRegression_forecast
from src.ts_models.lstm_service.lstm_pred import LSTM_forecast
from src.ts_models.prophet_service.prophet_pred import Prophet_forecast
from src.ts_models.rf_service.rf_pred import RandomForest_forecast
from src.ts_models.svr_service.svr_pred import SVR_forecast
from src.ts_models.ARIMA_service.arima_pred import ARIMA_forecast
from src.ts_models.SARIMA_service.sarima_pred import SARIMA_forecast
from src.ts_models.PatchTST_service.PatchTST_pred import PatchTST_forecast
from src.ts_models.DLinear_service.DLinear_pred import DLinear_forecast
from src.ts_models.TCN_service.TCN_pred import TCN_forecast
from src.ts_models.Transformer_service.Transformer_pred import Transformer_forecast

time_series_models_funcs = {
    "LSTM": LSTM_forecast,
    "XGBoost": XGBoost_forecast,
    "CatBoost": CatBoost_forecast,
    "LightGBM": LightGBM_forecast,
    "LinearRegression": LinearRegression_forecast,
    "RandomForest": RandomForest_forecast,
    "SVR": SVR_forecast,
    "Prophet": Prophet_forecast,
    # "ARIMA": ARIMA_forecast,
    # "SARIMA": SARIMA_forecast,
    "PatchTST": PatchTST_forecast,
    "DLinear": DLinear_forecast,
    "TCN": TCN_forecast,
    "Transformer": Transformer_forecast,
    # "ARIMAX": ARIMAX_forecast,
}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing an experiment")

MESSAGES = {
    "en": {
        "init": "Initializing experiment design",
        "done": "Experiment design initialization completed. Total experiments: {}",
        "dataset_missing": "dataset_csv not found for dataset_name={}",
        "dataset_error": "Error processing dataset={}",
        "key_error": "Missing required key in dataset: {}",
        "df_error": "Error creating df_experiment_design DataFrame",
        "count_error": "Error computing value_counts for result_dir_name",
        "rename_error": "Error renaming result_dir_name",
        "critical": "Critical error during experiment design initialization: {}",
        "model_not_registered": (
            "The model '{}' added to model_to_test is not registered "
            "in time_series_models_funcs. Remove it or implement it. "
            "Available methods: {}. See more details in "
            "src/setups/experiment_setup.py"
        )
    },
    "ru": {
        "init": "Инициализация дизайна экспериментов",
        "done": "Инициализация experiment design завершена. Количество экспериментов: {}",
        "dataset_missing": "dataset_csv не найден для dataset_name={}",
        "dataset_error": "Ошибка при обработке dataset={}",
        "key_error": "В dataset отсутствует обязательный ключ: {}",
        "df_error": "Ошибка создания DataFrame df_experiment_design",
        "count_error": "Ошибка подсчета value_counts для result_dir_name",
        "rename_error": "Ошибка переименования result_dir_name",
        "critical": "Критическая ошибка при инициализации experiment design: {}",
        "model_not_registered": (
            "Модель '{}' добавленная в model_to_test не зарегистрирована "
            "в time_series_models_funcs. Удалите ее или реализуйте. "
            "Доступные методы: {}. Подробнее смотри в "
            "src/setups/experiment_setup.py"
        )
    },
    "zh": {
        "init": "实验设计初始化",
        "done": "实验设计初始化完成，总实验数量: {}",
        "dataset_missing": "未找到 dataset_csv: dataset_name={}",
        "dataset_error": "处理 dataset 时出错={}",
        "key_error": "dataset 缺少必要字段: {}",
        "df_error": "创建 df_experiment_design 失败",
        "count_error": "计算 result_dir_name 频率失败",
        "rename_error": "重命名 result_dir_name 失败",
        "critical": "实验设计初始化严重错误: {}",
        "model_not_registered": (
            "添加到 model_to_test 的模型 '{}' 未在 "
            "time_series_models_funcs 中注册。请删除或实现该模型。"
            "可用方法: {}。更多信息请查看 "
            "src/setups/experiment_setup.py"
        )
    }
}

# trajectory_cols = ["baseline", "calendar_components", "engineered_datetime_features", "mi_features", "chi_features", "pearson_features", "сlassic_GA", "optuna_features", "GA_horizon_selected_features",]

trajectory_cols = ["baseline", "engineered_datetime_features", "mi_features", "chi_features", "pearson_features", "сlassic_GA", "optuna_features", "GA_horizon_selected_features",]


model_to_test = list(time_series_models_funcs.keys())

# model_to_test = ["ARIMAX"]
# model_to_test = ["ARIMAX"]

# not_exogenous_models = ["ARIMA", "SARIMA"]

not_exogenous_models = [""]


datasets = [
    {
        "dataset_name": "morocco_zone_1",
        "type": "Energy",
        "col_time": "Datetime",
        "col_target": "consumption",
        "additional_cols": [],
        "start_train_date": "2017-01-01",
        "end_train_date": "2017-11-29",
        "start_test_date": "2017-11-30",
        "end_test_date": "2017-12-30",
        "predict_points": 300,
    },
    {
        "dataset_name": "russia_elista",
        "type": "Energy",
        "col_time": "datetime",
        "col_target": "value",
        "additional_cols": [],
        "start_train_date": "2017-12-01",
        "end_train_date": "2023-07-28",
        "start_test_date": "2023-07-29",
        "end_test_date": "2023-08-31",
        "predict_points": 300,
    },
    {
        "dataset_name": "Istanbul_Traffic_Index",
        "type": "Traffic",
        "col_time": "datetime",
        "col_target": "average_traffic_index",
        "additional_cols": [],
        "start_train_date": "2015-08-06",
        "end_train_date": "2024-07-02",
        "start_test_date": "2024-07-03",
        "end_test_date": "2024-09-03",
        "predict_points": 100,
    },
    {
        "dataset_name": "NYC_Taxi_Traffic",
        "type": "Traffic",
        "col_time": "timestamp",
        "col_target": "value",
        "additional_cols": [],
        "start_train_date": "2014-07-01",
        "end_train_date": "2015-01-01",
        "start_test_date": "2015-01-02",
        "end_test_date": "2015-01-21",
        "predict_points": 100,
    },
    {
        "dataset_name": "Air_Quality_India",
        "type": "Climate",
        "col_time": "Timestamp",
        "col_target": "PM2.5",
        "additional_cols": [],
        "start_train_date": "2017-11-07",
        "end_train_date": "2022-05-03",
        "start_test_date": "2022-05-04",
        "end_test_date": "2022-06-04",
        "predict_points": 300,
    },
    {
        "dataset_name": "Daily_Climate",
        "type": "Climate",
        "col_time": "date",
        "col_target": "meantemp",
        "additional_cols": [],
        "start_train_date": "2013-01-01",
        "end_train_date": "2016-09-01",
        "start_test_date": "2016-09-02",
        "end_test_date": "2017-01-01",
        "predict_points": 300,
    },

]


def validate_models(
        model_to_test,
        time_series_models_funcs,
        logger,
        logger_language="en"
):
    msg = MESSAGES.get(logger_language, MESSAGES["en"])

    available_models = list(time_series_models_funcs.keys())

    for model_name in model_to_test:
        if model_name not in time_series_models_funcs:
            error_message = msg["model_not_registered"].format(
                model_name,
                available_models
            )

            logger.error(error_message)

            raise ValueError(error_message)

"""
Registry of all available time series forecasting models.
Each key is a model name, and each value is the corresponding forecasting function.
All forecasting functions must implement the same interface:

def model_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger
)

Arguments:
- col_target: target column
- time_column: datetime column
- df_train: train dataframe
- df_test: test dataframe
- lag: lag size
- col_for_train: additional training features
- logger: logger object

To add a new model:
1. Add implementation into src/ts_models
2. Add the function into time_series_models_funcs
3. Add the model name into model_to_test

Example implementation:
src/ts_models/xgboost_service/xgboost_pred.py
"""



# BSTS (Bayesian Structural Time Series)
# Prophet                                   ✘ ✔
# LinearRegression                          ✔
# ARIMA                                     ✘ ✔
# SARIMA                                    ✘ ✔
# SVR                                       ✔
# RandomForestRegressor                     ✔
# XGBRegressor                              ✔
# LGBMRegressor                             ✔
# LSTM                                      ✔
# DeepAR                                    ✘
# TCN           
# TFT                                       ✘ ✔
# PatchTST                                  ✔
# NBEATS
# NHiTS                                     ✔
# TimesFM                                   ✘
# MambaTS
# TimeNet
# DLinear                                   ✔
# Transformer                               ✘ ✔


def load_and_prepare_progress(progress_csv_path, columns):
    if not os.path.exists(progress_csv_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(progress_csv_path, index=False)
        return df

    df = pd.read_csv(progress_csv_path)
    return df[columns]


def build_key(df, key_cols):
    tmp = df[key_cols].copy()

    for c in tmp.columns:
        if "date" in c or "time" in c:
            tmp[c] = pd.to_datetime(tmp[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            tmp[c] = tmp[c].astype(str).str.strip()

    tmp = tmp.fillna("NULL")

    return tmp.agg("|".join, axis=1)


# def get_pending_experiments(df_experiment_design, df_ready_progress):
#
#     ['id', 'model', 'trajectory_cols', 'dataset_name', 'type',
#     'dataset_csv', 'col_time', 'col_target', 'additional_cols',
#     'start_train_date', 'end_train_date', 'start_test_date',
#     'predict_points', 'end_test_date', 'result_dir_name']
#
#     print(f"df_experiment_design df_experiment_design df_experiment_design ")
#     print(df_experiment_design)
#     print(f"df_experiment_design columns = {df_experiment_design.columns}")
#     print(f"df_experiment_design len = {len(df_experiment_design)}")
#     print("=" * 100)
#
#     print(f"df_ready_progress df_ready_progress df_ready_progress ")
#     print(df_ready_progress)
#     print(f"df_ready_progress columns = {df_ready_progress.columns}")
#     print(f"df_ready_progress len = {len(df_ready_progress)}")
#     print("=" * 100)
#
#     df_design = df_experiment_design.copy()
#     key_cols = df_design.columns.tolist()
#
#     if df_ready_progress is None or df_ready_progress.empty:
#         logger.info(f"Total: {len(df_design)}")
#         logger.info("Processed: 0")
#         logger.info(f"To process: {len(df_design)}")
#         return df_design
#
#     df_ready = df_ready_progress.copy()
#
#     df_design["_key"] = build_key(df_design, key_cols)
#     df_ready["_key"] = build_key(df_ready, key_cols)
#
#     ready_keys = set(df_ready["_key"])
#
#     df_pending = df_design[~df_design["_key"].isin(ready_keys)].drop(columns=["_key"])
#
#     print(f"df_pending df_pending df_pending ")
#     print(df_pending)
#     print(f"df_pending columns = {df_pending.columns}")
#     print(f"df_pending len = {len(df_pending)}")
#     print("=" * 100)
#
#     logger.info(f"Total: {len(df_design)}")
#     logger.info(f"Processed: {len(df_ready)}")
#     logger.info(f"To process: {len(df_pending)}")
#
#     return df_pending

def normalize_list_columns(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = df[c].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    return df

def build_key(df, cols):
    return df[cols].astype(str).agg("|".join, axis=1)

def get_pending_experiments(df_experiment_design, df_ready_progress):
    required_cols = [
        'id', 'model', 'trajectory_cols', 'dataset_name', 'type',
        'dataset_csv', 'col_time', 'col_target', 'additional_cols',
        'start_train_date', 'end_train_date', 'start_test_date',
        'predict_points', 'end_test_date', 'result_dir_name'
    ]

    list_cols = ["trajectory_cols", "additional_cols"]

    df_design = df_experiment_design.copy()
    df_design = df_design[required_cols]
    df_design = normalize_list_columns(df_design, list_cols)
    df_design = df_design.drop_duplicates(subset=required_cols).reset_index(drop=True)

    if df_ready_progress is None or df_ready_progress.empty:
        return df_design[required_cols]

    df_ready = df_ready_progress.copy()
    df_ready = df_ready[required_cols]
    df_ready = normalize_list_columns(df_ready, list_cols)
    df_ready = df_ready.drop_duplicates(subset=required_cols).reset_index(drop=True)

    df_design["_key"] = build_key(df_design, required_cols)
    df_ready["_key"] = build_key(df_ready, required_cols)

    ready_keys = set(df_ready["_key"])
    df_pending = df_design[~df_design["_key"].isin(ready_keys)]

    df_pending = df_pending.drop(columns=["_key"]).reset_index(drop=True)

    return df_pending

def append_experiment_to_csv(experiment, progress_csv_path):
    df_row = pd.DataFrame([experiment])

    file_exists = os.path.exists(progress_csv_path)

    df_row.to_csv(
        progress_csv_path,
        mode="a",
        header=not file_exists,
        index=False
    )


def init_experiment_setup() -> pd.DataFrame:

    logger = logging.getLogger(__name__)
    msg = MESSAGES.get(logger_language, MESSAGES["en"])

    validate_models(
        model_to_test=model_to_test,
        time_series_models_funcs=time_series_models_funcs,
        logger=logger,
        logger_language=logger_language
    )

    try:
        logger.info(msg["init"])

        if not isinstance(datasets, list) or len(datasets) == 0:
            raise ValueError("datasets должен быть непустым списком")

        if not isinstance(datasets_csv_dict, dict) or len(datasets_csv_dict) == 0:
            raise ValueError("datasets_csv_dict должен быть непустым словарем")

        if not isinstance(model_to_test, list) or len(model_to_test) == 0:
            raise ValueError("model_to_test должен быть непустым списком")

        if not isinstance(trajectory_cols, list) or len(trajectory_cols) == 0:
            raise ValueError("trajectory_cols должен быть непустым списком")

        idx = 0
        rows = []

        for d in datasets:
            try:
                dataset_name = d["dataset_name"]
                dataset_csv = datasets_csv_dict.get(dataset_name)

                if dataset_csv is None:
                    logger.error(
                        msg["dataset_missing"].format(dataset_name)
                    )
                    raise ValueError(
                        msg["dataset_missing"].format(dataset_name)
                    )

                for m in model_to_test:
                    for t in trajectory_cols:
                        key = f"{dataset_name}_{m}_{t}"

                        rows.append({
                            "id": idx,
                            "model": m,
                            "trajectory_cols": t,
                            "dataset_name": dataset_name,
                            "type": d["type"],
                            "dataset_csv": dataset_csv,
                            "col_time": d["col_time"],
                            "col_target": d["col_target"],
                            "additional_cols": d["additional_cols"],
                            "start_train_date": d["start_train_date"],
                            "end_train_date": d["end_train_date"],
                            "start_test_date": d["start_test_date"],
                            "predict_points": d["predict_points"],
                            "end_test_date": d["end_test_date"],
                            "result_dir_name": key
                        })
                        idx += 1

            except KeyError as e:
                logger.exception(msg["key_error"].format(e))
                raise

            except Exception as e:
                logger.exception(msg["dataset_error"].format(d))
                raise

        try:
            df_experiment_design = pd.DataFrame(rows)
        except Exception:
            logger.exception(msg["df_error"])
            raise

        try:
            counts = df_experiment_design["result_dir_name"].value_counts()
        except Exception:
            logger.exception(msg["count_error"])
            raise

        counters = {}

        def fix_name(x):
            try:
                if counts[x] == 1:
                    return x

                i = counters.get(x, 0)
                counters[x] = i + 1

                return f"{i}_{x}"

            except Exception:
                logger.exception(f"Ошибка обработки result_dir_name={x}")
                raise

        try:
            df_experiment_design["result_dir_name"] = (
                df_experiment_design["result_dir_name"].map(fix_name)
            )
        except Exception:
            logger.exception(msg["rename_error"])
            raise

        logger.info(msg["done"].format(len(df_experiment_design)))

        return df_experiment_design

    except Exception as e:
        logger.exception(msg["critical"].format(e))
        raise


def append_progress_to_csv(progress_row, progress_csv_path):
    df_row = pd.DataFrame([progress_row])

    file_exists = os.path.exists(progress_csv_path)

    df_row.to_csv(
        progress_csv_path,
        mode="a",
        header=not file_exists,
        index=False
    )
