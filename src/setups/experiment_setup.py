import logging
import pandas as pd

from src.configs.data_config import datasets_csv_dict
from config import logger_language

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
        "critical": "Critical error during experiment design initialization: {}"
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
        "critical": "Критическая ошибка при инициализации experiment design: {}"
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
        "critical": "实验设计初始化严重错误: {}"
    }
}

trajectory_cols = ["baseline", "all_time_2_vec_col", "stat_feature_filter", "best_time_2_vec_col"]

model_to_test = ["LSTM", "XGBoost"]

datasets = [
    {
        "dataset_name": "morocco_zone_1",
        "col_time": "Datetime",
        "col_target": "consumption",
        "additional_cols": [],
        "start_train_date": "2017-01-01",
        "end_train_date": "2017-11-29",
        "start_test_date": "2017-11-30",
        "end_test_date": "2017-12-30"
    },
    # {
    #     "dataset_name": "russia_elista",
    #     "col_time": "",
    #     "col_target": "",
    #     "start_train_date": "",
    #     "end_train_date": "",
    #     "start_test_date": "",
    #     "end_test_date": ""
    # }
]

def init_experiment_setup() -> pd.DataFrame:

    logger = logging.getLogger(__name__)
    msg = MESSAGES.get(logger_language, MESSAGES["en"])

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
                            "dataset_csv": dataset_csv,
                            "col_time": d["col_time"],
                            "col_target": d["col_target"],
                            "additional_cols": d["additional_cols"],
                            "start_train_date": d["start_train_date"],
                            "end_train_date": d["end_train_date"],
                            "start_test_date": d["start_test_date"],
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