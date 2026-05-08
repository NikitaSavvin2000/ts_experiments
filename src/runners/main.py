import os
import sys
import yaml
import shutil
import logging
import pandas as pd
from collections import defaultdict



from config import experiment_output_dir
from src.utils.logger import get_logger
from src.data.data_config import datasets_csv_dict, datasets_col_mapping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Start")

home_path = os.getcwd()

export_path = os.path.join(home_path, experiment_output_dir)
os.makedirs(export_path, exist_ok=True)

params_path = os.path.join(home_path, "src", "runners", "params.yaml")

with open(params_path, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

print(params)
experiment_dir_name = params["experiment_unique_name"]



try:
    experiment_path = os.path.join(
        export_path,
        experiment_dir_name
    )

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
        logger.info(
            f"Папка эксперимента создана: {experiment_path}"
        )
    else:
        logger.info(
            f"Папка эксперимента уже существует: {experiment_path}"
        )

except Exception as e:
    logger.error(
        f"Ошибка при создании папки эксперимента: {e}"
    )
    raise

logger = get_logger(log_dir=experiment_path, name="experiment")

logger.info("Начало")


src_path = os.path.join(home_path, "src")
zip_path = os.path.join(experiment_path, "src_snapshot")

try:
    shutil.make_archive(
        zip_path,
        "zip",
        src_path
    )

    logger.info(
        f"Снимок src директории успешно создан: {zip_path}.zip"
    )

except Exception as e:
    logger.error(
        f"Ошибка при создании снимка src директории: {e}"
    )
    raise


trajectory_cols = ["baseline", "all_time_2_vec_col", "best_time_2_vec_col"]

model_to_test = ["LSTM", "XGBoost"]

datasets = [
    {
        "dataset_name": "morocco_zone_1",
        "start_train_date": "",
        "end_train_date": "",
        "start_test_date": "",
        "end_test_date": ""
    },
    {
        "dataset_name": "russia_elista",
        "start_train_date": "",
        "end_train_date": "",
        "start_test_date": "",
        "end_test_date": ""
    }
]

idx = 0
rows = []

for d in datasets:
    dataset_name = d["dataset_name"]

    dataset_csv = datasets_csv_dict.get(dataset_name)
    if dataset_csv is None:
        logger.error(f"dataset_csv не найден в datasets_csv_dict для dataset_name={dataset_name}")
        sys.exit(1)

    for m in model_to_test:
        for t in trajectory_cols:
            key = f"{dataset_name}_{m}_{t}"

            rows.append({
                "id": idx,
                "model": m,
                "trajectory_cols": t,
                "dataset_name": dataset_name,
                "dataset_csv": dataset_csv,
                "start_train_date": d["start_train_date"],
                "end_train_date": d["end_train_date"],
                "start_test_date": d["start_test_date"],
                "end_test_date": d["end_test_date"],
                "result_dir_name": key
            })

            idx += 1


df_experiment_design = pd.DataFrame(rows)

counts = df_experiment_design["result_dir_name"].value_counts()

counters = {}

def fix_name(x):
    if counts[x] == 1:
        return x
    i = counters.get(x, 0)
    counters[x] = i + 1
    return f"{i}_{x}"

df_experiment_design["result_dir_name"] = df_experiment_design["result_dir_name"].map(fix_name)

experiment_design_path =os.path.join(experiment_path,"experiment_design.csv")
df_experiment_design.to_csv(experiment_design_path)
logger.info(f"Дизайн экспериментов создан. Доступен по пути {experiment_design_path}")



