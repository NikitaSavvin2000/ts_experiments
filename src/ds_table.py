"""
pdm run src/ds_table.py
"""

import pandas as pd
from src.setups.experiment_setup import datasets
from src.configs.data_config import datasets_csv_dict
from src.ts_models.ts_utils.timeseries_utils import calculate_discreteness_interval

csv_Metro_Traffic = datasets_csv_dict["Metro_Traffic"]

df_Metro_Traffic = pd.read_csv(csv_Metro_Traffic)

print(df_Metro_Traffic)


ru_article_datasets = ["russia_elista", "Weather", "Metro_Traffic"]

datasets = [d for d in datasets if d["dataset_name"] in ru_article_datasets]

rows_ru = []
rows_en = []
for dataset in datasets:
    type = dataset["type"]
    dataset_name = dataset["dataset_name"]
    csv = datasets_csv_dict[dataset_name]
    col_time = dataset["col_time"]
    col_target = dataset["col_target"]

    df = pd.read_csv(csv)
    df = df[[col_time, col_target]]
    discrentess_min = calculate_discreteness_interval(df=df, time_column=col_time) / 60
    df[col_time] = pd.to_datetime(df[col_time])

    min_date = df[col_time].min()
    max_date = df[col_time].max()
    total_points = len(df)

    row_ru = {
        "Датасет": dataset_name,
        "Тип": type,
        "Количество точек": total_points,
        "Начальная дата": min_date,
        "Конечная дата": max_date,
        "Дискретность (мин.)": discrentess_min,
    }

    row_en = {
        "Dataset": dataset_name,
        "Type": type,
        "Number of points": total_points,
        "Start date": min_date,
        "End date": max_date,
        "Frequency (min)": discrentess_min,
    }

    rows_ru.append(row_ru)
    rows_en.append(row_en)

df_ru = pd.DataFrame(rows_ru)
df_en = pd.DataFrame(rows_en)

path_ru = "export/dataset_info_ru.csv"
path_en = "export/dataset_info_en.csv"

df_ru.to_csv(path_ru)
df_en.to_csv(path_en)




