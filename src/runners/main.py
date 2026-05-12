import os
import sys
import pandas as pd

from src.configs.experiment_conf import init_experiment_config
from src.setups.experiment_setup import init_experiment_setup
from src.calendar_encoder.temporal_encoding import Time2Vec

from config import logger_language

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
        "experiment_error": "Error while creating experiment design: {}",
        "t2v_start": "Time2Vec encoding started",
        "t2v_success": "Time2Vec encoding finished successfully",
        "t2v_error": "Error during Time2Vec encoding: {}",
        "dataset_load_start": "Loading dataset: id={}, name={}",
        "dataset_load_success": "Dataset loaded successfully: id={}, name={}",
        "dataset_load_error": "Error loading dataset: id={}, name={}, path={}"
    },
    "ru": {
        "experiment_created": "Дизайн экспериментов создан. Доступен по пути {}",
        "experiment_error": "Ошибка при создании дизайна экспериментов: {}",
        "t2v_start": "Запуск Time2Vec encoding",
        "t2v_success": "Time2Vec encoding успешно завершён",
        "t2v_error": "Ошибка при Time2Vec encoding: {}",
        "dataset_load_start": "Загрузка датасета: id={}, имя={}",
        "dataset_load_success": "Датасет успешно загружен: id={}, имя={}",
        "dataset_load_error": "Ошибка загрузки датасета: id={}, имя={}, путь={}"
    },
    "zh": {
        "experiment_created": "实验设计已创建，路径：{}",
        "experiment_error": "创建实验设计时出错：{}",
        "t2v_start": "Time2Vec 编码开始",
        "t2v_success": "Time2Vec 编码成功完成",
        "t2v_error": "Time2Vec 编码错误：{}",
        "dataset_load_start": "正在加载数据集：id={}, 名称={}",
        "dataset_load_success": "数据集加载成功：id={}, 名称={}",
        "dataset_load_error": "数据集加载失败：id={}, 名称={}, 路径={}"
    }
}
msg = MESSAGES[logger_language]


# ============================================
# en: Experiment initialization (folders, logger, snapshot)
# ru: Инициализация эксперимента (папки, логгер, snapshot)
# zh: 实验初始化（文件夹、日志器、快照）
# ============================================
home_path, export_path, experiment_path, logger = init_experiment_config()

# ============================================
# en: Experiment design configuration initialization
# ru: Инициализация конфигурации дизайна экспериментов
# zh: 实验设计配置初始化
# ============================================
df_experiment_design = init_experiment_setup()

experiment_design_path =os.path.join(experiment_path,"experiment_design.csv")
df_experiment_design.to_csv(experiment_design_path)
logger.info(msg["experiment_created"].format(experiment_design_path))

# ============================================
# en: Experiment grid generation for models
# ru: Генерация сетки экспериментов для моделей
# zh: 模型实验网格生成
# ============================================
for _, experiment in df_experiment_design.iterrows():

    id = experiment["id"]
    model = experiment["model"]
    trajectory_cols = experiment["trajectory_cols"]
    dataset_name = experiment["dataset_name"]
    dataset_csv = experiment["dataset_csv"]
    col_time = experiment["col_time"]
    col_target = experiment["col_target"]
    start_train_date = experiment["start_train_date"]
    end_train_date = experiment["end_train_date"]
    start_test_date = experiment["start_test_date"]
    end_test_date = experiment["end_test_date"]
    result_dir_name = experiment["result_dir_name"]


    # ============================================
    # en: Dataset loading block
    # ru: Блок загрузки датасета
    # zh: 数据集加载模块
    # ============================================
    try:
        df_init = pd.read_csv(dataset_csv)
        logger.info(msg["dataset_load_success"].format(id, dataset_name))
    except Exception as e:
        logger.error(
            msg["dataset_load_error"].format(id, dataset_name, dataset_csv)
        )
        logger.error(str(e))
        raise

    # ============================================
    # en: Time2Vec encoder initialization block
    # ru: Блок инициализации энкодера Time2Vec
    # zh: Time2Vec 编码器初始化模块
    # ============================================
    logger.info(msg["t2v_start"])
    try:
        t2v = Time2Vec(col_time=col_time, col_target=col_target)
    except Exception as e:
        logger.error(msg["t2v_error"].format(str(e)))
        raise

    # ============================================
    # en: Time2Vec encoding execution block
    # ru: Блок создания эмбеддингов временного ряда
    # zh: 时间序列嵌入生成模块
    # ============================================
    try:
        df_t2v_embedding, min_val, max_val = t2v.encoder(df=df_init)
        logger.info(msg["t2v_success"])
    except Exception as e:
        logger.error(msg["t2v_error"].format(str(e)))
        raise

    print(df_t2v_embedding)








