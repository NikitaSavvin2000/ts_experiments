import os
import sys
import pandas as pd

from src.configs.experiment_conf import init_experiment_config
from src.setups.experiment_setup import init_experiment_setup

from config import logger_language
from src.pipelines.ts_pipeline import TSExperimentPipeline

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

    # ============================================
    # en: Initialize experiment pipeline instance
    # ru: Инициализация экземпляра пайплайна эксперимента
    # zh: 初始化实验管道实例
    # ============================================
    ts_pipeline = TSExperimentPipeline(
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
    ts_pipeline.init_design(df_experiment_design)
    
    # ============================================
    # en: Load dataset for current experiment
    # ru: Загрузка датасета для текущего эксперимента
    # zh: 加载当前实验的数据集
    # ============================================
    ts_pipeline.load_dataset()
    
    # ============================================
    # en: Apply Time2Vec temporal encoding
    # ru: Применение временного кодирования Time2Vec
    # zh: 应用 Time2Vec 时间编码
    # ============================================
    ts_pipeline.run_time2vec()
    
    # ============================================
    # en: Select optimal lag using PACF
    # ru: Выбор оптимального лага с помощью PACF
    # zh: 使用 PACF 选择最优滞后
    # ============================================
    ts_pipeline.run_lag_pacf()
    
    # ============================================
    # en: Split dataset into train and test sets
    # ru: Разделение данных на train и test выборки
    # zh: 将数据划分为训练集和测试集
    # ============================================
    ts_pipeline.run_split()

    # ============================================
    # en: Statistical feature selection for forecasting
    # ru: Статистический отбор признаков для прогнозирования
    # zh: 预测任务的统计特征选择
    # ============================================
    ts_pipeline.fetch_stat_select_features()



