import os
import pandas as pd

from src.calendar_encoder.temporal_encoding import Time2Vec
from src.ts_models.pacf_lag_selection import select_pacf_lag
from src.ts_models.time_series_split import split_train_test
from src.ts_models.feature_selection import stat_select_features

from config import logger_language


MESSAGES = {
    "en": {
        "experiment_created": "Experiment design created. Available at {}",
        "dataset_load_success": "Dataset loaded: id={}, name={}",
        "dataset_load_error": "Dataset load error: id={}, name={}, path={}",
        "t2v_start": "Time2Vec start",
        "t2v_success": "Time2Vec success",
        "t2v_error": "Time2Vec error: {}",
        "pacf_error": "PACF error: {}",
        "split_error": "Split error: {}",
        "stat_select_start": "Statistical feature selection start",
        "stat_select_done": "Statistical feature selection completed",
        "stat_select_error": "Statistical feature selection error: {}"
    },
    "ru": {
        "experiment_created": "Дизайн экспериментов создан. Путь: {}",
        "dataset_load_success": "Датасет загружен: id={}, имя={}",
        "dataset_load_error": "Ошибка загрузки: id={}, имя={}, путь={}",
        "t2v_start": "Time2Vec старт",
        "t2v_success": "Time2Vec готов",
        "t2v_error": "Time2Vec ошибка: {}",
        "pacf_error": "PACF ошибка: {}",
        "split_error": "Split ошибка: {}",
        "stat_select_start": "Статистический отбор признаков старт",
        "stat_select_done": "Статистический отбор признаков завершён",
        "stat_select_error": "Ошибка статистического отбора признаков: {}"
    },
    "zh": {
        "experiment_created": "实验设计已创建：{}",
        "dataset_load_success": "数据集加载成功：id={}, 名称={}",
        "dataset_load_error": "数据集加载失败：id={}, 名称={}, 路径={}",
        "t2v_start": "Time2Vec 开始",
        "t2v_success": "Time2Vec 完成",
        "t2v_error": "Time2Vec 错误：{}",
        "pacf_error": "PACF 错误：{}",
        "split_error": "split 错误：{}",
        "stat_select_start": "统计特征选择开始",
        "stat_select_done": "统计特征选择完成",
        "stat_select_error": "统计特征选择错误：{}"
    }
}


class TSExperimentPipeline:

    def __init__(
            self,
            experiment,
            home_path,
            export_path,
            experiment_path,
            logger,
            messages,
            logger_language
    ):
        """
        RU: Инициализация пайплайна эксперимента (без инфраструктуры)
        EN: Initialize experiment pipeline without infrastructure layer
        ZH: 初始化实验管道（不包含基础设施层）
        """

        self.home_path = home_path
        self.export_path = export_path
        self.experiment_path = experiment_path
        self.logger = logger

        self.msg = MESSAGES[logger_language]

        self.set_row(experiment)

        self.df_experiment_design = None

        self.df_init = None
        self.df_t2v = None
        self.df_train = None
        self.df_test = None
        self.pacf_lag = 1
        self.baseline_cols = ["year", "month", "day", "hour", "minute", "second"]

    def init_design(self, df_experiment_design):
        """
        RU: Сохранение дизайна экспериментов
        EN: Save experiment design
        ZH: 保存实验设计
        """
        self.df_experiment_design = df_experiment_design

        path = os.path.join(self.experiment_path, "experiment_design.csv")
        self.df_experiment_design.to_csv(path, index=False)

        self.logger.info(self.msg["experiment_created"].format(path))
        return self

    def set_row(self, experiment):
        """
        RU: Установка параметров эксперимента
        EN: Set experiment parameters
        ZH: 设置实验参数
        """
        self.id = experiment["id"]
        self.model = experiment["model"]
        self.trajectory_cols = experiment["trajectory_cols"]
        self.dataset_name = experiment["dataset_name"]
        self.dataset_csv = experiment["dataset_csv"]
        self.col_time = experiment["col_time"]
        self.col_target = experiment["col_target"]
        self.additional_cols = experiment["additional_cols"]
        self.start_train_date = experiment["start_train_date"]
        self.end_train_date = experiment["end_train_date"]
        self.start_test_date = experiment["start_test_date"]
        self.end_test_date = experiment["end_test_date"]
        self.result_dir_name = experiment["result_dir_name"]

        return self

    def load_dataset(self):
        """
        RU: Загрузка датасета
        EN: Load dataset
        ZH: 加载数据集
        """
        try:
            self.cols_to_select = [self.col_time, self.col_target] + self.additional_cols
            self.df_init = pd.read_csv(self.dataset_csv)
            self.existing_cols = [c for c in self.cols_to_select if c in self.df_init.columns]
            self.df_init = self.df_init[self.existing_cols]

            self.logger.info(self.msg["dataset_load_success"].format(self.id, self.dataset_name))
        except Exception as e:
            self.logger.error(self.msg["dataset_load_error"].format(self.id, self.dataset_name, self.dataset_csv))
            raise e

        return self

    def run_time2vec(self):
        """
        RU: Time2Vec кодирование временного ряда
        EN: Time2Vec encoding
        ZH: Time2Vec 编码
        """
        self.logger.info(self.msg["t2v_start"])

        try:
            t2v = Time2Vec(col_time=self.col_time, col_target=self.col_target)
            self.df_t2v, self.min_val, self.max_val = t2v.encoder(df=self.df_init)
            self.logger.info(self.msg["t2v_success"])
        except Exception as e:
            self.logger.error(self.msg["t2v_error"].format(str(e)))
            raise e

        return self

    def run_lag_pacf(self):
        """
        RU: Выбор лага через PACF
        EN: PACF lag selection
        ZH: PACF 滞后选择
        """
        try:
            self.pacf_lag = select_pacf_lag(
                df=self.df_t2v,
                col_target=self.col_target,
                col_time=self.col_time,
                logger=self.logger
            )
        except Exception as e:
            self.logger.error(self.msg["pacf_error"].format(str(e)))
            raise e

        return self

    def run_split(self):
        """
        RU: Разделение train/test
        EN: Train/test split
        ZH: 训练/测试划分
        """
        try:
            self.df_train, self.df_test = split_train_test(
                df=self.df_t2v,
                start_train_date=self.start_train_date,
                end_train_date=self.end_train_date,
                start_test_date=self.start_test_date,
                end_test_date=self.end_test_date,
                col_time=self.col_time,
                logger=self.logger
            )
        except Exception as e:
            self.logger.error(self.msg["split_error"].format(str(e)))
            raise e

        return self


    def fetch_stat_select_features(self):
        """
        RU: Статистический отбор признаков для временного ряда
        EN: Statistical feature selection for time series
        ZH: 时间序列统计特征选择
        """
        try:

            self.stat_selected_features = stat_select_features(
                df=self.df_t2v,
                col_time=self.col_time,
                col_target=self.col_target,
                logger=self.logger
            )

            self.logger.info(self.msg["stat_select_done"])

        except Exception as e:
            self.logger.error(self.msg["stat_select_error"].format(str(e)))
            raise e

        return self




