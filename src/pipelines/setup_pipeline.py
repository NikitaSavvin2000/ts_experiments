import os
import pandas as pd

from src.calendar_encoder.temporal_encoding import Time2Vec
from src.ts_models.pacf_lag_selection import select_pacf_lag
from src.ts_models.time_series_split import split_train_test
from src.ts_models.feature_selection import stat_select_features
from src.setups.experiment_setup import time_series_models_funcs, not_exogenous_models
from src.ts_models.ts_utils.timeseries_utils import (regression_metrics,
                                                     calculate_discreteness_interval,
                                                     generate_time_series_df,
                                                     test_method_visualize,
                                                     calculate_test_points_predict,
                                                     assign_end_train_start_test_date)

from src.ts_models.grids import models_grids, models_easy

import optuna
import numpy as np
import itertools

from tqdm import tqdm
from itertools import combinations
from tqdm import tqdm
import itertools
from itertools import combinations

import plotly.express as px

import random

import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import random
import random
import random
import numpy as np
from sklearn.feature_selection import mutual_info_regression


MESSAGES = {
    "en": {
        "experiment_created": "Experiment design created. Available at {}",
        "experiment_exists": "Experiment design already exists and will not be overwritten: {}",
        "dataset_load_success": "Dataset loaded: id={}, name={}",
        "dataset_load_error": "Dataset load error: id={}, name={}, path={}",
        "t2v_start": "Time2Vec start",
        "t2v_success": "Time2Vec success",
        "t2v_error": "Time2Vec error: {}",
        "pacf_error": "PACF error: {}",
        "split_error": "Split error: {}",
        "stat_select_start": "Statistical feature selection start",
        "stat_select_done": "Statistical feature selection completed",
        "stat_select_error": "Statistical feature selection error: {}",
    },
    "ru": {
        "experiment_created": "Дизайн экспериментов создан. Путь: {}",
        "experiment_exists": "Дизайн экспериментов уже существует и не будет перезаписан: {}",
        "dataset_load_success": "Датасет загружен: id={}, имя={}",
        "dataset_load_error": "Ошибка загрузки: id={}, имя={}, путь={}",
        "t2v_start": "Time2Vec старт",
        "t2v_success": "Time2Vec готов",
        "t2v_error": "Time2Vec ошибка: {}",
        "pacf_error": "PACF ошибка: {}",
        "split_error": "Split ошибка: {}",
        "stat_select_start": "Статистический отбор признаков старт",
        "stat_select_done": "Статистический отбор признаков завершён",
        "stat_select_error": "Ошибка статистического отбора признаков: {}",
    },
    "zh": {
        "experiment_created": "实验设计已创建：{}",
        "experiment_exists": "实验设计已存在，不会被覆盖: {}",
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


class SetupModel:

    def __init__(
            self,
            experiment,
            home_path,
            export_path,
            experiment_path,
            logger,
            messages,
            logger_language,
            test_points,
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
        self.test_points=test_points


        self.set_row(experiment)

        self.df_experiment_design = None

        self.df_init = None
        self.df_t2v = None
        self.df_train = None
        self.df_test = None
        self.pacf_lag = 1
        self.calendar_components_cols = ["year", "month", "day", "hour", "minute", "second"]
        self.time_series_models_funcs = time_series_models_funcs
        self.max_lag = 50

    def init_design(self, df_experiment_design):
        """
        RU: Сохранение дизайна экспериментов
        EN: Save experiment design
        ZH: 保存实验设计
        """
        self.df_experiment_design = df_experiment_design

        path = os.path.join(
            self.experiment_path,
            "experiment_design.csv"
        )

        if os.path.exists(path):
            self.logger.info(
                self.msg["experiment_exists"].format(path)
            )
            return self

        self.df_experiment_design.to_csv(path, index=False)

        self.logger.info(
            self.msg["experiment_created"].format(path)
        )

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
        self.result_dir_name = experiment["result_dir_name"]
        self.predict_points = experiment["predict_points"]
        self.baseline_cols = [self.col_time, self.col_target]
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

            self.last_known_data = pd.to_datetime(self.df_init[self.col_time]).max()
            self.first_known_data = pd.to_datetime(self.df_init[self.col_time]).min()
            self.discreteness_sec = calculate_discreteness_interval(df=self.df_init, time_column=self.col_time)

            self.start_train_date = self.first_known_data

            self.end_train_date, self.start_test_date = assign_end_train_start_test_date(
                df=self.df_init,
                col_time=self.col_time,
                test_points=self.test_points,
            )

            self.end_test_date = self.last_known_data

            self.logger.info(self.msg["dataset_load_success"].format(self.id, self.dataset_name))
        except Exception as e:
            self.logger.error(self.msg["dataset_load_error"].format(self.id, self.dataset_name, self.dataset_csv))
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

            self.df_eval = self.df_test.copy()
            self.df_eval = self.df_eval[[self.col_time, self.col_target]]
            self.df_test[self.col_target] = None

        except Exception as e:
            self.logger.error(self.msg["split_error"].format(str(e)))
            raise e

        return self


    def prepare_future_dataframe(self):
        """
        RU: Time2Vec кодирование временного ряда
        EN: Time2Vec encoding
        ZH: Time2Vec 编码
        """
        self.logger.info(self.msg["t2v_start"])

        try:
            self.last_known_data = pd.to_datetime(self.df_init[self.col_time]).max()
            self.discreteness_sec = calculate_discreteness_interval(df=self.df_init, time_column=self.col_time)

            self.df_real_pred = generate_time_series_df(
                start_date=self.last_known_data,
                n_rows=self.predict_points,
                freq_seconds=self.discreteness_sec ,
                col_time=self.col_time,
                col_target=self.col_target,

            )
        except Exception as e:
            self.logger.error(self.msg["t2v_error"].format(str(e)))
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
            self.df_real_pred_t2v, _, _ = t2v.encoder(df=self.df_real_pred)

            self.logger.info(self.msg["t2v_success"])
        except Exception as e:
            self.logger.error(self.msg["t2v_error"].format(str(e)))
            raise e

        return self

    def fetch_all_t2v_features(self):

        excluded = [self.col_time, self.col_target]

        self.all_t2v_cols = [
            col for col in self.df_t2v.columns
            if col not in excluded
        ]

        return self.all_t2v_cols


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


    # def run_setup_lag_by_model(self):
    #     try:
    #         if self.trajectory_cols == "baseline":
    #             self.col_for_train = []
    #         else:
    #             raise ValueError("Non-existent experiment trajectory. Please implement the logic for it or remove it from src/setups/experiment_setup.py")
    #
    #         self.forecast_func = time_series_models_funcs[self.model]
    #         params = models_easy[self.model]
    #
    #         best_score = float("inf")
    #         best_lag = None
    #         best_pred = None
    #
    #         for lag in range(1, 36):
    #             df_pred = self.forecast_func(
    #                 col_target=self.col_target,
    #                 time_column=self.col_time,
    #                 df_train=self.df_train,
    #                 df_test=self.df_test,
    #                 lag=lag,
    #                 col_for_train=self.col_for_train,
    #                 logger=self.logger,
    #                 params=params
    #             )
    #
    #             print(df_pred)
    #
    #             true = self.df_eval[self.col_target].tolist()
    #             pred = df_pred[self.col_target].tolist()
    #
    #             metrics = regression_metrics(true=true, pred=pred)
    #
    #             print(metrics)
    #
    #             r2 = metrics.get("r2", 0)
    #             mape = metrics.get("mape", 0)
    #             bp = metrics.get("bp", 0)
    #
    #             score = (1 - r2) + mape + bp
    #
    #             if score < best_score:
    #                 best_score = score
    #                 best_lag = lag
    #                 best_pred = pred
    #
    #         self.best_lag = best_lag
    #         self.best_score = best_score
    #         self.best_pred = best_pred
    #
    #         return self
    #
    #     except Exception as e:
    #         self.logger.error(self.msg["stat_select_error"].format(str(e)))
    #         raise e


    def run_setup_lag_by_model(self):
        try:

            self.pacf_lag = select_pacf_lag(
                df=self.df_t2v,
                col_target=self.col_target,
                col_time=self.col_time,
                # max_lag=self.max_lag,
                logger=self.logger
            )

            self.best_lag = self.pacf_lag

            return self.best_lag

        except Exception as e:
                self.logger.error(self.msg["stat_select_error"].format(str(e)))
                raise e

    def run_setup_models_params(self):
        print(f" >>>>>>>> MODEL {self.model}")

        try:

            if self.trajectory_cols == "baseline":
                self.col_for_train = ["year", "month", "day", "hour", "minute", "second"]
            else:
                raise ValueError(
                    "Non-existent experiment trajectory. Please implement the logic for it or remove it from src/setups/experiment_setup.py"
                )

            self.forecast_func = time_series_models_funcs[self.model]
            self.model_grid = models_grids[self.model]

            best_score = float("-inf")
            best_params = None
            best_pred = None
            best_metrics = {}

            keys = list(self.model_grid.keys())
            values = list(self.model_grid.values())

            true = self.df_eval[self.col_target].tolist()

            for combo in tqdm(itertools.product(*values)):
                params = dict(zip(keys, combo))

                metrics = None
                try:
                    df_pred = self.forecast_func(
                        col_target=self.col_target,
                        time_column=self.col_time,
                        df_train=self.df_train,
                        df_test=self.df_test,
                        lag=self.pacf_lag,
                        col_for_train=self.col_for_train,
                        logger=self.logger,
                        params=params
                    )

                    pred = df_pred[self.col_target].tolist()

                    if (
                            len(pred) > 0
                            and len(pred) == len(true)
                            and not any(pd.isna(x) for x in pred)
                    ):
                        metrics = regression_metrics(true=true, pred=pred)
                        r2 = metrics.get("r2", float("-inf"))

                        if np.isfinite(r2):
                            score = r2
                        else:
                            score = float("-inf")
                    else:
                        score = float("-inf")

                    print("=" * 120)
                    print(f" >>>>>>>> MODEL {self.model}")
                    print(f" >>>>>>>> PARAMS {params}")
                    print(f" >>>>>>>> METRICS {metrics}")
                    print("=" * 120)

                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_pred = pred
                        best_metrics = metrics or {}

                except Exception:
                    score = float("-inf")

            self.best_params = best_params
            self.best_score_params = best_score
            self.best_pred = best_pred
            self.best_metrics = best_metrics

            print(" >>>>>>>>>> END <<<<<<<<<<<<<<<")

            print(self.best_params)
            print(self.best_score_params)
            # print(self.best_pred)

        except Exception as e:
            self.logger.error(self.msg["stat_select_error"].format(str(e)))
            raise e

        return self


    #
    # def objective(self, trial):
    #
    #     try:
    #
    #         params = {}
    #
    #         for param_name, param_values in self.model_grid.items():
    #
    #             first_value = param_values[0]
    #
    #             if isinstance(first_value, int):
    #
    #                 params[param_name] = trial.suggest_int(
    #                     param_name,
    #                     min(param_values),
    #                     max(param_values)
    #                 )
    #
    #             elif isinstance(first_value, float):
    #
    #                 params[param_name] = trial.suggest_float(
    #                     param_name,
    #                     min(param_values),
    #                     max(param_values)
    #                 )
    #
    #             else:
    #
    #                 params[param_name] = trial.suggest_categorical(
    #                     param_name,
    #                     param_values
    #                 )
    #
    #         df_pred = self.forecast_func(
    #             col_target=self.col_target,
    #             time_column=self.col_time,
    #             df_train=self.df_train,
    #             df_test=self.df_test,
    #             lag=self.pacf_lag,
    #             col_for_train=self.col_for_train,
    #             logger=self.logger,
    #             params=params
    #         )
    #
    #         true = self.df_eval[self.col_target].tolist()
    #         pred = df_pred[self.col_target].tolist()
    #
    #         metrics = regression_metrics(
    #             true=true,
    #             pred=pred
    #         )
    #
    #         print("="*100)
    #         print(metrics)
    #         print("="*100)
    #
    #
    #         r2 = metrics.get("r2", 0)
    #         mape = metrics.get("mape", 0)
    #         bp = metrics.get("bp", 0)
    #
    #         if (
    #                 np.isnan(r2)
    #                 or np.isinf(r2)
    #                 or np.isnan(mape)
    #                 or np.isinf(mape)
    #                 or np.isnan(bp)
    #                 or np.isinf(bp)
    #         ):
    #             raise optuna.TrialPruned()
    #
    #         if mape > 1e6:
    #             raise optuna.TrialPruned()
    #
    #         score = (1 - r2) + mape + bp
    #
    #         trial.set_user_attr("metrics", metrics)
    #         trial.set_user_attr("pred", pred)
    #
    #         return score
    #
    #     except Exception:
    #         raise optuna.TrialPruned()

    # def run_setup_models_params(self):
    #
    #     try:
    #
    #         if self.trajectory_cols == "baseline":
    #             self.col_for_train = []
    #         else:
    #             raise ValueError(
    #                 "Non-existent experiment trajectory."
    #             )
    #
    #         self.forecast_func = time_series_models_funcs[self.model]
    #
    #         self.model_grid = models_grids[self.model]
    #
    #         study = optuna.create_study(
    #             direction="minimize"
    #         )
    #
    #         study.optimize(
    #             lambda trial: self.objective(trial),
    #             n_trials=30,
    #             n_jobs=1
    #         )
    #
    #         best_trial = study.best_trial
    #
    #         self.best_params = best_trial.params
    #         self.best_score_params = best_trial.value
    #         self.best_metrics = best_trial.user_attrs["metrics"]
    #         self.best_pred = best_trial.user_attrs["pred"]
    #
    #         print(self.best_params)
    #         print(self.best_score_params)
    #         print(self.best_metrics)
    #
    #     except Exception as e:
    #
    #         self.logger.error(
    #             self.msg["stat_select_error"].format(str(e))
    #         )
    #
    #         raise e
    #
    #     return self

