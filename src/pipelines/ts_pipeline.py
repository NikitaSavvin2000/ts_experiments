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
                                                     calculate_test_points_predict)

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
        self.start_train_date = experiment["start_train_date"]
        self.end_train_date = experiment["end_train_date"]
        self.start_test_date = experiment["start_test_date"]
        self.end_test_date = experiment["end_test_date"]
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

        return self.stat_selected_features


    # def select_best_t2v_columns(self):
    #     global_best_cols = []
    #     global_best_score = float("inf")
    #
    #     best_metrics = None
    #     best_df_test_pred = None
    #
    #     for col in tqdm(self.all_available_cols):
    #         current_cols = global_best_cols + [col]
    #
    #         self.logger.info(f"TRY COLS: {current_cols}")
    #
    #         df_test_pred = self.forecast_func(
    #             col_target=self.col_target,
    #             time_column=self.col_time,
    #             df_train=self.df_train,
    #             df_test=self.df_test,
    #             lag=self.pacf_lag,
    #             col_for_train=current_cols,
    #             logger=self.logger,
    #         )
    #
    #         true = self.df_eval[self.col_target].tolist()
    #         pred = df_test_pred[self.col_target].tolist()
    #
    #         metrics = regression_metrics(true=true, pred=pred)
    #         score = metrics.get("mape", float("inf"))
    #
    #         self.logger.info(f"SCORE: {score} | GLOBAL BEST: {global_best_score}")
    #
    #         if score < global_best_score:
    #             global_best_score = score
    #             global_best_cols = current_cols
    #             best_metrics = metrics
    #             best_df_test_pred = df_test_pred
    #
    #     improved = True
    #     while improved and len(global_best_cols) > 1:
    #         improved = False
    #
    #         local_best_cols = global_best_cols
    #         local_best_score = global_best_score
    #
    #         for i in range(len(global_best_cols)):
    #             test_cols = global_best_cols[:i] + global_best_cols[i + 1:]
    #
    #             self.logger.info(f"BACKWARD TRY: {test_cols}")
    #
    #             df_test_pred = self.forecast_func(
    #                 col_target=self.col_target,
    #                 time_column=self.col_time,
    #                 df_train=self.df_train,
    #                 df_test=self.df_test,
    #                 lag=self.pacf_lag,
    #                 col_for_train=test_cols,
    #                 logger=self.logger,
    #             )
    #
    #             true = self.df_eval[self.col_target].tolist()
    #             pred = df_test_pred[self.col_target].tolist()
    #
    #             metrics = regression_metrics(true=true, pred=pred)
    #             score = metrics.get("mape", float("inf"))
    #
    #             self.logger.info(f"BACKWARD SCORE: {score} | GLOBAL BEST: {global_best_score}")
    #
    #             if score < local_best_score:
    #                 local_best_score = score
    #                 local_best_cols = test_cols
    #                 best_metrics = metrics
    #                 best_df_test_pred = df_test_pred
    #
    #         if local_best_score < global_best_score:
    #             global_best_score = local_best_score
    #             global_best_cols = local_best_cols
    #             improved = True
    #
    #     return global_best_cols, best_metrics, best_df_test_pred


    def plot_feature_correlation_analysis(self, threshold=0.1):

        corr_cols = self.all_available_cols + [self.col_target]

        df_corr = self.df_train[corr_cols].corr(method="pearson")

        fig_matrix = px.imshow(
            df_corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1
        )

        fig_matrix.update_layout(
            title="Correlation Matrix",
            width=1200,
            height=1200
        )

        fig_matrix.show()

        G = nx.Graph()

        for col in df_corr.columns:
            G.add_node(col)

        for i in range(len(df_corr.columns)):
            for j in range(i + 1, len(df_corr.columns)):

                corr_value = df_corr.iloc[i, j]

                if abs(corr_value) > threshold:
                    G.add_edge(
                        df_corr.columns[i],
                        df_corr.columns[j]
                    )

        pos = nx.spring_layout(G, seed=42)

        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            hoverinfo="none"
        )

        node_x = []
        node_y = []
        node_text = []

        for node in G.nodes():
            x, y = pos[node]

            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hoverinfo="text",
            marker=dict(size=12)
        )

        fig_graph = go.Figure(
            data=[edge_trace, node_trace]
        )

        fig_graph.update_layout(
            title=f"Feature Correlation Graph | Threshold = {threshold}",
            showlegend=False,
            width=1400,
            height=1000
        )

        fig_graph.show()


    def select_best_t2v_columns(self, population_size=15, generations=1, mutation_rate=0.15, elite_size=5):

        all_features = self.all_available_cols.copy()

        def run_model(cols):
            df_test_pred = self.forecast_func(
                col_target=self.col_target,
                time_column=self.col_time,
                df_train=self.df_train,
                df_test=self.df_test,
                lag=self.pacf_lag,
                col_for_train=cols,
                logger=self.logger,
            )

            true = self.df_eval[self.col_target].tolist()
            pred = df_test_pred[self.col_target].tolist()

            metrics = regression_metrics(true=true, pred=pred)
            score = metrics.get("mape", float("inf"))

            return score, metrics, df_test_pred

        def init_population():
            pop = []
            for _ in range(population_size):
                ind = [f for f in all_features if random.random() > 0.5]
                if len(ind) == 0:
                    ind = random.sample(all_features, 1)
                pop.append(ind)
            return pop

        def crossover(p1, p2):
            cut = random.randint(1, max(1, min(len(p1), len(p2)) - 1))
            child = list(set(p1[:cut] + p2[cut:]))
            if len(child) == 0:
                child = random.sample(all_features, 1)
            return child

        def mutate(ind):
            ind = ind.copy()

            if random.random() < mutation_rate:
                if random.random() < 0.5 and len(ind) > 1:
                    ind.remove(random.choice(ind))
                else:
                    f = random.choice(all_features)
                    if f not in ind:
                        ind.append(f)

            if len(ind) == 0:
                ind = random.sample(all_features, 1)

            return ind

        population = init_population()

        best_individual = None
        best_score = float("inf")
        best_metrics = None
        best_pred = None

        no_improve = 0

        for gen in range(generations):

            self.logger.info(f"GEN {gen}")

            scored = []

            for ind in population:
                score, metrics, pred = run_model(ind)

                self.logger.info(f"GEN {gen} SIZE={len(ind)} MAPE={score}")

                scored.append((ind, score, metrics, pred))

            scored.sort(key=lambda x: x[1])

            current_best_score = scored[0][1]

            if current_best_score < best_score:
                best_individual = scored[0][0]
                best_score = current_best_score
                best_metrics = scored[0][2]
                best_pred = scored[0][3]
                no_improve = 0
            else:
                no_improve += 1

            self.logger.info(f"BEST GEN {gen} MAPE = {best_score}")

            if no_improve > 0:
                self.logger.info("EARLY STOP: MIN NOT IMPROVED")
                break

            elite = [x[0] for x in scored[:elite_size]]

            new_population = elite.copy()

            while len(new_population) < population_size:
                p1 = random.choice(elite)
                p2 = random.choice(elite)

                child = crossover(p1, p2)
                child = mutate(child)

                new_population.append(child)

            population = new_population

        final_cols = best_individual
        final_score, final_metrics, final_pred = run_model(final_cols)

        self.logger.info(f"FINAL COLS = {final_cols}")
        self.logger.info(f"FINAL MAPE = {final_score}")

        return final_cols, final_metrics, final_pred


    def run_test_predict(self):
        """
        RU: Статистический отбор признаков для временного ряда
        EN: Statistical feature selection for time series
        ZH: 时间序列统计特征选择
        """
        try:

            if self.trajectory_cols == "calendar_components":
                self.col_for_train = self.calendar_components_cols
            elif self.trajectory_cols == "baseline":
                self.col_for_train = []

                self.test_points = calculate_test_points_predict(
                    start_test_date=self.start_test_date,
                    end_test_date=self.end_test_date,
                    discreteness=self.discreteness_sec)

                self.max_lag = self.test_points


            elif self.trajectory_cols == "stat_selected_features":
                self.col_for_train = self.fetch_stat_select_features()
            elif self.trajectory_cols in ["engineered_datetime_features", "horizon_selected_features"]:
                self.col_for_train = self.fetch_all_t2v_features()
            else:
                raise ValueError("Non-existent experiment trajectory. Please implement the logic for it or remove it from src/setups/experiment_setup.py")

            self.forecast_func = time_series_models_funcs[self.model]


            self.pacf_lag = select_pacf_lag(
                df=self.df_t2v,
                col_target=self.col_target,
                col_time=self.col_time,
                # max_lag=self.max_lag,
                logger=self.logger
            )

            if self.trajectory_cols == "horizon_selected_features":
                self.all_available_cols = self.col_for_train
                self.col_for_train, self.metrix_dict, self.df_test_pred = self.select_best_t2v_columns()
            else:
                self.df_test_pred = self.forecast_func (
                    col_target=self.col_target,
                    time_column=self.col_time,
                    df_train=self.df_train,
                    df_test=self.df_test,
                    lag=self.pacf_lag,
                    col_for_train=self.col_for_train,
                    logger=self.logger,
                )

                self.true =self.df_eval[self.col_target].tolist()
                self.pred = self.df_test_pred[self.col_target].tolist()

                self.metrix_dict = regression_metrics(true=self.true, pred=self.pred)


            self.df_test_pred = self.df_test_pred.merge(
                self.df_eval[[self.col_time, self.col_target]].rename(columns={self.col_target: "true"}),
                on=self.col_time,
                how="left"
            )

            self.df_test_pred_not_norm = self.df_test_pred.copy()

            self.df_test_pred_not_norm[self.col_target] = (
                    self.df_test_pred_not_norm[self.col_target] * (self.max_val - self.min_val) + self.min_val
            )

            self.df_test_pred_not_norm["true"] = (
                    self.df_test_pred_not_norm["true"] * (self.max_val - self.min_val) + self.min_val
            )

            self.df_test_pred = self.df_test_pred.rename(columns={self.col_time: "datetime", self.col_target: "pred"})

            self.df_test_pred_not_norm = self.df_test_pred_not_norm.rename(columns={self.col_time: "datetime", self.col_target: "pred"})


            self.logger.info(self.msg["stat_select_done"])

        except Exception as e:
            self.logger.error(self.msg["stat_select_error"].format(str(e)))
            raise e

        return self

