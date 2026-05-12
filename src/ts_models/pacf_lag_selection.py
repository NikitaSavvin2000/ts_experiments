import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.stattools import pacf
from config import logger_language


MESSAGES = {
    "en": {
        "init": "Initializing PACF lag selection",
        "no_numeric": "No numeric columns available for PACF computation",
        "pacf_error": "Error computing PACF for column={}",
        "no_valid": "No valid PACF computations, fallback lag=1",
        "done": "PACF lag selected: {}",
        "critical": "Critical error during PACF lag selection: {}"
    },
    "ru": {
        "init": "Инициализация выбора лага по PACF",
        "no_numeric": "Нет числовых колонок для расчета PACF",
        "pacf_error": "Ошибка расчета PACF для колонки={}",
        "no_valid": "Нет валидных расчетов PACF, используем lag=1",
        "done": "Выбран лаг по PACF: {}",
        "critical": "Критическая ошибка при выборе лага по PACF: {}"
    },
    "zh": {
        "init": "PACF滞后选择初始化",
        "no_numeric": "没有可用于PACF计算的数值列",
        "pacf_error": "列PACF计算错误={}",
        "no_valid": "无有效PACF结果，默认lag=1",
        "done": "PACF选择的滞后: {}",
        "critical": "PACF滞后选择严重错误: {}"
    }
}


def select_pacf_lag(df, col_target, col_time=None, max_lag=200, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

    msg = MESSAGES.get(logger_language, MESSAGES["en"])

    try:
        logger.info(msg["init"])

        df = df.copy()

        if col_time is not None and col_time in df.columns:
            df = df.sort_values(col_time)
        else:
            df = df.sort_values(df.columns[0])

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        exclude = {col_target}
        if col_time is not None:
            exclude.add(col_time)

        numeric_cols = [c for c in numeric_cols if c not in exclude]

        if len(numeric_cols) == 0:
            logger.error(msg["no_numeric"])
            raise ValueError(msg["no_numeric"])

        lag_scores = np.zeros(max_lag)
        valid_cols = 0

        for col in numeric_cols:
            series = df[col].values
            series = pd.Series(series).ffill().bfill().values

            try:
                pacf_vals = pacf(series, nlags=max_lag, method="yw")
            except Exception:
                logger.exception(msg["pacf_error"].format(col))
                continue

            if len(pacf_vals) < max_lag + 1:
                continue

            lag_scores += np.abs(pacf_vals[1:max_lag + 1])
            valid_cols += 1

        if valid_cols == 0:
            logger.warning(msg["no_valid"])
            return 1

        lag_scores = lag_scores / valid_cols
        best_lag = int(np.argmax(lag_scores)) + 1

        logger.info(msg["done"].format(best_lag))

        logger.info(f"Selected lag: {best_lag}")

        return best_lag

    except Exception as e:
        logger.exception(msg["critical"].format(e))
        raise