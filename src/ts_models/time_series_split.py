import pandas as pd
import logging
from config import logger_language


MESSAGES = {
    "en": {
        "init": "Initializing time series train/test split",
        "parse_error": "Error parsing datetime column",
        "train_created": "Train set created: {} rows",
        "test_created": "Test set created: {} rows",
        "overlap_error": "Train and test sets have overlapping timestamps: {} rows",
        "empty_train": "Train set is empty",
        "empty_test": "Test set is empty",
        "critical": "Critical error during train/test split: {}",
        "invalid_date": "Invalid date format in {}: {}",
        "nat_date": "NaT detected in {}: {}",
        "date_order_error": "{} must be before {}"
    },
    "ru": {
        "init": "Инициализация разбиения временного ряда на train/test",
        "parse_error": "Ошибка преобразования datetime колонки",
        "train_created": "Train набор создан: {} строк",
        "test_created": "Test набор создан: {} строк",
        "overlap_error": "Пересечение train и test наборов: {} строк",
        "empty_train": "Train набор пустой",
        "empty_test": "Test набор пустой",
        "critical": "Критическая ошибка при разбиении train/test: {}",
        "invalid_date": "Неверный формат даты в {}: {}",
        "nat_date": "Обнаружен NaT в {}: {}",
        "date_order_error": "{} должно быть меньше {}"
    },
    "zh": {
        "init": "时间序列 train/test 划分初始化",
        "parse_error": "日期时间列解析错误",
        "train_created": "训练集已创建: {} 行",
        "test_created": "测试集已创建: {} 行",
        "overlap_error": "训练集和测试集存在重叠: {} 行",
        "empty_train": "训练集为空",
        "empty_test": "测试集为空",
        "critical": "train/test 划分严重错误: {}",
        "invalid_date": "日期格式无效 {}: {}",
        "nat_date": "检测到 NaT {}: {}",
        "date_order_error": "{} 必须小于 {}"
    }
}


def safe_to_datetime(value, name, logger):
    import pandas as pd
    from config import logger_language

    msg = MESSAGES.get(logger_language, MESSAGES["en"])

    try:
        dt = pd.to_datetime(value, errors="raise")
    except Exception:
        logger.error(msg["invalid_date"].format(name, value))
        raise ValueError(msg["invalid_date"].format(name, value))

    if pd.isna(dt):
        logger.error(msg["nat_date"].format(name, value))
        raise ValueError(msg["nat_date"].format(name, value))

    return dt


def split_train_test(
        df,
        start_train_date,
        end_train_date,
        start_test_date,
        end_test_date,
        col_time="Datetime",
        logger=None
):
    import pandas as pd
    import logging

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

    msg = MESSAGES.get(logger_language, MESSAGES["en"])

    try:
        logger.info(msg["init"])

        df = df.copy()

        df[col_time] = pd.to_datetime(df[col_time], errors="coerce")
        df = df.dropna(subset=[col_time])

        start_train_date = safe_to_datetime(start_train_date, "start_train_date", logger)
        end_train_date = safe_to_datetime(end_train_date, "end_train_date", logger)
        start_test_date = safe_to_datetime(start_test_date, "start_test_date", logger)
        end_test_date = safe_to_datetime(end_test_date, "end_test_date", logger)

        if start_train_date > end_train_date:
            raise ValueError("start_train_date > end_train_date")

        if start_test_date > end_test_date:
            raise ValueError("start_test_date > end_test_date")

        df_train = df[
            (df[col_time] >= start_train_date) &
            (df[col_time] <= end_train_date)
            ]

        df_test = df[
            (df[col_time] >= start_test_date) &
            (df[col_time] <= end_test_date)
            ]

        if df_train.empty:
            raise ValueError(msg["empty_train"])

        if df_test.empty:
            raise ValueError(msg["empty_test"])

        overlap = pd.merge(df_train[[col_time]], df_test[[col_time]], on=col_time, how="inner")

        if not overlap.empty:
            raise ValueError(msg["overlap_error"].format(len(overlap)))

        logger.info(msg["train_created"].format(len(df_train)))
        logger.info(msg["test_created"].format(len(df_test)))

        return df_train, df_test

    except Exception as e:
        logger.exception(msg["critical"].format(e))
        raise