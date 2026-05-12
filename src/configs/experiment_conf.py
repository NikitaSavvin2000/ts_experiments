import os
import yaml
import shutil
import logging

from src.utils.logger import get_logger
from config import experiment_output_dir, logger_language


MESSAGES = {
    "en": {
        "setup": "Experiment setup",
        "start": "Start",
        "folder_created": "Experiment folder created: {}",
        "folder_exists": "Experiment folder already exists: {}",
        "zip_success": "Src snapshot created: {}.zip",
        "error_folder": "Error while creating experiment folder: {}",
        "error_zip": "Error while creating src snapshot: {}"
    },
    "ru": {
        "setup": "Инициализация эксперимента",
        "start": "Начало",
        "folder_created": "Папка эксперимента создана: {}",
        "folder_exists": "Папка эксперимента уже существует: {}",
        "zip_success": "Снимок src директории успешно создан: {}.zip",
        "error_folder": "Ошибка при создании папки эксперимента: {}",
        "error_zip": "Ошибка при создании снимка src директории: {}"
    },
    "zh": {
        "setup": "实验初始化",
        "start": "开始",
        "folder_created": "实验文件夹已创建: {}",
        "folder_exists": "实验文件夹已存在: {}",
        "zip_success": "src目录快照已创建: {}.zip",
        "error_folder": "创建实验文件夹时出错: {}",
        "error_zip": "创建src快照时出错: {}"
    }
}


def init_experiment_config():

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    msg = MESSAGES.get(logger_language, MESSAGES["en"])

    logger.info(msg["setup"])

    home_path = os.getcwd()

    export_path = os.path.join(
        home_path,
        experiment_output_dir
    )

    os.makedirs(export_path, exist_ok=True)

    params_path = os.path.join(
        home_path,
        "src",
        "runners",
        "params.yaml"
    )

    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    experiment_dir_name = params["experiment_unique_name"]

    try:
        experiment_path = os.path.join(
            export_path,
            experiment_dir_name
        )

        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
            logger.info(msg["folder_created"].format(experiment_path))
        else:
            logger.info(msg["folder_exists"].format(experiment_path))

    except Exception as e:
        logger.error(msg["error_folder"].format(e))
        raise

    logger = get_logger(
        log_dir=experiment_path,
        name="experiment"
    )

    logger.info(msg["start"])

    src_path = os.path.join(home_path, "src")
    zip_path = os.path.join(experiment_path, "src_snapshot")

    try:
        shutil.make_archive(
            zip_path,
            "zip",
            src_path
        )

        logger.info(msg["zip_success"].format(zip_path))

    except Exception as e:
        logger.error(msg["error_zip"].format(e))
        raise

    return home_path, export_path, experiment_path, logger