"""
日志工具模块

提供统一的 loguru 日志配置，支持控制台和文件输出。
所有模块应通过 get_logger() 获取日志器实例。
"""

import os
import shutil
from datetime import datetime
from loguru import logger


def _ensure_directory(path: str) -> str:
    """Create a directory, moving aside a conflicting file if needed."""
    if os.path.isdir(path):
        return path

    if os.path.lexists(path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{path}.file_bak_{timestamp}"
        counter = 1
        while os.path.lexists(backup_path):
            backup_path = f"{path}.file_bak_{timestamp}_{counter}"
            counter += 1
        shutil.move(path, backup_path)

    os.makedirs(path, exist_ok=True)
    return path


def setup_logger(config: dict) -> "logger":
    """
    根据配置初始化 loguru 日志器。

    参数:
        config: 日志配置字典，包含 level、log_dir、log_to_file、
                log_to_console、rotation、retention 等键

    返回:
        配置好的 loguru logger 实例
    """
    # 移除默认的 handler
    logger.remove()

    level = config.get("level", "INFO")
    log_dir = config.get("log_dir", "./output/logs")

    # 确保日志目录存在
    _ensure_directory(log_dir)

    # 控制台输出
    if config.get("log_to_console", True):
        logger.add(
            lambda msg: print(msg, end=""),
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
        )

    # 文件输出
    if config.get("log_to_file", True):
        log_file = os.path.join(log_dir, "pipeline_{time:YYYY-MM-DD}.log")
        logger.add(
            log_file,
            level=level,
            rotation=config.get("rotation", "10 MB"),
            retention=config.get("retention", "30 days"),
            encoding="utf-8",
            enqueue=True,
        )

    return logger


def get_logger():
    """
    获取全局 loguru 日志器实例。

    返回:
        loguru.logger 实例

    注意:
        使用前必须先调用 setup_logger() 进行初始化。
    """
    return logger
