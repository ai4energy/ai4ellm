"""
配置读取与参数校验模块

从 YAML 文件加载流水线配置，提供便捷的访问方法。
"""

import os
import yaml


class PipelineConfig:
    """从 YAML 文件加载并验证流水线配置。"""

    def __init__(self, config_path: str):
        """
        加载 YAML 配置，校验必填字段，设置默认值。

        参数:
            config_path: YAML 配置文件的路径
        """
        config_path = os.path.abspath(config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

        validate_config(self._data)

    # ---- 便捷访问方法 ----

    def get_paths(self) -> dict:
        """返回路径配置。"""
        return self._data.get("paths", {})

    def get_scan_config(self) -> dict:
        """返回文档扫描配置。"""
        return self._data.get("scan", {})

    def get_extractor_config(self) -> dict:
        """返回提取器配置。"""
        return self._data.get("extractors", {})

    def get_cleaner_config(self) -> dict:
        """返回清洗器配置。"""
        return self._data.get("cleaner", {})

    def get_splitter_config(self) -> dict:
        """返回分割器配置。"""
        return self._data.get("splitter", {})

    def get_quality_config(self) -> dict:
        """返回语料质量评分配置。"""
        return self._data.get("quality", {})

    def get_dedup_config(self) -> dict:
        """返回去重配置。"""
        return self._data.get("dedup", {})

    def get_exporter_config(self) -> dict:
        """返回导出器配置。"""
        return self._data.get("exporter", {})

    def get_logging_config(self) -> dict:
        """返回日志配置。"""
        return self._data.get("logging", {})

    def get_pipeline_steps(self) -> dict:
        """返回流水线步骤开关。"""
        return self._data.get("pipeline", {}).get("steps", {})

    def get_failure_strategy(self) -> str:
        """返回失败策略（continue / abort）。"""
        return self._data.get("pipeline", {}).get("failure_strategy", "continue")

    @property
    def raw(self) -> dict:
        """返回原始配置字典（用于调试）。"""
        return self._data


def validate_config(data: dict) -> None:
    """
    校验配置中必填字段是否存在，缺失则抛出 ValueError。

    参数:
        data: 配置字典

    异常:
        ValueError: 当缺少必填字段时
    """
    required_sections = ["paths", "scan", "extractors", "cleaner", "dedup", "exporter"]
    for section in required_sections:
        if section not in data:
            raise ValueError(f"配置文件缺少必填段落: {section}")

    paths = data.get("paths", {})
    if not paths.get("input_dir"):
        raise ValueError("paths.input_dir 不能为空")
    if not paths.get("output_dir"):
        raise ValueError("paths.output_dir 不能为空")


def load_config(config_path: str = "config.yaml") -> PipelineConfig:
    """
    加载配置文件的便捷函数。

    参数:
        config_path: YAML 配置文件路径

    返回:
        PipelineConfig 实例
    """
    return PipelineConfig(config_path)
