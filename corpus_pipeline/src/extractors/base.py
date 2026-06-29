"""
提取器抽象基类模块

定义所有文档提取器必须实现的接口。
"""

from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    """文档提取器抽象基类。"""

    def __init__(self, config: dict):
        """
        初始化提取器。

        参数:
            config: 提取器相关配置字典
        """
        self._config = config

    @abstractmethod
    def extract(self, file_path: str, output_dir: str) -> str | None:
        """
        提取单个文档的文本。

        参数:
            file_path: 输入文档的绝对路径
            output_dir: 输出目录

        返回:
            输出文件的绝对路径，失败返回 None
        """
        ...

    @abstractmethod
    def supports_extension(self, ext: str) -> bool:
        """
        判断该提取器是否支持给定文件扩展名。

        参数:
            ext: 文件扩展名，如 ".pdf"

        返回:
            是否支持
        """
        ...
