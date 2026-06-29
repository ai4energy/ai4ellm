"""
文件工具模块

提供文件扫描、断点续跑、路径处理、文本合并等通用工具函数。
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Optional

from src.logger import get_logger

logger = get_logger()


def scan_files(root_dir: str, extensions: list[str], recursive: bool = True) -> list[str]:
    """
    扫描目录，返回匹配扩展名的文件绝对路径列表。

    参数:
        root_dir: 要扫描的根目录
        extensions: 文件扩展名列表，如 [".pdf", ".docx"]
        recursive: 是否递归扫描子目录

    返回:
        按路径排序的文件绝对路径列表
    """
    files = []
    ext_set = {ext.lower() for ext in extensions}

    if recursive:
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in ext_set:
                    files.append(os.path.abspath(os.path.join(dirpath, fname)))
    else:
        for fname in os.listdir(root_dir):
            full_path = os.path.join(root_dir, fname)
            if os.path.isfile(full_path) and os.path.splitext(fname)[1].lower() in ext_set:
                files.append(os.path.abspath(full_path))

    files.sort()
    return files


def compute_file_hash(file_path: str) -> str:
    """
    计算文件的 SHA-256 哈希值。

    参数:
        file_path: 文件路径

    返回:
        SHA-256 十六进制字符串
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def ensure_dir(path: str) -> str:
    """
    确保目录存在，返回绝对路径。

    参数:
        path: 目录路径

    返回:
        绝对路径
    """
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def resolve_checkpoint_path(output_dir: str, checkpoint_name: str) -> str:
    """
    解析断点文件路径。

    相对路径统一放在 output_dir 下；绝对路径保持不变。
    """
    if os.path.isabs(checkpoint_name):
        return os.path.abspath(checkpoint_name)
    return os.path.abspath(os.path.join(output_dir, checkpoint_name))


class ResumeTracker:
    """断点续跑管理器，记录已处理文件的哈希和状态。"""

    def __init__(self, checkpoint_path: str):
        """
        加载或创建 processed_files.json 断点记录。

        参数:
            checkpoint_path: 断点记录文件的完整路径
        """
        self._checkpoint_path = os.path.abspath(checkpoint_path)
        if os.path.exists(self._checkpoint_path):
            try:
                with open(self._checkpoint_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {"version": 1, "files": {}}
        else:
            self._data = {"version": 1, "files": {}}

    def is_processed(self, file_path: str, required_stages: Optional[list[str]] = None) -> bool:
        """
        判断某文件是否已经完成所有要求的处理阶段。

        参数:
            file_path: 文件绝对路径
            required_stages: 需要完成的阶段列表，如 ["extract_pdf", "clean_markdown"]
                           为 None 时只检查是否有任何记录

        返回:
            是否已完成
        """
        abs_path = os.path.abspath(file_path)
        if abs_path not in self._data.get("files", {}):
            return False

        record = self._data["files"][abs_path]
        stages = record.get("stages", {})

        if required_stages is None:
            return len(stages) > 0

        for stage in required_stages:
            if stages.get(stage, {}).get("status") != "completed":
                return False

        return True

    def is_stage_completed(self, file_path: str, stage: str) -> bool:
        """
        判断某文件的某个特定阶段是否已完成。

        参数:
            file_path: 文件绝对路径
            stage: 阶段名称

        返回:
            该阶段是否已完成
        """
        abs_path = os.path.abspath(file_path)
        record = self._data.get("files", {}).get(abs_path, {})
        stage_info = record.get("stages", {}).get(stage, {})
        if stage_info.get("status") != "completed":
            return False

        # 文件变化后不能沿用旧断点。
        try:
            stat = os.stat(abs_path)
            if record.get("size") != stat.st_size:
                return False
            if record.get("file_hash") and record.get("file_hash") != compute_file_hash(abs_path):
                return False
        except OSError:
            return False

        # 如果该阶段声明了输出产物，产物必须仍然存在。否则清空 output 后
        # checkpoint 会导致“跳过提取但下游无文件”的假完成。
        output = stage_info.get("output", "")
        if output and not os.path.exists(output):
            return False

        return True

    def mark_processed(self, file_path: str, stage: str, output_path: str = ""):
        """
        标记某文件在某个阶段已完成处理。

        参数:
            file_path: 文件绝对路径
            stage: 阶段名称
            output_path: 该阶段产生的输出文件路径
        """
        abs_path = os.path.abspath(file_path)
        if abs_path not in self._data["files"]:
            try:
                stat = os.stat(abs_path)
                file_hash = compute_file_hash(abs_path)
                self._data["files"][abs_path] = {
                    "file_hash": file_hash,
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "stages": {},
                }
            except OSError:
                self._data["files"][abs_path] = {
                    "file_hash": "",
                    "mtime": 0,
                    "size": 0,
                    "stages": {},
                }

        self._data["files"][abs_path]["stages"][stage] = {
            "status": "completed",
            "output": os.path.abspath(output_path) if output_path else "",
            "completed_at": datetime.now().isoformat(),
        }

    def mark_failed(self, file_path: str, stage: str, error: str):
        """
        标记某文件在某阶段处理失败。

        参数:
            file_path: 文件绝对路径
            stage: 阶段名称
            error: 错误信息
        """
        abs_path = os.path.abspath(file_path)
        if abs_path not in self._data["files"]:
            self._data["files"][abs_path] = {
                "file_hash": "",
                "mtime": 0,
                "size": 0,
                "stages": {},
            }

        self._data["files"][abs_path]["stages"][stage] = {
            "status": "failed",
            "error": str(error),
            "completed_at": datetime.now().isoformat(),
        }

    def save(self):
        """将处理记录持久化到磁盘。"""
        self._data["updated_at"] = datetime.now().isoformat()
        ensure_dir(os.path.dirname(self._checkpoint_path))
        with open(self._checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    @property
    def failed_files(self) -> list[str]:
        """返回所有处理失败的文件路径列表。"""
        failed = []
        for path, record in self._data.get("files", {}).items():
            for stage_info in record.get("stages", {}).values():
                if stage_info.get("status") == "failed":
                    failed.append(path)
                    break
        return failed
