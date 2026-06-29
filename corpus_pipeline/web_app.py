#!/usr/bin/env python3
"""
语料库构建流水线 — Web 应用入口

用法:
    streamlit run web_app.py --server.port 7860 --server.address 0.0.0.0

本应用封装了完整的语料库构建流水线，提供浏览器界面进行：
文件上传、参数配置、流水线运行、日志查看、结果下载。
"""

import os
import sys
import re
import shutil
import json
import time
import zipfile
import glob
import yaml
import hashlib
from datetime import datetime
from pathlib import Path

import streamlit as st

# 将项目根目录加入 Python 路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# ---------- 常量 ----------
INPUT_DIR = os.path.join(ROOT_DIR, "input_docs")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")
ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".txt", ".md"}
MAX_FILE_SIZE_MB = 500
IN_DOCKER = os.path.exists("/.dockerenv") or os.environ.get("RUNNING_IN_DOCKER") == "1"


def ensure_directory(path: str) -> str:
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

# 跳过解压的文件/目录模式
ZIP_SKIP_DIRS = {"__MACOSX", ".DS_Store", "thumbs.db"}
ZIP_SKIP_FILES = {".DS_Store", "thumbs.db", "Thumbs.db"}
ZIP_SKIP_PATTERNS = [r"^~\$.*"]  # Office 临时文件


def _check_magic_pdf_available() -> bool:
    """检测当前环境是否安装了 magic-pdf。"""
    try:
        from magic_pdf.data.data_reader_writer import FileBasedDataReader  # noqa: F401
        from magic_pdf.tools.common import do_parse  # noqa: F401
        return True
    except ImportError:
        return False


MAGIC_PDF_AVAILABLE = _check_magic_pdf_available()

st.set_page_config(
    page_title="语料构建工作台",
    page_icon="CP",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp {
        background: #f6f7f9;
      }
      [data-testid="stSidebar"] {
        background: #111827;
      }
      [data-testid="stSidebar"] * {
        color: #f9fafb;
      }
      [data-testid="stSidebar"] input,
      [data-testid="stSidebar"] textarea,
      [data-testid="stSidebar"] select {
        color: #111827 !important;
      }
      .block-container {
        padding-top: 1.25rem;
        max-width: 1440px;
      }
      .hero {
        background: linear-gradient(135deg, #111827 0%, #1f2937 55%, #0f766e 100%);
        color: #fff;
        padding: 24px 28px;
        border-radius: 8px;
        margin-bottom: 18px;
      }
      .hero h1 {
        font-size: 30px;
        line-height: 1.2;
        margin: 0 0 8px 0;
        letter-spacing: 0;
      }
      .hero p {
        margin: 0;
        color: #d1d5db;
        font-size: 15px;
      }
      .metric-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 18px;
      }
      .metric-card {
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
      }
      .metric-card .label {
        color: #6b7280;
        font-size: 12px;
        margin-bottom: 6px;
      }
      .metric-card .value {
        color: #111827;
        font-size: 18px;
        font-weight: 700;
      }
      .metric-card .sub {
        color: #6b7280;
        font-size: 12px;
        margin-top: 4px;
      }
      div[data-testid="stFileUploader"] section {
        border-radius: 8px;
        border-color: #d1d5db;
        background: #fff;
      }
      div.stButton > button,
      div.stDownloadButton > button {
        border-radius: 6px;
        min-height: 38px;
      }
      pre {
        white-space: pre-wrap !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
      }
      @media (max-width: 900px) {
        .metric-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_runtime_status() -> dict:
    """检测当前 Web 运行环境能力。"""
    status = {
        "pdf_engine": "MinerU / magic-pdf" if MAGIC_PDF_AVAILABLE else "未安装",
        "gpu": "未检测",
        "gpu_count": 0,
        "torch": "未安装",
    }
    try:
        import torch
        status["torch"] = torch.__version__
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(count)]
            status["gpu_count"] = count
            status["gpu"] = ", ".join(names)
        else:
            status["gpu"] = "CUDA 不可用"
    except Exception as e:
        status["gpu"] = f"检测失败: {e}"
    return status


# ============================================================
# 辅助函数 — 通用
# ============================================================

def format_size(size_bytes: int) -> str:
    """格式化文件大小。"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _get_config_output_dir() -> str:
    """从 config.yaml 读取 output_dir，缺省返回 ./output。"""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("paths", {}).get("output_dir", "./output")
    except Exception:
        return "./output"


def _resolve_output_dir(user_value: str) -> str:
    """将用户输入的输出目录解析为绝对路径。"""
    if IN_DOCKER:
        # 容器无法直接写 Windows 盘符路径。宿主机目录必须通过 compose
        # volume 挂载到 /app/output，应用内部始终写容器路径。
        return OUTPUT_DIR
    if os.path.isabs(user_value):
        return user_value
    return os.path.abspath(os.path.join(ROOT_DIR, user_value))


def _is_windows_host_path(value: str) -> bool:
    return bool(re.match(r"^[a-zA-Z]:[\\/]", value.strip()))


def _save_output_dir_to_config(output_dir: str):
    """将输出目录保存到 config.yaml。"""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("paths", {})["output_dir"] = output_dir
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    except Exception as e:
        st.error(f"保存 config.yaml 失败: {e}")


def _bytes_hash(data: bytes) -> str:
    """计算上传内容哈希，用于避免 Streamlit rerun 重复保存同一文件。"""
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ============================================================
# 辅助函数 — 文件管理
# ============================================================

def is_supported_document(file_path: Path, supported_exts: set[str]) -> bool:
    """判断是否为支持的文档格式。"""
    return file_path.suffix.lower() in supported_exts


def _should_skip_zip_entry(name: str) -> bool:
    """判断 ZIP 条目是否应跳过。"""
    # 目录名检查
    for skip_dir in ZIP_SKIP_DIRS:
        if skip_dir in name.split("/"):
            return True
    # 文件名检查
    basename = os.path.basename(name)
    if basename in ZIP_SKIP_FILES:
        return True
    # 正则模式检查
    for pat in ZIP_SKIP_PATTERNS:
        if re.match(pat, basename):
            return True
    # 扩展名检查
    return Path(basename).suffix.lower() not in ALLOWED_EXTENSIONS


def safe_extract_zip(zip_path: Path, target_dir: Path) -> dict:
    """
    安全解压 ZIP，防止路径穿越，过滤不支持文件。

    返回统计信息字典。
    """
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "imported": 0, "skipped": 0, "skip_reasons": []}
    saved_paths = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for entry in zf.namelist():
                stats["total"] += 1

                # 跳过不支持的文件
                if _should_skip_zip_entry(entry):
                    stats["skipped"] += 1
                    stats["skip_reasons"].append(f"不支持格式: {entry}")
                    continue

                # 安全路径检查（防止路径穿越）
                entry_path = Path(entry)
                resolved = (target_dir / entry_path).resolve()
                if not str(resolved).startswith(str(target_dir)):
                    stats["skipped"] += 1
                    stats["skip_reasons"].append(f"路径穿越（已拒绝）: {entry}")
                    continue

                # 跳过目录条目
                if entry.endswith("/"):
                    stats["skipped"] += 1
                    continue

                # 目标路径。ZIP 导入默认扁平化到 input_docs 根目录，避免深层目录
                # 在 Web 管理界面里难以辨认。
                dest = target_dir / entry_path.name

                # 如果同一目录已有同名文件，加时间戳
                if dest.exists():
                    incoming = zf.read(entry)
                    try:
                        if _file_hash(dest) == _bytes_hash(incoming):
                            stats["skipped"] += 1
                            stats["skip_reasons"].append(f"重复文件: {entry}")
                            continue
                    except OSError:
                        pass
                    stem = entry_path.stem
                    ext = entry_path.suffix
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    dest = target_dir / f"{stem}_{ts}{ext}"
                else:
                    incoming = zf.read(entry)

                # 写入文件
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as out:
                    out.write(incoming)
                saved_paths.append(str(dest.resolve()))
                stats["imported"] += 1

    except zipfile.BadZipFile:
        stats["error"] = "无效的 ZIP 文件"
    except Exception as e:
        stats["error"] = str(e)

    stats["saved_paths"] = saved_paths
    return stats


def extract_zip_to_input_docs(zip_file, input_dir: Path) -> dict:
    """
    安全解压上传的 ZIP 文件到 input_docs。

    参数:
        zip_file: Streamlit UploadedFile 对象
        input_dir: 目标目录 Path

    返回:
        统计信息字典
    """
    input_dir = input_dir.resolve()
    tmp_zip = input_dir / f"_tmp_upload_{int(time.time() * 1000)}.zip"

    try:
        with open(tmp_zip, "wb") as f:
            f.write(zip_file.getvalue())
        stats = safe_extract_zip(tmp_zip, input_dir)
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink()

    return stats


def save_uploaded_file(uploaded_file, input_dir: str) -> str | None:
    """
    保存上传的文件到 input_docs/，处理重名。

    参数:
        uploaded_file: Streamlit UploadedFile 对象
        input_dir: 目标目录路径

    返回:
        保存的绝对路径，失败或格式不支持返回 None
    """
    input_path = Path(input_dir).resolve()
    input_path.mkdir(parents=True, exist_ok=True)

    original_name = uploaded_file.name
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return None

    data = uploaded_file.getvalue()
    name_without_ext = Path(original_name).stem
    target = input_path / original_name

    # 处理重名：如果内容完全相同，视为已存在，不再保存。
    # Streamlit 会在页面 rerun 时保留 uploader 的文件对象；这里必须幂等，
    # 否则一次上传会被保存成多个时间戳副本。
    if target.exists():
        try:
            if target.stat().st_size == len(data) and _file_hash(target) == _bytes_hash(data):
                return str(target.resolve())
        except OSError:
            pass
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = input_path / f"{name_without_ext}_{ts}{ext}"

    with open(target, "wb") as f:
        f.write(data)

    return str(target.resolve())


def get_file_list(directory: str) -> list[dict]:
    """递归获取目录中所有支持格式的文件信息（含子目录）。"""
    if not os.path.isdir(directory):
        return []
    files = []
    for dirpath, _, filenames in os.walk(directory):
        # 跳过临时/隐藏目录
        if "__MACOSX" in dirpath or dirpath.startswith("."):
            continue
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                stat = os.stat(fpath)
                # 相对路径用于显示
                rel = os.path.relpath(fpath, directory)
                files.append({
                    "name": fname,
                    "display_path": rel,
                    "full_path": os.path.abspath(fpath),
                    "ext": ext,
                    "size": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                })
    return sorted(files, key=lambda x: x["mtime"], reverse=True)


def delete_files(file_paths: list[str], input_dir: str) -> list[str]:
    """
    安全删除 input_docs 中的文件，并同步清理断点记录。

    参数:
        file_paths: 要删除的文件绝对路径列表
        input_dir: input_docs 的绝对路径

    返回:
        错误信息列表
    """
    input_root = Path(input_dir).resolve()
    errors = []

    for fpath_str in file_paths:
        fpath = Path(fpath_str).resolve()

        # 安全检查：必须在 input_docs 下
        if not str(fpath).startswith(str(input_root)):
            errors.append(f"禁止删除 input_docs 之外的文件: {fpath_str}")
            continue

        if not fpath.exists():
            errors.append(f"文件已不存在: {fpath.name}")
            continue

        if not fpath.is_file():
            errors.append(f"不是文件: {fpath.name}")
            continue

        try:
            fpath.unlink()
            # 同步清理断点记录
            _remove_checkpoint_entry(str(fpath))
        except PermissionError:
            errors.append(f"文件被占用，无法删除: {fpath.name}")
        except OSError as e:
            errors.append(f"删除失败 {fpath.name}: {e}")

    return errors


def _remove_checkpoint_entry(file_path: str):
    """从 processed_files.json 中删除指定文件的记录。"""
    from src.file_utils import resolve_checkpoint_path

    checkpoint = resolve_checkpoint_path(OUTPUT_DIR, "processed_files.json")
    if not os.path.exists(checkpoint):
        return
    try:
        with open(checkpoint, "r", encoding="utf-8") as f:
            data = json.load(f)
        abs_path = os.path.abspath(file_path)
        if abs_path in data.get("files", {}):
            del data["files"][abs_path]
            with open(checkpoint, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ============================================================
# 辅助函数 — 流水线运行
# ============================================================

def run_pipeline(output_dir: str) -> dict:
    """
    运行完整流水线。

    参数:
        output_dir: 输出目录绝对路径

    返回:
        统计结果字典
    """
    from src.config import load_config, PipelineConfig
    from src.file_utils import resolve_checkpoint_path
    from src.pipeline import CorpusPipeline

    # 加载配置
    config = load_config(CONFIG_PATH)

    # 应用 UI 配置覆盖
    max_chars = st.session_state.get("max_segment_chars", 500)
    min_chars = st.session_state.get("min_segment_chars", 50)
    force_rerun = st.session_state.get("force_rerun", False)
    rule_dedup = st.session_state.get("rule_dedup", True)
    semantic_dedup = st.session_state.get("semantic_dedup", False)
    max_pages = st.session_state.get("max_pdf_pages", 2000)
    quality_enabled = st.session_state.get("quality_enabled", True)
    quality_filter_enabled = st.session_state.get("quality_filter_enabled", False)
    min_quality_score = st.session_state.get("min_quality_score", 0.3)
    formula_enable = st.session_state.get("formula_enable", False)
    table_enable = st.session_state.get("table_enable", False)
    num_gpus = st.session_state.get("num_gpus", 1)

    # 覆盖输出目录
    config.raw["paths"]["output_dir"] = output_dir

    # 覆盖其他配置
    config.raw["splitter"]["max_segment_chars"] = max_chars
    config.raw["splitter"]["min_segment_chars"] = min_chars
    config.raw["dedup"]["rule_dedup"]["enabled"] = rule_dedup
    config.raw["dedup"]["semantic_dedup"]["enabled"] = semantic_dedup
    config.raw["extractors"]["pdf"]["max_pages"] = max_pages
    config.raw["extractors"]["pdf"]["formula_enable"] = formula_enable
    config.raw["extractors"]["pdf"]["table_enable"] = table_enable
    config.raw["extractors"]["pdf"]["num_gpus"] = num_gpus
    config.raw["quality"]["enabled"] = quality_enabled
    config.raw["quality"]["filter_enabled"] = quality_filter_enabled
    config.raw["quality"]["min_score"] = min_quality_score

    # 断点路径也跟随输出目录
    checkpoint_path = resolve_checkpoint_path(output_dir, "processed_files.json")

    # 如果强制重跑，清空断点记录
    if force_rerun and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass

    # 确保输出目录存在
    ensure_directory(os.path.join(output_dir, "logs"))
    ensure_directory(os.path.join(output_dir, "reports"))

    # 运行流水线
    pipeline = CorpusPipeline(config)
    # 覆盖 tracker 的 checkpoint 路径
    pipeline.tracker._checkpoint_path = os.path.abspath(checkpoint_path)
    stats = pipeline.run(input_dir=INPUT_DIR)

    # 返回统计结果
    return {
        "files_scanned": stats.files_scanned,
        "files_converted": stats.files_converted,
        "files_extracted": stats.files_extracted,
        "files_text_read": stats.files_text_read,
        "sections_cleaned": stats.sections_cleaned,
        "chunks_generated": stats.chunks_generated,
        "chunks_kept": stats.chunks_kept,
        "chunks_filtered": stats.chunks_filtered,
        "lines_before_dedup": stats.lines_before_dedup,
        "lines_after_rule_dedup": stats.lines_after_rule_dedup,
        "lines_after_semantic_dedup": stats.lines_after_semantic_dedup,
        "output_records": stats.output_records,
        "errors": stats.errors,
        "duration": stats.end_time - stats.start_time if stats.start_time and stats.end_time else 0,
    }


def get_latest_log(output_dir: str) -> tuple[str, str]:
    """获取最新日志文件内容和文件名。"""
    log_dir = os.path.join(output_dir, "logs")
    if not os.path.isdir(log_dir):
        return "暂无日志", ""

    log_files = glob.glob(os.path.join(log_dir, "pipeline_*.log"))
    if not log_files:
        return "暂无日志", ""

    latest = max(log_files, key=os.path.getmtime)
    try:
        with open(latest, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return content, os.path.basename(latest)
    except Exception:
        return "读取日志失败", ""


def get_pipeline_report(output_dir: str) -> str:
    """获取流水线统计报告。"""
    report_path = os.path.join(output_dir, "reports", "pipeline_report.txt")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            pass
    return "暂无报告，请先运行处理流程。"


def get_quality_report(output_dir: str) -> dict | None:
    """获取质量过滤报告。"""
    report_path = os.path.join(output_dir, "reports", "filtered_chunks.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "total_filtered": len(data),
                "avg_score": sum(c.get("quality_score", 0) for c in data) / len(data) if data else 0,
                "sample": data[:5] if data else [],
            }
        except Exception:
            pass
    return None


def create_output_zip(output_dir: str) -> str | None:
    """打包整个输出目录为 zip 文件。"""
    if not os.path.isdir(output_dir):
        return None

    zip_path = os.path.join(output_dir, "output_archive.zip")
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(output_dir):
                for fname in files:
                    if fname == "output_archive.zip":
                        continue
                    fpath = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, output_dir)
                    zf.write(fpath, arcname)
        return zip_path
    except Exception:
        return None


def cleanup_output(output_dir: str):
    """清空输出目录下的中间结果和最终结果。"""
    if not os.path.isdir(output_dir):
        return
    for subdir in ["extracted_text", "cleaned_text", "merged", "deduplicated",
                    "jsonl", "json", "chunks", "metadata"]:
        dirpath = os.path.join(output_dir, subdir)
        if os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
            os.makedirs(dirpath, exist_ok=True)
    cleanup_checkpoint(output_dir)


def cleanup_checkpoint(output_dir: str):
    """清空断点续跑记录。"""
    from src.file_utils import resolve_checkpoint_path

    checkpoint = resolve_checkpoint_path(output_dir, "processed_files.json")
    if os.path.exists(checkpoint):
        os.remove(checkpoint)


def get_downloadable_files(output_dir: str) -> list[dict]:
    """获取可下载的文件列表。"""
    result = []

    # JSONL 输出
    jsonl_dir = os.path.join(output_dir, "jsonl")
    if os.path.isdir(jsonl_dir):
        for f in os.listdir(jsonl_dir):
            if f.endswith(".jsonl"):
                result.append({"name": f"JSONL: {f}", "path": os.path.join(jsonl_dir, f), "type": "jsonl"})

    # JSON 输出
    json_dir = os.path.join(output_dir, "json")
    if os.path.isdir(json_dir):
        for f in os.listdir(json_dir):
            if f.endswith(".json"):
                result.append({"name": f"JSON: {f}", "path": os.path.join(json_dir, f), "type": "json"})

    # 清洗后文本
    cleaned_dir = os.path.join(output_dir, "cleaned_text")
    if os.path.isdir(cleaned_dir):
        for f in os.listdir(cleaned_dir):
            if f.endswith("_cleaned.txt"):
                result.append({"name": f"清洗文本: {f}", "path": os.path.join(cleaned_dir, f), "type": "txt"})
            elif f.endswith("_sections.json"):
                result.append({"name": f"章节结构: {f}", "path": os.path.join(cleaned_dir, f), "type": "json"})

    # Chunks
    chunks_dir = os.path.join(output_dir, "chunks")
    if os.path.isdir(chunks_dir):
        if os.path.exists(os.path.join(chunks_dir, "_chunks.json")):
            result.append({"name": "结构化 Chunks (JSON)", "path": os.path.join(chunks_dir, "_chunks.json"), "type": "json"})

    # 报告
    reports_dir = os.path.join(output_dir, "reports")
    if os.path.isdir(reports_dir):
        for f in os.listdir(reports_dir):
            fpath = os.path.join(reports_dir, f)
            if os.path.isfile(fpath):
                result.append({"name": f"报告: {f}", "path": fpath, "type": "txt"})

    # 元数据
    meta_dir = os.path.join(output_dir, "metadata")
    if os.path.isdir(meta_dir):
        for f in os.listdir(meta_dir):
            if f.endswith(".json"):
                result.append({"name": f"元数据: {f}", "path": os.path.join(meta_dir, f), "type": "json"})

    # Output archive
    archive_path = os.path.join(output_dir, "output_archive.zip")
    if os.path.exists(archive_path):
        result.append({"name": "完整输出 (ZIP)", "path": archive_path, "type": "zip"})

    return sorted(result, key=lambda x: x["name"])


def download_file(file_path: str, file_name: str):
    """提供文件下载按钮。"""
    if not os.path.exists(file_path):
        st.button(file_name, disabled=True, help="文件不存在")
        return

    file_size = os.path.getsize(file_path)
    with open(file_path, "rb") as f:
        st.download_button(
            label=f"📥 {file_name} ({format_size(file_size)})",
            data=f.read(),
            file_name=os.path.basename(file_path),
            mime="application/octet-stream",
        )


# ============================================================
# 页面布局
# ============================================================

# ===== 标题区 =====
runtime_status = get_runtime_status()
st.markdown(
    """
    <div class="hero">
      <h1>语料构建工作台</h1>
      <p>PDF / Word / PPT / TXT / Markdown → MinerU 提取 → 清洗 → 切分 → 去重 → JSONL</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="metric-row">
      <div class="metric-card"><div class="label">PDF 引擎</div><div class="value">{runtime_status['pdf_engine']}</div><div class="sub">OCR / 公式识别</div></div>
      <div class="metric-card"><div class="label">GPU</div><div class="value">{runtime_status['gpu_count']} 张</div><div class="sub">{runtime_status['gpu']}</div></div>
      <div class="metric-card"><div class="label">输入目录</div><div class="value">input_docs</div><div class="sub">支持批量与 ZIP</div></div>
      <div class="metric-card"><div class="label">输出格式</div><div class="value">JSONL / JSON</div><div class="sub">含来源追踪和质量分</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===== 侧边栏配置 =====
with st.sidebar:
    st.header("配置参数")

    # 输出目录
    st.subheader("输出目录")
    current_out = _get_config_output_dir()
    output_dir_input = st.text_input(
        "输出目录路径",
        value=OUTPUT_DIR if IN_DOCKER else current_out,
        disabled=IN_DOCKER,
        help="容器内固定写入 /app/output；如需改 Windows 宿主机目录，请修改 docker-compose.yml 的 output volume 或设置 CORPUS_OUTPUT_DIR。",
    )
    save_to_config = st.checkbox("保存到 config.yaml（持久化）", value=False)
    if IN_DOCKER:
        st.info("当前运行在 Docker 中，页面输出目录固定为 /app/output。宿主机实际目录由 Docker volume 决定。")
        if _is_windows_host_path(current_out):
            st.warning("config.yaml 中保存了 Windows 路径，但容器内不会使用它。本次运行会写入 /app/output。")
    if save_to_config:
        abs_output = _resolve_output_dir(output_dir_input)
        _save_output_dir_to_config(abs_output)
        st.success(f"已保存: {abs_output}")

    st.divider()

    st.subheader("文本切分")
    st.session_state["max_segment_chars"] = st.slider(
        "Chunk 最大字符数",
        min_value=100,
        max_value=2000,
        value=500,
        step=50,
        help="超过此长度的段落会被进一步切分",
    )
    st.session_state["min_segment_chars"] = st.slider(
        "Chunk 最小字符数",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="短于此长度的段落会被跳过",
    )

    st.subheader("去重")
    st.session_state["rule_dedup"] = st.checkbox("启用规则去重", value=True)
    st.session_state["semantic_dedup"] = st.checkbox("启用语义去重", value=False, help="默认关闭；需要 sentence-transformers 模型")

    st.subheader("PDF 提取")
    st.session_state["formula_enable"] = st.checkbox("启用公式识别", value=False, help="会增加显存/内存占用")
    st.session_state["table_enable"] = st.checkbox("启用表格识别", value=False, help="表格模型较重，需要时再打开")
    st.session_state["num_gpus"] = st.number_input(
        "GPU 数量",
        min_value=0,
        max_value=8,
        value=1,
        step=1,
        help="0 表示 CPU；Windows Docker Desktop 通常先用 1，服务器可设 4",
    )
    st.session_state["max_pdf_pages"] = st.slider(
        "PDF 最大页数",
        min_value=100,
        max_value=5000,
        value=2000,
        step=100,
        help="超过此页数的 PDF 将被截断",
    )

    st.subheader("质量评分")
    st.session_state["quality_enabled"] = st.checkbox("启用质量评分", value=True)
    st.session_state["quality_filter_enabled"] = st.checkbox("按质量分过滤删除", value=False, help="默认不删除，只给 chunk 附加质量分")
    st.session_state["min_quality_score"] = st.slider(
        "最低质量阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="只有打开过滤删除时才生效",
    )

    st.subheader("运行选项")
    st.session_state["force_rerun"] = st.checkbox("强制重跑（忽略断点记录）", value=False)

    st.divider()
    if MAGIC_PDF_AVAILABLE:
        st.info("PDF 提取引擎：MinerU / magic-pdf")
    else:
        st.warning("当前环境未安装 MinerU，PDF 文件请使用全功能镜像或本地 conda 环境。")


# ===== 主内容区：分两列 =====
tab_run, tab_observe, tab_download = st.tabs(["处理工作台", "日志与报告", "结果下载"])

# 确定本次运行使用的输出目录
output_dir = _resolve_output_dir(output_dir_input)

# ---- 工作台：文件管理 + 运行控制 ----
with tab_run:
    col_left, col_right = st.columns([1.05, 0.95])
    with col_left:
        st.header("文件管理")

    # 确保 input_docs 目录存在
    os.makedirs(INPUT_DIR, exist_ok=True)

    # 上传区 — 多文件批量上传
    uploaded = st.file_uploader(
        "上传文档（支持批量选择多个文件）",
        type=["pdf", "doc", "docx", "ppt", "pptx", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded:
        handled_uploads = st.session_state.setdefault("handled_uploads", set())
        # 检查是否有 PDF 文件上传但缺少 magic-pdf
        has_pdf = any(Path(uf.name).suffix.lower() == ".pdf" for uf in uploaded)
        if has_pdf and not MAGIC_PDF_AVAILABLE:
            st.warning("⚠️ 检测到 PDF 文件，但当前环境未安装 MinerU / magic-pdf，将无法处理 PDF。请使用完整版镜像或本地环境。")

        success_count = 0
        error_count = 0
        skip_count = 0
        duplicate_count = 0
        for uf in uploaded:
            uf_data = uf.getvalue()
            upload_sig = f"{uf.name}:{len(uf_data)}:{_bytes_hash(uf_data)}"
            if upload_sig in handled_uploads:
                duplicate_count += 1
                continue
            # 检查文件大小
            if len(uf_data) > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"文件 {uf.name} 超过 {MAX_FILE_SIZE_MB}MB 限制，跳过")
                error_count += 1
                continue
            result = save_uploaded_file(uf, INPUT_DIR)
            if result:
                success_count += 1
                handled_uploads.add(upload_sig)
            else:
                skip_count += 1

        if success_count > 0:
            st.success(f"成功上传 {success_count} 个文件")
        if duplicate_count > 0 and success_count == 0:
            st.info("文件已在本次会话中保存，不再重复写入。")
        if skip_count > 0:
            st.warning(f"{skip_count} 个文件格式不支持，已跳过")
        if error_count > 0:
            st.error(f"{error_count} 个文件上传失败（超过大小限制）")

    # 上传区 — ZIP 压缩包上传（文件夹上传兼容方案）
    st.caption("或上传 ZIP 压缩包（用于导入整个文件夹）：")
    uploaded_zip = st.file_uploader(
        "",
        type=["zip"],
        accept_multiple_files=False,
        key="zip_uploader",
    )

    if uploaded_zip:
        with st.spinner("正在解压 ZIP ..."):
            zip_stats = extract_zip_to_input_docs(uploaded_zip, Path(INPUT_DIR))

        if "error" in zip_stats:
            st.error(f"ZIP 解压失败: {zip_stats['error']}")
        else:
            st.success(
                f"ZIP 导入完成："
                f"总条目 {zip_stats['total']}，"
                f"成功 {zip_stats['imported']}，"
                f"跳过 {zip_stats['skipped']}"
            )
            if zip_stats["skip_reasons"]:
                with st.expander(f"查看跳过原因（{len(zip_stats['skip_reasons'])} 条）"):
                    for r in zip_stats["skip_reasons"][:20]:
                        st.text(r)
                    if len(zip_stats["skip_reasons"]) > 20:
                        st.text(f"... 还有 {len(zip_stats['skip_reasons']) - 20} 条")

    # 文件列表
    st.subheader("输入文件列表")
    files = get_file_list(INPUT_DIR)
    if files:
        # 显示为表格
        file_df = {
            "文件名": [f["name"] for f in files],
            "路径": [f["display_path"] for f in files],
            "类型": [f["ext"] for f in files],
            "大小": [format_size(f["size"]) for f in files],
            "修改时间": [f["mtime"] for f in files],
        }
        st.dataframe(file_df, use_container_width=True, hide_index=True)

        # 删除功能
        st.caption("选择要删除的文件：")
        files_to_delete = st.multiselect(
            "",
            options=[f["full_path"] for f in files],
            format_func=lambda x: os.path.relpath(x, INPUT_DIR),
            key="delete_files_select",
        )
        if files_to_delete:
            st.warning(f"即将删除 {len(files_to_delete)} 个文件，同时清理对应断点记录。")
            if st.button("🗑️ 确认删除", type="secondary"):
                errs = delete_files(files_to_delete, INPUT_DIR)
                if errs:
                    for err in errs:
                        st.error(err)
                else:
                    st.success(f"已删除 {len(files_to_delete)} 个文件")
                st.rerun()
    else:
        st.info("input_docs/ 目录中暂无文件，请上传。")

    st.divider()

    # 运行控制
    st.header("运行控制")
    st.caption(f"输出目录: {output_dir}")

    if st.button("▶️ 开始处理", type="primary", use_container_width=True):
        # 检查是否有文件
        if not files:
            st.error("没有可处理的文件，请先上传文档。")
        else:
            # 检查输出目录合法性
            output_path = Path(output_dir).resolve()
            input_path = Path(INPUT_DIR).resolve()
            root_path = Path(ROOT_DIR).resolve()

            # 安全检查
            if output_path == root_path:
                st.error("输出目录不能设置为项目根目录")
            elif str(output_path).startswith(str(input_path)):
                st.error("输出目录不能设置在 input_docs/ 之下")
            else:
                # 创建目录
                try:
                    output_path.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    st.error(f"输出目录无权限写入: {output_dir}")
                    output_path = None

                if output_path:
                    with st.spinner("流水线正在运行，请稍候..."):
                        try:
                            result = run_pipeline(str(output_path))
                            st.success("处理完成！")

                            # 显示统计
                            st.json({
                                "扫描文件数": result["files_scanned"],
                                "Office 转换": result["files_converted"],
                                "PDF 提取": result["files_extracted"],
                                "TXT/MD 读取": result["files_text_read"],
                                "清洗章节数": result["sections_cleaned"],
                                "切分 Chunk 数": result["chunks_generated"],
                                "质量过滤": f"{result['chunks_kept']} 保留 / {result['chunks_filtered']} 过滤",
                                "去重前行数": result["lines_before_dedup"],
                                "精确去重后": result["lines_after_rule_dedup"],
                                "输出记录数": result["output_records"],
                                "总耗时(秒)": round(result["duration"], 1),
                                "错误数": len(result["errors"]),
                            })

                            if result["errors"]:
                                st.error("处理过程中出现以下错误：")
                                for err in result["errors"]:
                                    st.error(f"  - {err}")

                            st.rerun()
                        except Exception as e:
                            st.error(f"流水线运行失败：{e}")
                            import traceback
                            st.code(traceback.format_exc())

    st.caption("流水线运行期间请勿关闭页面。运行完成后自动刷新。")

    st.divider()

    # 清理区
    st.header("清理")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ 清空输出结果", use_container_width=True):
            st.warning("此操作将清空所有输出文件（不含日志和报告）。")
            if st.checkbox("确认清空输出", key="confirm_cleanup"):
                cleanup_output(output_dir)
                st.success("输出结果已清空")
                st.rerun()

    with c2:
        if st.button("📋 清空断点记录", use_container_width=True):
            st.warning("清空后所有文件将被重新处理。")
            if st.checkbox("确认清空断点", key="confirm_checkpoint"):
                cleanup_checkpoint(output_dir)
                st.success("断点记录已清空")
                st.rerun()

    if st.button("📦 打包输出为 ZIP"):
        zip_path = create_output_zip(output_dir)
        if zip_path:
            st.success("ZIP 打包完成，请在下方下载。")
            st.rerun()
        else:
            st.error("打包失败")

    with col_right:
        st.header("运行概览")
        st.info("全功能镜像会在容器内使用 MinerU/magic-pdf。首次运行会下载或加载模型，请优先挂载 models_cache。")
        st.markdown(
            f"""
            - PDF 引擎：`{runtime_status['pdf_engine']}`
            - GPU：`{runtime_status['gpu']}`
            - PyTorch：`{runtime_status['torch']}`
            - Checkpoint：`output/processed_files.json`
            """
        )
        report = get_pipeline_report(output_dir)
        st.text_area("报告内容", value=report, height=360, disabled=True, label_visibility="collapsed", key="overview_report")

# ---- 日志 + 结果 ----
with tab_observe:
    # 日志区
    st.header("运行日志")

    log_content, log_name = get_latest_log(output_dir)
    if log_name:
        st.caption(f"最新日志：{log_name}")

    # 提供刷新按钮
    if st.button("🔄 刷新日志"):
        st.rerun()

    # 过滤日志
    log_filter = st.radio(
        "日志过滤",
        ["全部", "仅错误", "仅 WARNING 及以上"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if log_filter == "仅错误":
        log_lines = [l for l in log_content.split("\n") if "ERROR" in l or "error" in l.lower()]
        log_content = "\n".join(log_lines)
    elif log_filter == "仅 WARNING 及以上":
        log_lines = [l for l in log_content.split("\n") if any(x in l for x in ["WARNING", "ERROR", "CRITICAL"])]
        log_content = "\n".join(log_lines)

    st.text_area("", value=log_content, height=350, disabled=True, label_visibility="collapsed", key="log_text")

    st.divider()

    # 结果统计
    st.header("处理结果")

    report = get_pipeline_report(output_dir)
    st.text_area("报告内容", value=report, height=360, disabled=True, label_visibility="collapsed", key="report_text")

    # 质量报告
    quality = get_quality_report(output_dir)
    if quality:
        st.subheader("质量过滤报告")
        st.info(f"被过滤的 chunk 数：**{quality['total_filtered']}**，平均分：{quality['avg_score']:.3f}")
        if quality["sample"]:
            with st.expander("查看被过滤样例"):
                for s in quality["sample"]:
                    st.markdown(f"**{s.get('section_title', '')}** | score={s.get('quality_score', 0):.3f}")
                    st.caption(s.get("content_preview", "")[:200])

    st.divider()

# ---- 下载 ----
with tab_download:
    st.header("文件下载")

    downloadable = get_downloadable_files(output_dir)
    if downloadable:
        for dtype in ["jsonl", "json", "txt", "zip"]:
            type_files = [f for f in downloadable if f["type"] == dtype]
            if type_files:
                type_labels = {"jsonl": "JSONL 语料", "json": "JSON 数据", "txt": "文本文件", "zip": "完整输出"}
                st.subheader(type_labels.get(dtype, dtype))
                for df in type_files:
                    download_file(df["path"], df["name"])
    else:
        st.info("暂无可下载文件，请先运行处理流程。")

    # 打包下载
    if st.button("📦 重新打包并下载"):
        zip_path = create_output_zip(output_dir)
        if zip_path and os.path.exists(zip_path):
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="📥 下载完整输出 (ZIP)",
                    data=f.read(),
                    file_name="corpus_output.zip",
                    mime="application/zip",
                )
        else:
            st.error("打包失败")


# ===== 页脚 =====
st.divider()
if MAGIC_PDF_AVAILABLE:
    st.caption("文档语料库构建系统 | 基于 MinerU / magic-pdf | `python main.py` 命令行模式仍然可用")
else:
    st.caption("文档语料库构建系统 | **CPU 轻量版**（不含 MinerU/PDF 提取）| `python main.py` 命令行模式仍然可用")
