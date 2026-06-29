"""
流水线编排模块

总调度器：协调文件扫描、文档提取、元数据提取、章节清洗、结构化切块、
质量评分过滤、chunk 级规则去重、（可选）语义去重、富格式导出、预训练格式化。
每个步骤可独立开关，单文件失败不影响整体流程。

相对原 zip 版的改动：
- rule_dedup 改为 chunk 级（原版在 all_merged.txt 行级做，下游不消费 = 孤儿）
- 移除 legacy split / merge_txt 主路径（结构化切块为唯一路径）
- 新增预训练格式化步骤
"""

import os
import glob
import json
import time

from src.config import PipelineConfig
from src.file_utils import scan_files, ensure_dir, ResumeTracker, resolve_checkpoint_path
from src.logger import get_logger, setup_logger
from src.cleaners.markdown_cleaner import process_markdown_files
from src.cleaners.text_cleaner import clean_text_file
from src.cleaners.quality_scorer import filter_chunks, save_filtered_report
from src.splitters.text_splitter import split_sections
from src.exporters.jsonl_exporter import export_to_jsonl
from src.exporters.json_exporter import export_to_json
from src.pretrain.format import chunks_to_pretrain_jsonl, content_hash, build_category_map


class PipelineStats:
    """流水线运行统计信息。"""

    def __init__(self):
        self.files_scanned = 0
        self.files_converted = 0
        self.files_extracted = 0
        self.files_text_read = 0
        self.sections_cleaned = 0
        self.chunks_generated = 0
        self.chunks_kept = 0
        self.chunks_filtered = 0
        self.chunks_after_rule_dedup = 0
        self.chunks_after_semantic_dedup = 0
        self.output_records = 0
        self.pretrain_records = 0
        self.start_time = None
        self.end_time = None
        self.errors = []

    def generate_report(self) -> str:
        duration = ""
        if self.start_time and self.end_time:
            duration = f"总耗时: {self.end_time - self.start_time:.1f} 秒"

        quality_info = ""
        if self.chunks_filtered > 0:
            quality_info = f"\n  质量过滤:           {self.chunks_generated} → {self.chunks_kept}（过滤 {self.chunks_filtered}）"

        report = f"""
{'=' * 60}
  语料库构建流水线 统计报告
{'=' * 60}
  扫描文件数:            {self.files_scanned}
  Office 转换成功:       {self.files_converted}
  PDF 提取成功:          {self.files_extracted}
  TXT/MD 读取成功:       {self.files_text_read}
  清洗章节数:            {self.sections_cleaned}
  切分 Chunk 数:         {self.chunks_generated}{quality_info}
  chunk 级规则去重后:    {self.chunks_after_rule_dedup}
  语义去重后记录数:      {self.chunks_after_semantic_dedup}
  富 JSONL 输出记录数:   {self.output_records}
  预训练 JSONL 记录数:   {self.pretrain_records}
  错误数:                {len(self.errors)}
  {duration}
{'=' * 60}"""
        if self.errors:
            report += "\n错误详情:\n"
            for err in self.errors[:20]:
                report += f"  - {err}\n"
            if len(self.errors) > 20:
                report += f"  ... 还有 {len(self.errors) - 20} 条错误\n"
        return report


class CorpusPipeline:
    """语料库构建流水线总调度器。"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = setup_logger(config.get_logging_config())
        paths = config.get_paths()
        checkpoint_name = paths.get("processed_checkpoint", "processed_files.json")
        self.tracker = ResumeTracker(resolve_checkpoint_path(paths["output_dir"], checkpoint_name))
        self.stats = PipelineStats()

    def run(self, input_dir: str | None = None) -> PipelineStats:
        self.stats.start_time = time.time()
        input_dir = input_dir or self.config.get_paths()["input_dir"]
        output_dir = self.config.get_paths()["output_dir"]
        steps = self.config.get_pipeline_steps()

        for subdir in ["extracted_text", "cleaned_text", "jsonl", "json",
                       "chunks", "logs", "reports", "metadata"]:
            ensure_dir(os.path.join(output_dir, subdir))

        self.logger.info("=" * 60)
        self.logger.info("语料库构建流水线启动")
        self.logger.info(f"输入目录: {os.path.abspath(input_dir)}")
        self.logger.info(f"输出目录: {os.path.abspath(output_dir)}")
        self.logger.info("=" * 60)

        # Step 0: 扫描文件
        all_files = self._scan_files(input_dir)
        if not all_files:
            self.logger.warning("未找到任何可处理的文件")
            return self.stats

        pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]
        office_files = [f for f in all_files if f.lower().endswith((".doc", ".docx", ".ppt", ".pptx"))]
        text_files = [f for f in all_files if f.lower().endswith((".txt", ".md"))]

        extracted_dir = os.path.join(output_dir, "extracted_text")
        metadata_dir = os.path.join(output_dir, "metadata")

        # Step 1: Office 转换 → extracted_text/
        if steps.get("convert_office", False) and office_files:
            self._step_convert_office(office_files, extracted_dir, metadata_dir)

        # Step 2: PDF 提取 → extracted_text/
        if steps.get("extract_pdf", False) and pdf_files:
            self._step_extract_pdf(pdf_files, extracted_dir, metadata_dir)

        # Step 3: TXT/MD 直接读取 → extracted_text/
        if steps.get("extract_text", False) and text_files:
            self._step_extract_text(text_files, extracted_dir, metadata_dir)

        # Step 4: 清洗 extracted_text/ → cleaned_text/（含 _sections.json）
        if steps.get("clean_markdown", False):
            self._step_clean_all(output_dir)

        # Step 5: 结构化切分（从 _sections.json）
        chunks = []
        if steps.get("export", False) or steps.get("pretrain", False):
            chunks = self._step_split_structured(output_dir)

        # Step 6: 质量评分过滤
        if steps.get("quality_filter", False) and chunks:
            chunks = self._step_quality_filter(output_dir, chunks)

        # Step 7: chunk 级规则去重（修复原 zip 版孤儿问题）
        if steps.get("rule_dedup", False) and chunks:
            chunks = self._step_rule_dedup_chunks(output_dir, chunks)

        # Step 8: chunk 级语义去重（可选）
        if steps.get("semantic_dedup", False) and chunks:
            chunks = self._step_semantic_dedup_chunks(output_dir, chunks)

        # Step 9: 导出富 JSONL/JSON
        if steps.get("export", False) and chunks:
            self._step_export_from_chunks(output_dir, chunks)

        # Step 10: 预训练格式化
        if steps.get("pretrain", False) and chunks:
            self._step_pretrain(output_dir, chunks, input_dir)

        # 保存断点
        self.tracker.save()

        self.stats.end_time = time.time()
        report = self.stats.generate_report()
        self.logger.info(report)

        report_path = os.path.join(output_dir, "reports", "pipeline_report.txt")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
        except OSError:
            pass

        return self.stats

    # ---- 各步骤实现 ----

    def _scan_files(self, input_dir: str) -> list[str]:
        extensions = self.config.get_scan_config().get("supported_extensions", [".pdf"])
        recursive = self.config.get_scan_config().get("recursive", True)
        files = scan_files(input_dir, extensions, recursive)
        self.stats.files_scanned = len(files)
        self.logger.info(f"扫描到 {len(files)} 个文件")
        return files

    def _save_file_metadata(self, file_path: str, metadata_dir: str):
        try:
            from src.extractors.metadata_extractor import extract_file_metadata, save_metadata
            meta = extract_file_metadata(file_path)
            save_metadata(meta, metadata_dir)
        except Exception as e:
            self.logger.warning(f"元数据提取失败 {file_path}: {e}")

    def _step_convert_office(self, office_files, extracted_dir, metadata_dir):
        self.logger.info("步骤 1: Office 文档转换")
        from src.extractors.office_extractor import OfficeExtractor
        extractor = OfficeExtractor(self.config.get_extractor_config().get("office", {}))
        for f in office_files:
            try:
                if self.tracker.is_stage_completed(f, "convert_office"):
                    self.logger.info(f"跳过已转换: {f}")
                    self.stats.files_converted += 1
                    continue
                result = extractor.extract(f, extracted_dir)
                if result:
                    self.stats.files_converted += 1
                    self.tracker.mark_processed(f, "convert_office", result)
                    self._save_file_metadata(f, metadata_dir)
            except Exception as e:
                self.tracker.mark_failed(f, "convert_office", str(e))
                self.stats.errors.append(f"Office 转换失败 {f}: {e}")
                self.logger.error(f"Office 转换失败 {f}: {e}")

    def _step_extract_pdf(self, pdf_files, extracted_dir, metadata_dir):
        self.logger.info("步骤 2: PDF 文本提取")
        from src.extractors.pdf_extractor import process_folder_parallel
        pdf_config = self.config.get_extractor_config().get("pdf", {})

        pending = [f for f in pdf_files if not self.tracker.is_stage_completed(f, "extract_pdf")]
        already_done = [f for f in pdf_files if self.tracker.is_stage_completed(f, "extract_pdf")]
        self.stats.files_extracted = len(already_done)
        for f in already_done:
            self.logger.info(f"跳过已提取 PDF: {f}")
        if pending:
            self.logger.info(f"待提取 PDF: {len(pending)} 个")

        if pending:
            results = process_folder_parallel(pending, extracted_dir, pdf_config)
            for pdf_file, output, error in results:
                if output:
                    self.stats.files_extracted += 1
                    self.tracker.mark_processed(pdf_file, "extract_pdf", output)
                    self._save_file_metadata(pdf_file, metadata_dir)
                else:
                    detail = error or "提取返回 None"
                    self.tracker.mark_failed(pdf_file, "extract_pdf", detail)
                    self.stats.errors.append(f"PDF 提取失败: {pdf_file} | {detail}")
                    self.logger.error(f"PDF 提取失败: {pdf_file} | {detail}")

    def _step_extract_text(self, text_files, extracted_dir, metadata_dir):
        self.logger.info("步骤 3: TXT/MD 文本读取")
        from src.extractors.text_extractor import TextExtractor
        extractor = TextExtractor({})
        for f in text_files:
            try:
                if self.tracker.is_stage_completed(f, "extract_text"):
                    self.logger.info(f"跳过已读取: {f}")
                    self.stats.files_text_read += 1
                    continue
                result = extractor.extract(f, extracted_dir)
                if result:
                    self.stats.files_text_read += 1
                    self.tracker.mark_processed(f, "extract_text", result)
                    self._save_file_metadata(f, metadata_dir)
            except Exception as e:
                self.tracker.mark_failed(f, "extract_text", str(e))
                self.stats.errors.append(f"文本读取失败 {f}: {e}")
                self.logger.error(f"文本读取失败 {f}: {e}")

    def _step_clean_all(self, output_dir: str):
        self.logger.info("步骤 4: 文本清洗")
        extracted_dir = os.path.join(output_dir, "extracted_text")
        cleaned_dir = os.path.join(output_dir, "cleaned_text")
        cleaner_config = self.config.get_cleaner_config()

        if not os.path.isdir(extracted_dir):
            self.logger.warning("提取目录不存在，跳过清洗")
            return

        md_files = glob.glob(os.path.join(extracted_dir, "*.md"))
        if md_files:
            self.logger.info(f"清洗 {len(md_files)} 个 Markdown 文件")
            process_markdown_files(
                input_dir=extracted_dir,
                output_dir=cleaned_dir,
                skip_sections=set(cleaner_config.get("skip_sections", [])),
                min_body_chars=cleaner_config.get("min_body_chars", 50),
                min_final_body_chars=cleaner_config.get("min_final_body_chars", 150),
                remove_garbled=cleaner_config.get("remove_garbled", True),
                clean_refs=cleaner_config.get("clean_refs", True),
                watermark_keywords=cleaner_config.get("watermark_keywords", []),
            )

        txt_files = glob.glob(os.path.join(extracted_dir, "*.txt"))
        if txt_files:
            self.logger.info(f"清洗 {len(txt_files)} 个纯文本文件")
            for tf in txt_files:
                out_file = os.path.join(cleaned_dir, f"{os.path.splitext(os.path.basename(tf))[0]}_cleaned.txt")
                clean_text_file(
                    tf, out_file,
                    watermark_keywords=cleaner_config.get("watermark_keywords", []),
                    fix_broken_lines=cleaner_config.get("fix_broken_lines", True),
                )

        cleaned_files = glob.glob(os.path.join(cleaned_dir, "*_sections.json"))
        self.stats.sections_cleaned = len(cleaned_files)
        self.logger.info(f"清洗完成: {self.stats.sections_cleaned} 个结构化文件")

    def _step_split_structured(self, output_dir: str) -> list[dict]:
        self.logger.info("步骤 5: 结构化文本切分")
        cleaned_dir = os.path.join(output_dir, "cleaned_text")
        chunks_dir = os.path.join(output_dir, "chunks")
        splitter_config = self.config.get_splitter_config()

        if not os.path.isdir(cleaned_dir):
            self.logger.warning("清洗目录不存在，跳过切分")
            return []

        sections_files = glob.glob(os.path.join(cleaned_dir, "*_sections.json"))
        if not sections_files:
            self.logger.warning("未找到 _sections.json 文件，无 chunk 可生成")
            return []

        chunks = split_sections(
            cleaned_dir=cleaned_dir,
            output_dir=chunks_dir,
            max_segment_chars=splitter_config.get("max_segment_chars", 500),
            min_segment_chars=splitter_config.get("min_segment_chars", 50),
        )
        self.stats.chunks_generated = len(chunks)
        self.logger.info(f"结构化切分完成: {len(chunks)} 个 chunk")
        return chunks

    def _step_quality_filter(self, output_dir: str, chunks: list[dict]) -> list[dict]:
        self.logger.info("步骤 6: 质量评分过滤")
        quality_config = self.config.get_quality_config()
        if not quality_config.get("enabled", True):
            self.logger.info("质量评分未启用，跳过过滤")
            return chunks

        filter_enabled = quality_config.get("filter_enabled", False)
        min_score = quality_config.get("min_score", 0.3)
        max_noise = quality_config.get("max_noise_ratio", 0.1)

        if filter_enabled:
            kept, filtered = filter_chunks(chunks, min_score=min_score, max_noise_ratio=max_noise)
        else:
            kept, _ = filter_chunks(chunks, min_score=0.0, max_noise_ratio=1.0)
            filtered = []

        self.stats.chunks_kept = len(kept)
        self.stats.chunks_filtered = len(filtered)

        if filtered:
            report_path = os.path.join(output_dir, "reports", "filtered_chunks.json")
            save_filtered_report(filtered, report_path)

        if filter_enabled:
            self.logger.info(f"质量过滤: {len(chunks)} → {len(kept)}（过滤 {len(filtered)}）")
        else:
            self.logger.info(f"质量评分完成: {len(chunks)} 个 chunk 已评分，未删除低质量文本")
        return kept

    def _step_rule_dedup_chunks(self, output_dir: str, chunks: list[dict]) -> list[dict]:
        """步骤 7: chunk 级规则去重（归一化 hash，修复原 zip 版行级孤儿问题）。"""
        self.logger.info("步骤 7: chunk 级规则去重")
        rule_config = self.config.get_dedup_config().get("rule_dedup", {})
        if not rule_config.get("enabled", True):
            self.stats.chunks_after_rule_dedup = len(chunks)
            return chunks

        case_sensitive = rule_config.get("case_sensitive", False)
        seen = set()
        kept = []
        for chunk in chunks:
            content = (chunk.get("content") or "").strip()
            if not content:
                continue
            key = content if case_sensitive else content_hash(content)
            if key in seen:
                continue
            seen.add(key)
            kept.append(chunk)

        removed = len(chunks) - len(kept)
        self.stats.chunks_after_rule_dedup = len(kept)
        self.logger.info(f"chunk 规则去重: {len(chunks)} → {len(kept)}（移除 {removed}）")
        return kept

    def _step_semantic_dedup_chunks(self, output_dir: str, chunks: list[dict]) -> list[dict]:
        self.logger.info("步骤 8: chunk 级语义去重")
        from src.dedup.semantic_dedup import semantic_deduplicate_chunks  # 延迟导入：避免 numpy 强依赖
        semantic_config = self.config.get_dedup_config().get("semantic_dedup", {})
        report_path = os.path.join(output_dir, "reports", "semantic_filtered_chunks.json")
        kept, filtered = semantic_deduplicate_chunks(
            chunks=chunks,
            model_name=semantic_config.get("model_name", "shibing624/text2vec-base-chinese"),
            similarity_threshold=semantic_config.get("similarity_threshold", 0.9),
            batch_size=semantic_config.get("batch_size", 1024),
            device=semantic_config.get("device", "auto"),
            report_path=report_path,
        )
        self.stats.chunks_after_semantic_dedup = len(kept)
        if filtered:
            self.logger.info(f"语义去重过滤报告已保存: {report_path}")
        self.logger.info(f"chunk 语义去重: {len(chunks)} → {len(kept)}")
        return kept

    def _step_export_from_chunks(self, output_dir: str, chunks: list[dict]):
        self.logger.info("步骤 9: 导出富格式语料")
        exporter_config = self.config.get_exporter_config()
        formats = exporter_config.get("formats", ["jsonl"])
        chunks_dir = os.path.join(output_dir, "chunks")

        chunks_json_path = os.path.join(chunks_dir, "_chunks.json")
        if chunks:
            with open(chunks_json_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

        if "jsonl" in formats:
            jsonl_dir = os.path.join(output_dir, "jsonl")
            total = export_to_jsonl(
                chunks_dir, jsonl_dir,
                ensure_ascii=exporter_config.get("jsonl", {}).get("ensure_ascii", False),
            )
            self.stats.output_records += total
        if "json" in formats:
            json_file = os.path.join(output_dir, "json", "corpus.json")
            total = export_to_json(
                chunks_dir, json_file,
                ensure_ascii=exporter_config.get("json", {}).get("ensure_ascii", False),
                indent=exporter_config.get("json", {}).get("indent", 2),
            )
            self.stats.output_records += total

    def _step_pretrain(self, output_dir: str, chunks: list[dict], input_dir: str):
        """步骤 10: 预训练格式化（富 chunk → 预训练 JSONL）。"""
        self.logger.info("步骤 10: 预训练格式化")
        pretrain_config = self.config.raw.get("pretrain", {})
        if not pretrain_config.get("enabled", True):
            return

        # 从输入目录扫描构建 category_map（stem→category），
        # 让 chunk 的 source_file（纯 stem）能反查到 category。
        category_map = build_category_map(input_dir)
        self.logger.info(f"构建 category_map: {len(category_map)} 个文件 → 类目")

        output_file = os.path.join(output_dir, pretrain_config.get("output_file", "pretrain.jsonl"))
        stats = chunks_to_pretrain_jsonl(
            chunks=chunks,
            output_path=output_file,
            fmt=pretrain_config.get("format", "rich"),
            min_length=pretrain_config.get("min_length", 100),
            max_length=pretrain_config.get("max_length", 1000000),
            include_section_prefix=pretrain_config.get("include_section_prefix", True),
            dedup=False,  # rule_dedup 已在步骤 7 做过
            input_root=input_dir,
            category_map=category_map,
        )
        self.stats.pretrain_records = stats["written"]
        self.logger.info(f"预训练 JSONL: {output_file} ({stats['written']} 条)")
