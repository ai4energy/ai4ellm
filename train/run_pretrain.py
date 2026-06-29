"""
Qwen3.6-14B LoRA 继续预训练脚本

基于 corpus_pipeline 产出的 pretrain.jsonl（每行 {"text": ..., "meta": {...}}），
用 TRL SFTTrainer + PEFT LoRA 做继续预训练。

容器内运行：
    python /app/train/run_pretrain.py --config /app/train/configs/pretrain.yaml

宿主机调试（需 torch 环境）：
    python train/run_pretrain.py --config train/configs/pretrain.yaml
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path

import yaml


def parse_args():
    ap = argparse.ArgumentParser(description="Qwen3.6 LoRA 继续预训练")
    ap.add_argument("--config", required=True, help="YAML 配置文件路径")
    return ap.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_model_path(cfg: dict) -> str:
    """解析真实模型路径：HF snapshot 用 hash 命名，自动取第一个 snapshot。"""
    m = cfg["model"]
    base = m["name_or_path"]
    if m.get("snapshot_dir"):
        return m["snapshot_dir"]

    # base 可能指向 models--X/snapshots 目录，自动取里面的第一个 snapshot
    if os.path.isdir(base):
        snapshots = sorted(glob.glob(os.path.join(base, "*")))
        snapshots = [s for s in snapshots if os.path.isdir(s)]
        if snapshots:
            # 校验 snapshot 内有 config.json
            for s in snapshots:
                if os.path.exists(os.path.join(s, "config.json")):
                    print(f"[model] 用 snapshot: {s}")
                    return s
            print(f"[model] snapshots 内无 config.json，用第一个: {snapshots[0]}")
            return snapshots[0]
    return base


def build_dataset(cfg, tokenizer):
    """
    从 pretrain.jsonl 加载，返回 HuggingFace Dataset（只含 text 列）。
    预训练：每个 chunk 作为独立文本序列，不做 instruction 格式。
    """
    from datasets import load_dataset

    data_cfg = cfg["data"]
    pretrain_file = data_cfg["pretrain_file"]
    text_field = data_cfg.get("text_field", "text")

    print(f"[data] 加载预训练数据: {pretrain_file}")

    # rich 格式每行 {"text":..., "meta":...}；standard 格式每行 {"text":...}
    # load_dataset 的 json 能直接读 jsonl
    ds = load_dataset("json", data_files=pretrain_file, split="train")

    # 确保只有 text 列（预训练不消费 meta）
    if text_field in ds.column_names:
        # 丢弃其他列
        keep = [text_field]
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])

    print(f"[data] 样本数: {len(ds)}")
    print(f"[data] 列: {ds.column_names}")
    # 打印一条样本预览
    sample_text = ds[0][text_field][:200]
    print(f"[data] 样本预览: {sample_text!r}...")

    # packing 由 SFTConfig 控制，这里只返回原始 text
    return ds


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 延迟导入（--help 不触发重型依赖）
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_path = resolve_model_path(cfg)
    tok_path = cfg["model"].get("tokenizer_name_or_path") or model_path
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(cfg["model"].get("torch_dtype", "bfloat16"), torch.bfloat16)
    attn_impl = cfg["model"].get("attn_implementation", "sdpa")

    # ── tokenizer ──
    print(f"[tokenizer] 加载: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Qwen 系列 pad token 设置
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── model ──
    qcfg = cfg.get("quantization", {})
    use_4bit = qcfg.get("enabled") and qcfg.get("load_in_4bit", False)

    print(f"[model] 加载: {model_path} (dtype={dtype}, attn={attn_impl}, 4bit={use_4bit})")
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_map.get(
                qcfg.get("bnb_4bit_compute_dtype", "bfloat16"), torch.bfloat16),
            bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=qcfg.get("bnb_4bit_use_double_quant", True),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_cfg,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        # prepare_model_for_kbit_training 会把量化层输入 cast 到 float32，
        # 对 14B+ 模型吃显存；用 use_gradient_checkpointing 让其配合 grad ckpt。
        # 若仍 OOM，可设 quantization.prepare_kbit=false 跳过（LoRA 也能训，
        # 但数值稳定性略差）。
        if qcfg.get("prepare_kbit", True):
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg["training"].get(
                    "gradient_checkpointing", False))
        else:
            if cfg["training"].get("gradient_checkpointing", False):
                model.config.use_cache = False
                model.gradient_checkpointing_enable()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )

    # ── gradient checkpointing ──
    if cfg["training"].get("gradient_checkpointing", False):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    # ── LoRA ──
    lora_cfg_dict = cfg.get("lora", {})
    if lora_cfg_dict.get("enabled", False):
        lora_config = LoraConfig(
            r=lora_cfg_dict.get("r", 64),
            lora_alpha=lora_cfg_dict.get("lora_alpha", 128),
            lora_dropout=lora_cfg_dict.get("lora_dropout", 0.05),
            target_modules=lora_cfg_dict.get("target_modules",
                                             ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias=lora_cfg_dict.get("bias", "none"),
            task_type=lora_cfg_dict.get("task_type", "CAUSAL_LM"),
        )
        if use_4bit:
            model = get_peft_model(model, lora_config)
        else:
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # 全参微调：冻结可选
        for p in model.parameters():
            p.requires_grad = True

    # ── dataset ──
    dataset = build_dataset(cfg, tokenizer)

    # ── SFTConfig ──
    tcfg = cfg["training"]
    sft_config = SFTConfig(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg.get("num_train_epochs", 1),
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 16),
        learning_rate=tcfg.get("learning_rate", 2e-4),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=tcfg.get("warmup_ratio", 0.03),
        weight_decay=tcfg.get("weight_decay", 0.0),
        max_grad_norm=tcfg.get("max_grad_norm", 1.0),
        logging_steps=tcfg.get("logging_steps", 10),
        save_strategy=tcfg.get("save_strategy", "steps"),
        save_steps=tcfg.get("save_steps", 500),
        save_total_limit=tcfg.get("save_total_limit", 3),
        bf16=tcfg.get("bf16", True),
        gradient_checkpointing=tcfg.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs=tcfg.get("gradient_checkpointing_kwargs",
                                                {"use_reentrant": False}),
        report_to=tcfg.get("report_to", "tensorboard"),
        dataloader_num_workers=tcfg.get("dataloader_num_workers", 4),
        seed=tcfg.get("seed", 42),
        max_length=cfg["data"].get("max_length", 2048),
        packing=cfg["data"].get("packing", True),
        dataset_text_field=cfg["data"].get("text_field", "text"),
        # 预训练：不设 response template（不是 instruction tuning）
        completion_only_loss=False,
    )

    # ── trainer ──
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # ── resume ──
    resume_ckpt = cfg.get("resume", {}).get("resume_from_checkpoint")
    if resume_ckpt:
        print(f"[resume] 从 checkpoint 恢复: {resume_ckpt}")

    # ── 训练 ──
    print("[train] 开始训练...")
    trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else None)

    # ── 保存 ──
    final_dir = os.path.join(tcfg["output_dir"], "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[done] 模型已保存到: {final_dir}")


if __name__ == "__main__":
    main()
