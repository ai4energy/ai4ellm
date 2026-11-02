#!/usr/bin/env python3
"""P0 §4: Qwen3.8-27B 4bit QLoRA 100 步冒烟（单卡 GPU2）。

用法（容器内）:
  python /work/train/qlora_smoke.py --config /work/train/qlora_smoke.yaml
  python /work/train/qlora_smoke.py --config ... --verify-only

判据: 100 步不 OOM、loss 有限且整体下降、adapter 保存并能被 peft 重新加载。
所有指标写入 output_dir/metrics.jsonl 与 summary.json。
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

from loss_utils import causal_lm_loss_chunked


def log(msg: str) -> None:
    # torchrun 下只让 rank0 打印，避免多进程日志交错
    if os.environ.get("RANK", "0") != "0":
        return
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty dataset: {path}")
    return rows


def tokenize_rows(tokenizer, rows: list[dict], max_length: int) -> list[dict]:
    samples = []
    for row in rows:
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        ids = tokenizer(
            text, truncation=True, max_length=max_length, padding=False
        )["input_ids"]
        if len(ids) < 8:
            continue
        samples.append({"input_ids": ids, "labels": list(ids)})
    if not samples:
        raise RuntimeError("no usable samples after tokenization")
    return samples


def collate(sample: dict, device: torch.device) -> dict:
    input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([sample["labels"]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_model(cfg: dict, device_index: int = 0):
    quant = cfg["quant"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bool(quant["load_in_4bit"]),
        bnb_4bit_quant_type=quant["quant_type"],
        bnb_4bit_use_double_quant=bool(quant["double_quant"]),
        bnb_4bit_compute_dtype=getattr(torch, quant["compute_dtype"]),
    )
    log(f"loading base from {cfg['model_path']} (4bit, device_map={{'': {device_index}}})")
    t0 = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg["model_path"],
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map={"": device_index},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    log(f"base loaded in {time.perf_counter() - t0:.1f}s; "
        f"max_memory_allocated={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return model


def apply_lora(model, cfg: dict):
    tr = cfg["train"]
    if tr.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(tr.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lc = cfg["lora"]
    lora_cfg = LoraConfig(
        r=int(lc["r"]),
        lora_alpha=int(lc["alpha"]),
        lora_dropout=float(lc["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lc["target_modules"]),
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"LoRA attached: trainable={trainable:,} ({100 * trainable / total:.3f}% of {total:,})")
    return model


def make_scheduler(optimizer, warmup_steps: int, max_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def sample_batches(samples: list[dict], seed: int, device: torch.device):
    rng = random.Random(seed)
    while True:
        order = list(range(len(samples)))
        rng.shuffle(order)
        for i in order:
            yield collate(samples[i], device)


def train(cfg: dict) -> dict:
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    tr = cfg["train"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = load_rows(cfg["data_path"])
    samples = tokenize_rows(tokenizer, rows, int(tr["max_length"]))
    lens = [len(s["input_ids"]) for s in samples]
    log(f"dataset: {len(samples)} usable rows; tokens min/avg/max="
        f"{min(lens)}/{sum(lens) / len(lens):.0f}/{max(lens)}")

    model = build_model(cfg)
    model = apply_lora(model, cfg)
    device = torch.device("cuda:0")

    params = [p for p in model.parameters() if p.requires_grad]
    # AdamW 8bit：LoRA 参数约 2 亿，fp32 两套动量约 1.6GiB；8bit 省一半以上。
    optimizer = bnb.optim.PagedAdamW8bit(
        params, lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"])
    )
    max_steps = int(tr["max_steps"])
    scheduler = make_scheduler(optimizer, int(tr["warmup_steps"]), max_steps)
    batches = sample_batches(samples, int(cfg["seed"]), device)

    torch.cuda.reset_peak_memory_stats()
    model.train()
    train_start = time.perf_counter()
    losses: list[float] = []
    t_step_start = time.perf_counter()

    with metrics_path.open("w", encoding="utf-8") as mf:
        for step in range(1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            tokens = 0
            for _ in range(int(tr["grad_accum"])):
                batch = next(batches)
                loss = causal_lm_loss_chunked(model, **batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
                (loss / int(tr["grad_accum"])).backward()
                step_loss += loss.item() / int(tr["grad_accum"])
                tokens += batch["input_ids"].numel()

            torch.nn.utils.clip_grad_norm_(params, float(tr["max_grad_norm"]))
            optimizer.step()
            scheduler.step()

            step_time = time.perf_counter() - t_step_start
            t_step_start = time.perf_counter()
            losses.append(step_loss)
            rec = {
                "step": step,
                "loss": round(step_loss, 6),
                "lr": scheduler.get_last_lr()[0],
                "step_time_s": round(step_time, 3),
                "tokens": tokens,
                "tokens_per_s": round(tokens / step_time, 1),
                "gpu_alloc_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
                "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            }
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            if step % int(tr["logging_steps"]) == 0 or step == 1 or step == max_steps:
                log(f"step {step}/{max_steps} loss={rec['loss']:.4f} "
                    f"lr={rec['lr']:.2e} {rec['step_time_s']:.2f}s "
                    f"{rec['tokens_per_s']:.1f} tok/s peak={rec['gpu_peak_gib']:.2f}GiB")

    train_time = time.perf_counter() - train_start
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    log(f"adapter saved -> {adapter_dir}")

    # 同进程 peft 重新加载校验：load_adapter 走 PEFT 的 adapter 读取路径
    reload_loss = None
    try:
        model.load_adapter(adapter_dir, adapter_name="verify_reload")
        model.set_adapter("verify_reload")
        model.eval()
        with torch.no_grad():
            batch = collate(samples[0], device)
            reload_loss = float(causal_lm_loss_chunked(model, **batch).item())
        log(f"adapter reload OK: reload_loss={reload_loss:.6f}")
    except Exception as exc:  # noqa: BLE001 - 记录后继续产出报告
        log(f"adapter reload FAILED: {type(exc).__name__}: {exc}")

    summary = {
        "max_steps": max_steps,
        "grad_accum": int(tr["grad_accum"]),
        "batch_size": int(tr["batch_size"]),
        "max_length": int(tr["max_length"]),
        "train_time_s": round(train_time, 1),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "min_loss": min(losses),
        "loss_decreasing": losses[-1] < losses[0],
        "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "gpu_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "mean_tokens_per_s": round(
            sum(s["tokens"] for s in map(json.loads, metrics_path.read_text().splitlines()))
            / train_time, 1),
        "reload_loss": reload_loss,
        "adapter_dir": str(adapter_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("SUMMARY " + json.dumps(summary, ensure_ascii=False))
    return summary


def verify_only(cfg: dict) -> dict:
    out_dir = Path(cfg["output_dir"])
    adapter_dir = out_dir / "adapter"
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=False)
    rows = load_rows(cfg["data_path"])
    samples = tokenize_rows(tokenizer, rows, int(cfg["train"]["max_length"]))
    model = build_model(cfg)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    n_adapter = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
    with torch.no_grad():
        loss = causal_lm_loss_chunked(
            model, **collate(samples[0], torch.device("cuda:0"))
        ).item()
    result = {"adapter_dir": str(adapter_dir), "lora_tensors": n_adapter,
              "verify_loss": loss, "finite": math.isfinite(loss)}
    log("VERIFY " + json.dumps(result))
    (out_dir / "verify.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="覆盖配置里的 max_steps（dry run 用）")
    ap.add_argument("--output-dir", default=None, help="覆盖配置里的 output_dir")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    log(f"config={args.config} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"device_count={torch.cuda.device_count()}")
    assert torch.cuda.is_available(), "CUDA unavailable"
    cap = torch.cuda.get_device_capability(0)
    log(f"gpu0={torch.cuda.get_device_name(0)} capability={cap}")
    if cap != (12, 0):
        log(f"WARNING: expected sm_120 (12, 0), got {cap}")
    if args.verify_only:
        verify_only(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
