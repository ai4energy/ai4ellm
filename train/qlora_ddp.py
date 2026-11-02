#!/usr/bin/env python3
"""双卡 DDP QLoRA 验证（torchrun --standalone --nproc_per_node=2）。

复用单卡 P0 的模型/LoRA/数据/分块 loss；新增:
  - 每进程 device_map={"": local_rank}
  - DDP(find_unused_parameters=True)（纯文本时 vision tower LoRA 未使用）
  - 数据按 rank 分片，梯度自动 all-reduce
  - rank0 保存 adapter / 写 metrics / 做 peft 重载校验
用法见 scripts/06_run_qlora_ddp.sh。
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from loss_utils import causal_lm_loss_chunked
from qlora_smoke import (
    apply_lora,
    build_model,
    collate,
    load_rows,
    log,
    make_scheduler,
    tokenize_rows,
)


class ChunkedLossModule(torch.nn.Module):
    """把 PeftModel 包成 nn.Module，forward 返回分块 CE loss。

    必须是 DDP.forward 调用到的模块，DDP 才能准备/触发梯度 reducer；
    内部仍只跑 language_model + 分块 lm_head，不物化完整 logits。
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None, labels=None):
        return causal_lm_loss_chunked(self.model, input_ids, attention_mask, labels)


def sample_batches(samples: list[dict], seed: int, rank: int, world: int, device):
    rng = random.Random(seed + rank)
    while True:
        order = list(range(len(samples)))
        rng.shuffle(order)
        if world > 1:
            order = [i for i in order if i % world == rank]
        if not order:
            raise RuntimeError(f"rank {rank} got no data (dataset < world size)")
        for i in order:
            yield collate(samples[i], device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    assert torch.cuda.is_available(), "CUDA unavailable"
    log(f"rank={rank}/{world} local_rank={local_rank} "
        f"gpu={torch.cuda.get_device_name(local_rank)} "
        f"capability={torch.cuda.get_device_capability(local_rank)}")

    out_dir = Path(cfg["output_dir"])
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()

    tr = cfg["train"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = load_rows(cfg["data_path"])
    samples = tokenize_rows(tokenizer, rows, int(tr["max_length"]))
    lens = [len(s["input_ids"]) for s in samples]
    log(f"dataset: {len(samples)} usable rows; tokens min/avg/max="
        f"{min(lens)}/{sum(lens) / len(lens):.0f}/{max(lens)}")

    model = build_model(cfg, device_index=local_rank)
    model = apply_lora(model, cfg)
    module = ChunkedLossModule(model)
    if world > 1:
        module = DDP(
            module,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    log("DDP wrapped" if world > 1 else "single-process module")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(
        params, lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"])
    )
    max_steps = int(tr["max_steps"])
    scheduler = make_scheduler(optimizer, int(tr["warmup_steps"]), max_steps)
    batches = sample_batches(samples, int(cfg["seed"]), rank, world, device)

    torch.cuda.reset_peak_memory_stats()
    model.train()
    train_start = time.perf_counter()
    losses: list[float] = []
    total_tokens_logged = 0
    metrics_path = out_dir / "metrics.jsonl"
    mf = metrics_path.open("w", encoding="utf-8") if rank == 0 else None
    t_step_start = time.perf_counter()

    try:
        for step in range(1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            step_loss_local = 0.0
            tokens_local = 0
            for _ in range(int(tr["grad_accum"])):
                batch = next(batches)
                loss = module(**batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
                (loss / int(tr["grad_accum"])).backward()
                step_loss_local += loss.item() / int(tr["grad_accum"])
                tokens_local += batch["input_ids"].numel()

            torch.nn.utils.clip_grad_norm_(params, float(tr["max_grad_norm"]))
            optimizer.step()
            scheduler.step()

            loss_t = torch.tensor(step_loss_local, device=device)
            tok_t = torch.tensor(tokens_local, device=device, dtype=torch.long)
            if world > 1:
                dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(tok_t, op=dist.ReduceOp.SUM)
                loss_t /= world
            step_time = time.perf_counter() - t_step_start
            t_step_start = time.perf_counter()
            step_loss = float(loss_t.item())
            tokens = int(tok_t.item())
            losses.append(step_loss)
            total_tokens_logged += tokens
            rec = {
                "step": step,
                "loss": round(step_loss, 6),
                "lr": scheduler.get_last_lr()[0],
                "step_time_s": round(step_time, 3),
                "tokens": tokens,
                "tokens_per_s": round(tokens / step_time, 1),
                "gpu_alloc_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
                "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "world_size": world,
            }
            if mf is not None:
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
                if step % int(tr["logging_steps"]) == 0 or step in (1, max_steps):
                    log(f"step {step}/{max_steps} loss={rec['loss']:.4f} "
                        f"lr={rec['lr']:.2e} {rec['step_time_s']:.2f}s "
                        f"{rec['tokens_per_s']:.1f} tok/s peak={rec['gpu_peak_gib']:.2f}GiB")
    finally:
        if mf is not None:
            mf.close()

    train_time = time.perf_counter() - train_start
    adapter_dir = out_dir / "adapter"
    if rank == 0:
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        log(f"adapter saved -> {adapter_dir}")
    if world > 1:
        dist.barrier()

    reload_loss = None
    if rank == 0:
        try:
            model.load_adapter(adapter_dir, adapter_name="verify_reload")
            model.set_adapter("verify_reload")
            model.eval()
            with torch.no_grad():
                batch = collate(samples[0], device)
                reload_loss = float(causal_lm_loss_chunked(model, **batch).item())
            log(f"adapter reload OK: reload_loss={reload_loss:.6f}")
        except Exception as exc:  # noqa: BLE001
            log(f"adapter reload FAILED: {type(exc).__name__}: {exc}")

        summary = {
            "world_size": world,
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
            "mean_tokens_per_s": round(total_tokens_logged / train_time, 1),
            "reload_loss": reload_loss,
            "adapter_dir": str(adapter_dir),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log("SUMMARY " + json.dumps(summary, ensure_ascii=False))

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
