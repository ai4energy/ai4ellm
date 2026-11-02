#!/usr/bin/env python3
"""P0 §3 预检: sm_120 + 4bit(bnb) 加载 + 单步前向/反向。

在 GPU2 上加载真实 27B 基座（NF4），挂 LoRA，取最长样本跑一步 forward+backward，
验证 bitsandbytes 在 RTX 5090 (sm_120) / CUDA 13 上可用。
结果写入 OUT_DIR/preflight.json（默认 runs/2026-11-02-env-smoke/）。
"""
import json
import os
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

from loss_utils import causal_lm_loss_chunked
from qlora_smoke import collate, load_rows, tokenize_rows

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/llm/Qwen3.8-27B-bf16")
DATA_PATH = os.environ.get("DATA_PATH", "/work/data/synthetic-50.jsonl")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/work/runs/2026-11-02-env-smoke"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA unavailable"
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    info = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "bitsandbytes": bnb.__version__,
        "device_name": name,
        "capability": list(cap),
        "capability_is_sm120": cap == (12, 0),
    }
    print("PREFLIGHT_ENV " + json.dumps(info), flush=True)
    assert cap == (12, 0), f"expected sm_120 (12,0), got {cap}"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    load_s = time.perf_counter() - t0
    n_4bit = sum(1 for m in model.modules() if isinstance(m, bnb.nn.Linear4bit))
    info["load_seconds"] = round(load_s, 1)
    info["linear4bit_modules"] = n_4bit
    info["gpu_alloc_after_load_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    print("PREFLIGHT_LOAD " + json.dumps(info), flush=True)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                        "down_proj", "in_proj_qkv", "in_proj_z", "in_proj_b",
                        "in_proj_a", "out_proj"],
    ))
    model.train()

    rows = load_rows(DATA_PATH)
    samples = tokenize_rows(tokenizer, rows, 2048)
    sample = max(samples, key=lambda s: len(s["input_ids"]))
    device = torch.device("cuda:0")
    batch = collate(sample, device)
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    loss = causal_lm_loss_chunked(model, **batch)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss: {loss.item()}")
    loss.backward()
    step_s = time.perf_counter() - t0
    grads = [p.grad.detach().norm().item() for p in model.parameters()
             if p.requires_grad and p.grad is not None]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.PagedAdamW8bit(params, lr=1e-4)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    info.update({
        "sample_tokens": len(sample["input_ids"]),
        "loss": round(loss.item(), 6),
        "loss_finite": True,
        "backward_ok": len(grads) > 0 and all(g == g and abs(g) != float("inf") for g in grads),
        "lora_tensors_with_grad": len(grads),
        "grad_norm_sum": round(sum(grads), 6),
        "optimizer": "PagedAdamW8bit",
        "optimizer_step_ok": True,
        "step_seconds": round(step_s, 2),
        "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    })
    print("PREFLIGHT_RESULT " + json.dumps(info), flush=True)
    (OUT_DIR / "preflight.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if not info["backward_ok"]:
        raise RuntimeError("backward produced no finite LoRA gradients")


if __name__ == "__main__":
    main()
