"""低显存 causal LM loss：不在显存里保留 [seq, vocab] 的完整 fp32 logits。

背景: Qwen3.8-27B 词表 248320，seq 2048 时 transformers 默认 loss 会先物化
2048×248320 的 fp32 logits（约 1.89 GiB），在 32GB 5090 上 backward 必 OOM。
这里改为:
  1. 只跑 text model 拿 last_hidden_state（约 21 MiB）；
  2. 按 chunk 过 lm_head 并算 CE，每个 chunk 用 torch.utils.checkpoint 重算，
     使 fp32 logits/softmax 的峰值只与 chunk size 有关，与总序列长度无关。
数学等价于标准 shifted cross-entropy（reduction=sum / 非 ignore token 数）。
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_ce(lm_head, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = lm_head(hidden)  # [B, T_chunk, vocab], bf16
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="sum",
        ignore_index=-100,
    )


def causal_lm_loss_chunked(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    chunk_size: int = 256,
) -> torch.Tensor:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    lm_head = base.lm_head
    text_model = base.model.language_model
    out = text_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden = out.last_hidden_state  # [B, T, H]

    if labels is None:
        labels = input_ids
    shift_hidden = hidden[:, :-1, :]
    shift_labels = labels[:, 1:]
    n_tokens = (shift_labels != -100).sum().clamp(min=1)
    total = torch.zeros((), dtype=torch.float32, device=hidden.device)

    seq_len = shift_hidden.size(1)
    for start in range(0, seq_len, chunk_size):
        h = shift_hidden[:, start:start + chunk_size, :]
        y = shift_labels[:, start:start + chunk_size]
        if torch.is_grad_enabled():
            loss_sum = checkpoint(
                lambda hh, yy: _chunk_ce(lm_head, hh, yy), h, y, use_reentrant=False
            )
        else:
            loss_sum = _chunk_ce(lm_head, h, y)
        total = total + loss_sum
    return total / n_tokens
