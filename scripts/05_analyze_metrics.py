#!/usr/bin/env python3
"""汇总 runs/*-qlora-smoke/metrics.jsonl，输出 stats.json 与文本摘要（std-lib only）。"""
import json
import statistics
import sys
from pathlib import Path


def main() -> None:
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/2026-11-02-qlora-smoke")
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line]
    if not rows:
        raise SystemExit("empty metrics")
    loss = [r["loss"] for r in rows]
    steps = [r["step"] for r in rows]
    times = [r["step_time_s"] for r in rows]
    tps = [r["tokens_per_s"] for r in rows]
    xs = list(range(len(loss)))
    mx, my = sum(xs) / len(xs), sum(loss) / len(loss)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, loss)) / sum((x - mx) ** 2 for x in xs)
    # 前后各 10 步均值，避免单步噪声
    k = min(10, len(loss) // 2)
    stats = {
        "run_dir": str(run_dir),
        "steps": len(rows),
        "first_step": steps[0],
        "last_step": steps[-1],
        "loss_first": loss[0],
        "loss_last": loss[-1],
        "loss_min": min(loss),
        "loss_max": max(loss),
        "loss_mean_first10": round(statistics.mean(loss[:k]), 6),
        "loss_mean_last10": round(statistics.mean(loss[-k:]), 6),
        "loss_slope_per_step": slope,
        "loss_finite_all": all(x == x and abs(x) != float("inf") for x in loss),
        "step_time_s_mean": round(statistics.mean(times), 3),
        "step_time_s_median": round(statistics.median(times), 3),
        "step_time_s_p95": round(sorted(times)[int(0.95 * (len(times) - 1))], 3),
        "step_time_s_max": round(max(times), 3),
        "tokens_per_s_mean": round(statistics.mean(tps), 1),
        "tokens_per_s_median": round(statistics.median(tps), 1),
        "gpu_peak_gib": max(r["gpu_peak_gib"] for r in rows),
        "total_tokens": sum(r["tokens"] for r in rows),
    }
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
