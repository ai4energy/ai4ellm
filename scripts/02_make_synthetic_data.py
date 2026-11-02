#!/usr/bin/env python3
"""生成 P0 QLoRA 冒烟用的 50 条合成样本（标注 synthetic）。

输出: data/synthetic-50.jsonl，每行 {"messages":[...], "synthetic":true, "source":"synthetic"}
用途仅限训练管线冒烟，不代表真实数据分布。
"""
import json
import pathlib

SYSTEM = (
    "你是一个能源领域编程助手，熟悉 ESML 建模语言、热泵/蒸汽/空气/水四类 DTIL、"
    "Rust/Julia/Python。回答要给出可执行代码或可验证步骤，不要编造不存在的 API。"
)

TOPICS = [
    ("python_kw_to_btu", "写一个 Python 函数，把热泵制热量从 kW 换算成 Btu/h，并断言输入非负。"),
    ("python_steam_interp", "用 Python 实现蒸汽表的线性插值：给定压力和温度，返回饱和温度与焓值。"),
    ("python_dataset_check", "写一个 Python 校验器，检查 esml.dataset/v1 数据集里每条记录的 unitCoverageComplete。"),
    ("rust_residual", "用 Rust 写一个求解残差检查函数：输入 residual 和 tolerance，返回是否收敛及错误信息。"),
    ("rust_result", "Rust 中如何用 Result 处理求解器不收敛？给出一个带自定义错误类型的例子。"),
    ("julia_esml", "在 Julia 里用 ESMLKit.jl 提交一个热泵循环模型，并读取 converged 与 residual。"),
    ("esml_prv_cv", "解释 ESML 中 PRV 与 CV 的区别，并给出一个水系统定压阀的建模片段。"),
    ("esml_unit_coverage", "一个蒸汽循环模型单位覆盖不完整，列出排查顺序和需要检查的字段。"),
    ("debug_cop", "下面的 Python 代码在计算 COP 时除零：cop = q_out / (q_out - q_in)。给出修复方案。"),
    ("debug_rust_types", "Rust 代码把 f64 残差和整数容差比较导致类型错误，说明如何修复并保持精度。"),
]

BODIES = {
    "python_kw_to_btu": '''```python
def kw_to_btu_per_h(kw: float) -> float:
    if kw < 0:
        raise ValueError("power must be non-negative")
    return kw * 3412.141633
```

1 kW = 3412.141633 Btu/h。测试：
```python
assert abs(kw_to_btu_per_h(1.0) - 3412.141633) < 1e-6
```''',
    "python_steam_interp": '''```python
def interp_sat(p: float, table: list[tuple[float, float, float]]) -> tuple[float, float]:
    """table: [(pressure_kpa, t_sat_c, h_fg_kj_kg), ...]，按压力升序。"""
    if not table:
        raise ValueError("empty steam table")
    if p <= table[0][0]:
        return table[0][1], table[0][2]
    if p >= table[-1][0]:
        return table[-1][1], table[-1][2]
    for (p0, t0, h0), (p1, t1, h1) in zip(table, table[1:]):
        if p0 <= p <= p1:
            a = (p - p0) / (p1 - p0)
            return t0 + a * (t1 - t0), h0 + a * (h1 - h0)
    raise AssertionError("unreachable")
```''',
    "python_dataset_check": '''```python
import json, sys

def check(path: str) -> list[str]:
    errors = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            rec = json.loads(line)
            if rec.get("schema") != "esml.dataset/v1":
                errors.append(f"line {i}: bad schema")
            if not rec.get("unitCoverageComplete", False):
                errors.append(f"line {i}: unit coverage incomplete")
    return errors

if __name__ == "__main__":
    errs = check(sys.argv[1])
    print("\\n".join(errs) if errs else "OK")
    sys.exit(1 if errs else 0)
```''',
    "rust_residual": '''```rust
#[derive(Debug)]
pub enum SolveError {
    NotConverged { residual: f64, tolerance: f64 },
}

pub fn check_converged(residual: f64, tolerance: f64) -> Result<(), SolveError> {
    if !residual.is_finite() {
        return Err(SolveError::NotConverged { residual, tolerance });
    }
    if residual.abs() <= tolerance {
        Ok(())
    } else {
        Err(SolveError::NotConverged { residual, tolerance })
    }
}
```''',
    "rust_result": '''```rust
use std::fmt;

#[derive(Debug)]
pub enum SolverError {
    NotConverged { residual: f64 },
    InvalidInput(String),
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotConverged { residual } => write!(f, "residual {residual}"),
            Self::InvalidInput(s) => write!(f, "invalid input: {s}"),
        }
    }
}

pub fn solve(initial: f64) -> Result<f64, SolverError> {
    if !initial.is_finite() {
        return Err(SolverError::InvalidInput(initial.to_string()));
    }
    Ok(initial)
}
```''',
    "julia_esml": '''```julia
using ESMLKit

model = load_model("heatpump_cycle.esml")
result = solve(model; tolerance = 1e-6)
if result.converged
    @info "converged" residual = result.residual
else
    @warn "not converged" residual = result.residual
end
```''',
    "esml_prv_cv": '''PRV（pressure reducing valve）维持下游定压，CV（control valve）按控制信号调节开度。
```esml
component PRV downstream {
  inlet: WaterPort;
  outlet: WaterPort;
  setpoint: 300 kPa;
  equation: outlet.p = setpoint;
}
```
CV 则通常给定开度或由控制器输出，不要用固定压力方程替代。''',
    "esml_unit_coverage": '''排查顺序：
1. 列出每个端口的 quantity/unit，确认压力、温度、质量流量都带单位；
2. 检查 component 的 equation 是否隐式引入未声明单位；
3. 检查连接处单位是否一致（kPa vs Pa、kg/s vs t/h）；
4. 跑 `esml check` 看缺失字段，再跑 `esml solve` 看残差；
5. 最后用 `unitCoverageComplete` 断言，100% 才收数据。''',
    "debug_cop": '''除零的根因是 `q_out - q_in == 0` 时 COP 无定义。修复：
```python
def cop(q_out: float, q_in: float) -> float:
    if q_in <= 0:
        raise ValueError("q_in must be positive")
    work = q_out - q_in
    if work <= 0:
        raise ValueError("q_out must exceed q_in for a heat pump")
    return q_out / work
```''',
    "debug_rust_types": '''`residual` 是 `f64`，容差不能写成整数再强转；用 `1e-6_f64` 并显式比较：
```rust
let tolerance: f64 = 1e-6;
if residual.abs() <= tolerance {
    // converged
}
```
保持 `f64` 精度，避免 `as i32` 截断。''',
}

# 10 topics × 5 variants = 50 条
rows = []
variant_notes = [
    "请给出最小可运行示例。",
    "请指出一个常见错误并说明如何避免。",
    "请补上边界条件检查。",
    "请用注释解释每个关键步骤。",
    "如果输入是空列表或 NaN，应该怎么处理？",
]
for rep in range(5):
    for key, prompt in TOPICS:
        body = BODIES[key]
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{prompt}{variant_notes[rep]}"},
                {"role": "assistant", "content": body + f"\n\n（合成样本 #{len(rows) + 1}，变体 {rep + 1}）"},
            ],
            "synthetic": True,
            "source": "synthetic",
        })

# 每 5 条抽 1 条追加长附录，使部分样本接近 max_length=2048，覆盖真实序列长度下的显存/吞吐。
long_appendix = "\n\n## 参考附录（合成）\n" + "\n\n".join(BODIES.values())
for idx in range(0, len(rows), 5):
    rows[idx]["messages"][-1]["content"] += long_appendix * 2

out = pathlib.Path(__file__).resolve().parent.parent / "data" / "synthetic-50.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} rows -> {out}")
