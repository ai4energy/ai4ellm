#!/usr/bin/env python3
"""按文件名判定能源/化工/电气范围：keep / drop / review。

默认只生成决策文件，不删除。--apply 时把 drop 移到 books-out-of-scope/。
review 默认保留（避免误删），需要 --drop-review 才处理。
"""
import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path("/home/ai4e-hermes/books-processed")
OUT_DIR = Path("/home/ai4e-hermes/ai4ellm/data/corpus-index")
DROP_DIR = Path("/home/ai4e-hermes/books-out-of-scope")

KEEP = [
    # English
    r"energy", r"power", r"electric", r"electrical", r"electricity", r"grid",
    r"voltage", r"motor", r"generator", r"transformer", r"circuit", r"electronic",
    r"semiconductor", r"battery", r"fuel", r"hydrogen", r"solar", r"wind",
    r"geothermal", r"biomass", r"nuclear", r"combustion", r"turbine", r"boiler",
    r"steam", r"heat", r"thermal", r"thermodynamic", r"refrigerat", r"hvac",
    r"cooling", r"cryogen", r"exergy", r"district_heating", r"smart_grid",
    r"chem", r"chemical", r"process", r"reactor", r"distillation", r"separation",
    r"absorption", r"adsorption", r"catalys", r"petrochem", r"petroleum", r"oil",
    r"gas", r"refiner", r"corrosion", r"polymer", r"material", r"fluid",
    r"hydraulic", r"pneumatic", r"control", r"instrument", r"measurement",
    r"signal", r"electromagnetic", r"drive", r"machinery", r"protection", r"relay",
    r"mechanical", r"flow", r"transport_phenomena", r"hydrodynam", r"aerodynam",
    r"compressible", r"turbulen", r"boundary_layer", r"cfd", r"pump", r"compressor",
    r"valve", r"pipe", r"pipeline", r"nozzle", r"heat_exchanger", r"packed_bed",
    r"aspen", r"hysys", r"carbon_capture", r"ccs", r"decarbon", r"emission",
    r"climate", r"plasma", r"tokamak", r"fusion", r"propulsion", r"rocket",
    r"substation", r"power_transmission", r"power_distribution", r"renewable",
    r"sustainab", r"environment", r"pollution", r"waste", r"desalination",
    r"membrane", r"electrochem", r"thermochem", r"kinetics", r"purification",
    r"industrial", r"organic", r"inorganic", r"analytical", r"physical_chemistry",
    r"metallurg", r"extractive", r"mineral", r"mining", r"smelt", r"hydrometallurg",
    r"pyrometallurg", r"electroplat", r"welding", r"switchgear", r"busbar",
    r"breaker", r"cable", r"insulation", r"grounding", r"earthing", r"scada",
    r"plc", r"dcs", r"hvdc", r"microgrid", r"energy_storage", r"supercapacitor",
    r"photovoltaic", r"tidal", r"wave_energy", r"gasification", r"pyrolysis",
    r"syngas", r"fuel_cell", r"bioenergy", r"anaerobic", r"waste_to_energy",
    r"cooling_tower", r"district_heating", r"blower", r"fan", r"process_safety",
    # Chinese
    r"能源", r"电力", r"电气", r"电网", r"电机", r"发电", r"锅炉", r"汽轮机",
    r"热力", r"热工", r"传热", r"工程热", r"化工", r"化学", r"反应", r"精馏",
    r"分离", r"催化", r"石油", r"天然气", r"燃气", r"燃烧", r"制冷", r"空调",
    r"储能", r"电池", r"氢能", r"风电", r"光伏", r"核电", r"新能源", r"节能",
    r"过程", r"控制", r"仪表", r"流体", r"机械", r"材料", r"电工", r"电子",
    r"流动", r"传质", r"泵", r"压缩机", r"阀", r"管道", r"换热", r"碳捕集",
    r"碳中和", r"排放", r"环境", r"水处理", r"膜", r"电解", r"电化学",
    r"变电站", r"输配电", r"继电保护", r"自动化", r"检测", r"仪器",
    r"仿真", r"模拟", r"工业", r"有机", r"无机", r"分析化学", r"物理化学",
    r"动力学", r"冶金", r"矿物", r"采矿", r"冶炼", r"电镀", r"焊接", r"开关柜",
    r"断路器", r"电缆", r"绝缘", r"接地", r"微电网", r"光伏", r"水电", r"潮汐",
    r"气化", r"热解", r"燃料电池", r"生物质", r"冷却塔", r"风机", r"泵",
]

DROP = [
    # 医学/生物/健康
    r"medicine", r"medical", r"clinical", r"nursing", r"pharmacolog", r"dentist",
    r"veterinar", r"anatomy", r"physiolog", r"pathology", r"surgery", r"cancer",
    r"immunolog", r"microbiolog", r"biolog", r"botany", r"zoolog", r"ecology",
    r"psycholog", r"psychiatr", r"nutrition", r"diet", r"中医", r"医学", r"临床",
    r"护理", r"药理", r"解剖", r"生理", r"病理", r"生物", r"生态", r"心理",
    # 法律/历史/哲学/艺术/教育/语言
    r"law", r"legal", r"jurisprud", r"history", r"historical", r"philosoph",
    r"religion", r"theolog", r"art", r"music", r"literature", r"poetry",
    r"language", r"linguistic", r"education", r"teaching", r"pedagog",
    r"法律", r"历史", r"哲学", r"宗教", r"艺术", r"音乐", r"文学", r"语言",
    r"教育", r"教学",
    # 考试/题库（非领域）
    r"jee_main", r"jee_advanced", r"neet", r"gre_", r"gmat", r"sat_", r"ielts",
    r"toefl", r"multiple-choice", r"practice_sets", r"question_bank",
    r"高考", r"考研", r"题库", r"习题", r"试卷",
    # 管理/经济/金融/营销/政治/军事
    r"management", r"economics", r"finance", r"marketing", r"accounting",
    r"business", r"politics", r"political", r"military", r"weapon", r"war",
    r"管理", r"经济", r"金融", r"营销", r"会计", r"政治", r"军事", r"武器",
    # 生活/旅行/体育
    r"cooking", r"recipe", r"travel", r"tourism", r"sport", r"fitness",
    r"motor_cycle", r"motorcycle", r"烹饪", r"旅行", r"体育",
    # 纯数学/物理/天文/计算机（不含工程应用）
    r"algebra", r"calculus", r"geometry", r"number_theory", r"topology",
    r"astronomy", r"astrophys", r"cosmolog", r"computer_science", r"programming",
    r"algorithm", r"machine_learning", r"deep_learning", r"neural_network",
    r"数学", r"微积分", r"几何", r"天文学", r"计算机", r"编程", r"算法",
]

KEEP_RE = re.compile("|".join(KEEP), re.I)
DROP_RE = re.compile("|".join(DROP), re.I)


def decide(name: str) -> str:
    if KEEP_RE.search(name):
        return "keep"          # keep 优先，避免“能源经济学”被误删
    if DROP_RE.search(name):
        return "drop"
    return "review"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--drop-review", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(ROOT.glob("*.pdf")):
        dec = decide(p.name)
        rows.append({"name": p.name, "path": str(p), "decision": dec})
    out = OUT_DIR / "scope_decision.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    counts = {k: sum(1 for r in rows if r["decision"] == k)
              for k in ("keep", "drop", "review")}
    drops = [r["name"] for r in rows if r["decision"] == "drop"]
    review = [r["name"] for r in rows if r["decision"] == "review"]
    result = {
        "apply": args.apply,
        "total": len(rows),
        "counts": counts,
        "decision_file": str(out),
        "drop_samples": drops[:30],
        "review_samples": review[:30],
    }
    if args.apply:
        DROP_DIR.mkdir(parents=True, exist_ok=True)
        moved = 0
        for r in rows:
            if r["decision"] == "drop" or (args.drop_review and r["decision"] == "review"):
                src = Path(r["path"])
                dst = DROP_DIR / src.name
                if dst.exists():
                    dst = DROP_DIR / f"{src.stem}-{r['path'].split('/')[-1][:8]}{src.suffix}"
                shutil.move(str(src), str(dst))
                moved += 1
        result["moved"] = moved
        result["drop_dir"] = str(DROP_DIR)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
