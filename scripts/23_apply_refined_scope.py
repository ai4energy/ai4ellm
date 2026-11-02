#!/usr/bin/env python3
"""应用细化 scope：refined KEEP/DROP + 少量人工覆盖，移动文件并更新索引。"""
import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

REF = Path("/home/ai4e-hermes/ai4ellm/data/corpus-index/scope_refined.jsonl")
PROC = Path("/home/ai4e-hermes/books-processed")
OUT = Path("/home/ai4e-hermes/books-out-of-scope")
DB = Path("/home/ai4e-hermes/ai4ellm/data/corpus-index/corpus.sqlite")
LOG = Path("/home/ai4e-hermes/ai4ellm/data/corpus-index/scope_refined_moves.jsonl")

KEEP_OVERRIDE = re.compile(
    r"power_plant|power-plant|nuclear_power_plant|energy_management_system|iso_50001|"
    r"guide_to_energy_management|energy_management_principles|energy_management_and_efficiency|"
    r"energy_management_and_control|energy_management_systems|continuous_biopharmaceutical|"
    r"protein_and_pharmaceutical_engineering|filtration_and_purification_in_the_biopharmaceutical|"
    r"greening_of_pharmaceutical_engineering|radfrac|emanuele-mpc|"
    r"energy_efficient_digital_networks|无人驾驶车辆模型预测控制", re.I)
DROP_OVERRIDE = re.compile(
    r"history_of_the_internal_combustion_engine|history_of_rotating_machinery|"
    r"history_of_dakota_gasification|synfuels_energy_pioneers|business_process_modeling|"
    r"workflow-based_process_controlling|collaborative_business_process_engineering|"
    r"enterprise_process_management|negro_employment|geopolitics_of_energy|"
    r"IT_automation|MPC_Renaissance|Global_Warming_Toolbox", re.I)


def extraction_running() -> bool:
    out = subprocess.run(["pgrep", "-af", r"main.py --config .*corpus_full.yaml"],
                         capture_output=True, text=True)
    return bool(out.stdout.strip()) and "pgrep" not in out.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if extraction_running() and not args.force:
        raise SystemExit("抽取仍在运行，先停止再加 --apply")

    rows = [json.loads(l) for l in REF.open(encoding="utf-8")]
    conn = sqlite3.connect(DB)
    moves = {r[0]: r[1] for r in conn.execute(
        "SELECT original_path,new_path FROM moves WHERE kind='out_of_scope'")}
    cols = [r[1] for r in conn.execute("PRAGMA table_info(files)")]
    for col in ("in_scope", "out_of_scope_path", "processed_path"):
        if col not in cols:
            conn.execute(f"ALTER TABLE files ADD COLUMN {col}")

    actions = []
    for r in rows:
        name = r["name"]
        final = r["refined"]
        if KEEP_OVERRIDE.search(name):
            final = "KEEP"
        if DROP_OVERRIDE.search(name):
            final = "DROP"
        old = Path(r["path"])
        if old.exists():
            current, where = old, "processed"
        elif r["path"] in moves and Path(moves[r["path"]]).exists():
            current, where = Path(moves[r["path"]]), "out"
        elif (OUT / old.name).exists():
            current, where = OUT / old.name, "out"
        else:
            actions.append({"name": name, "path": r["path"], "action": "missing",
                            "final": final})
            continue
        if final == "KEEP" and where == "out":
            actions.append({"name": name, "src": str(current), "action": "restore"})
        elif final == "DROP" and where == "processed":
            actions.append({"name": name, "src": str(current), "action": "drop"})
        else:
            actions.append({"name": name, "src": str(current), "action": "stay",
                            "final": final})

    restore = [a for a in actions if a["action"] == "restore"]
    drop = [a for a in actions if a["action"] == "drop"]
    missing = [a for a in actions if a["action"] == "missing"]
    result = {"apply": args.apply, "keep": 3539, "restore": len(restore),
              "drop": len(drop), "missing": len(missing),
              "restore_samples": [a["name"] for a in restore[:20]],
              "drop_samples": [a["name"] for a in drop[:20]]}
    if args.apply:
        PROC.mkdir(exist_ok=True)
        OUT.mkdir(exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        moved = 0
        with LOG.open("a", encoding="utf-8") as lf:
            for a in restore + drop:
                src = Path(a["src"])
                dest_dir = PROC if a["action"] == "restore" else OUT
                dst = dest_dir / src.name
                if dst.exists():
                    dst = dest_dir / f"{src.stem}-{str(abs(hash(src)))[:8]}{src.suffix}"
                try:
                    os.rename(src, dst)
                except OSError:
                    shutil.move(str(src), str(dst))
                lf.write(json.dumps({**a, "dst": str(dst), "ts": ts},
                                    ensure_ascii=False) + "\n")
                is_keep = a["action"] == "restore"
                conn.execute(
                    "UPDATE files SET in_scope=?, processed_path=?, out_of_scope_path=? "
                    "WHERE name=?", (1 if is_keep else 0,
                                     str(dst) if is_keep else None,
                                     None if is_keep else str(dst), a["name"]))
                conn.execute("INSERT INTO moves VALUES(?,?,?,?,?)",
                             (None, str(src), str(dst),
                              "scope_restore" if is_keep else "scope_drop", ts))
                moved += 1
            conn.commit()
        result["moved"] = moved
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
