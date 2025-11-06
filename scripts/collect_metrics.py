# scripts/collect_metrics.py
from ultralytics import YOLO
import pathlib, csv

EVALS = [
    ("A_teacher",      "runs/A_teacher/weights/best.pt",      "runs/Eval_A_teacher"),
    ("B_no_distill",   "runs/B_no_distill/weights/best.pt",   "runs/Eval_A_after_B_no_distill"),
    ("B_with_distill", "runs/B_with_distill/weights/best.pt", "runs/Eval_A_after_B_with_distill"),
]

def read_results_csv(folder):
    p = pathlib.Path(folder) / "results.csv"
    if not p.exists():
        return None, None
    # 마지막 행의 mAP50, mAP50-95 유사 컬럼을 탐색
    with p.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    last = rows[-1]
    # 후보 컬럼명들
    m50_keys   = [k for k in last.keys() if "mAP50" in k and "95" not in k]
    m5095_keys = [k for k in last.keys() if "mAP50-95" in k or "mAP_50_95" in k or "mAP50_95" in k or "map50-95" in k]
    m50   = float(last[m50_keys[0]])   if m50_keys and last[m50_keys[0]]   not in ("", "None") else None
    m5095 = float(last[m5095_keys[0]]) if m5095_keys and last[m5095_keys[0]] not in ("", "None") else None
    return m50, m5095

def safe_get_metrics(r):
    # 1) 객체 속성(v11)
    m50 = getattr(getattr(r, "box", None), "map50", None)
    m5095 = getattr(getattr(r, "box", None), "map", None)
    # 2) dict 폴백
    d = getattr(r, "results_dict", {}) or {}
    m50 = m50 if m50 is not None else d.get("metrics/mAP50")
    m5095 = m5095 if m5095 is not None else d.get("metrics/mAP50-95")
    return m50, m5095

rows = []
for name, ckpt, eval_dir in EVALS:
    print(f"\n🔹 Evaluating {name} ...")
    m = YOLO(ckpt)
    r = m.val(data="coco8.yaml", project="runs", name=pathlib.Path(eval_dir).name, plots=False, save_json=False)
    m50, m5095 = safe_get_metrics(r)

    # 3) 여전히 None이면 CSV에서 읽기
    if m50 is None or m5095 is None:
        csv_m50, csv_m5095 = read_results_csv(eval_dir)
        m50   = m50   if m50   is not None else csv_m50
        m5095 = m5095 if m5095 is not None else csv_m5095

    print(f"  mAP50={m50}  mAP50-95={m5095}")
    rows.append({"name": name, "mAP50": m50, "mAP50-95": m5095})

# 기준(A_teacher)
base = next((x["mAP50-95"] for x in rows if x["name"]=="A_teacher" and x["mAP50-95"] is not None), None)
for x in rows:
    if x["name"] == "A_teacher" or base is None or x["mAP50-95"] is None:
        x["drop_mAP50-95_vs_A"] = None
    else:
        x["drop_mAP50-95_vs_A"] = round(base - x["mAP50-95"], 6)

print("\n=== A-test 결과 비교 ===")
for x in rows:
    m50 = x["mAP50"]; m5095 = x["mAP50-95"]; drop = x["drop_mAP50-95_vs_A"]
    def f(v): return "N/A" if v is None else f"{v:.3f}"
    print(f'{x["name"]:16s}  mAP50={f(m50):>6}  mAP50-95={f(m5095):>6}  drop={f(drop)}')

out = pathlib.Path("runs/metrics_compare.csv")
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\n✅ Saved: {out}")
