# analyze_results.py
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import yaml


# =========================
# 유틸 함수
# =========================
def parse_run_name(run_name: str):
    """
    예: 'moe_multi_s42' -> model='moe', domain='multi', seed=42
        'base_nonmoving_s0' -> model='base', domain='nonmoving', seed=0
    """
    parts = run_name.split("_")
    if len(parts) < 3:
        # 예상치 못한 이름이면 안전하게 처리
        return None, None, None

    model = parts[0]       # 'base' or 'moe'
    domain = parts[1]      # 'multi' / 'nonmoving' / 'rider' ...
    seed_str = parts[2]    # 's0', 's42' ...

    try:
        seed = int(seed_str.lstrip("s"))
    except ValueError:
        seed = None

    return model, domain, seed


def load_args_yaml(run_dir: Path):
    """
    Ultralytics v8에서는 run 디렉토리에 args.yaml 저장됨.
    없으면 빈 dict 리턴.
    """
    for name in ("args.yaml", "hyp.yaml", "opt.yaml"):
        f = run_dir / name
        if f.exists():
            with f.open("r") as rf:
                return yaml.safe_load(rf)
    return {}


def flatten_dict(d, parent_key: str = "", sep: str = "/"):
    """
    중첩 dict를 'a/b/c' 형태의 flat dict로 변환.
    하이퍼파라미터 같이 저장할 때 사용.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def plot_domain_model_bar(df: pd.DataFrame, out_path: Path):
    """
    도메인×모델별 mAP50-95 mean, std를 막대 그래프로 저장.
    """
    stats = (
        df
        .groupby(["domain", "model"])["mAP5095"]
        .agg(["mean", "std"])
        .reset_index()
    )

    domains = stats["domain"].unique()
    models = stats["model"].unique()

    # x축: domain, 각 domain 안에서 model 2개 (base, moe ...)
    x = range(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, model in enumerate(models):
        sub = stats[stats["model"] == model]
        # domain 순서 맞추기
        sub = sub.set_index("domain").loc[domains].reset_index()

        xs = [v + (i - 0.5) * width for v in x]
        ax.bar(xs, sub["mean"], width=width, label=model, yerr=sub["std"], capsize=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels(domains)
    ax.set_ylabel("mAP50-95")
    ax.set_title("mAP50-95 by Domain and Model (last epoch)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved bar plot to {out_path}")


def plot_seed_boxplot(df: pd.DataFrame, out_path: Path):
    """
    도메인별로 seed variance를 보고 싶을 때:
    각 domain마다 base/moe mAP5095를 박스플롯으로 비교.
    """
    domains = sorted(df["domain"].unique())
    fig, axes = plt.subplots(1, len(domains),
                             figsize=(5 * len(domains), 5),
                             sharey=True)

    if len(domains) == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains):
        sub = df[df["domain"] == domain]
        data = [
            sub[sub["model"] == "base"]["mAP5095"].values,
            sub[sub["model"] == "moe"]["mAP5095"].values,
        ]
        ax.boxplot(data, labels=["base", "moe"])
        ax.set_title(domain)
        ax.set_ylabel("mAP50-95")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("mAP50-95 distribution per Domain (seed variance)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved seed boxplot to {out_path}")


# =========================
# 공개 함수: 다른 스크립트에서 쓰는 진입점
# =========================
def run_analysis(project_root, out_dir):
    """
    project_root: runs가 들어있는 디렉토리 (str 또는 Path)
    out_dir:     분석 결과를 저장할 디렉토리 (str 또는 Path)
    """
    project_root = Path(project_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) run 스캔 및 summary 테이블 생성
    rows = []

    for run_dir in sorted(project_root.glob("*")):
        results_file = run_dir / "results.csv"
        if not results_file.exists():
            continue

        model, domain, seed = parse_run_name(run_dir.name)
        if model is None:
            print(f"[WARN] Unexpected run name format: {run_dir.name}")
            continue

        # results.csv 읽기
        df = pd.read_csv(results_file)

        # 마지막 epoch 기준
        last = df.iloc[-1]

        # args.yaml에서 하이퍼파라미터 읽기
        args = load_args_yaml(run_dir)
        flat_args = flatten_dict(args)

        row = {
            "run": run_dir.name,
            "model": model,
            "domain": domain,
            "seed": seed,
            "epoch": int(last["epoch"]),
            "precision": last.get("metrics/precision(B)", float("nan")),
            "recall": last.get("metrics/recall(B)", float("nan")),
            "mAP50": last.get("metrics/mAP50(B)", float("nan")),
            "mAP5095": last.get("metrics/mAP50-95(B)", float("nan")),
        }

        important_keys = [
            "lr0",
            "lrf",
            "momentum",
            "weight_decay",
            "warmup_epochs",
            "warmup_momentum",
            "warmup_bias_lr",
            "epochs",
            "batch",
            "imgsz",
            "optimizer",
            "task",
        ]

        for k in important_keys:
            if k in args:
                row[f"hp/{k}"] = args[k]
            else:
                # flatten_dict 경로에 있을 수 있음 (예: 'train/epochs')
                for fk, fv in flat_args.items():
                    if fk.endswith(f"/{k}"):
                        row[f"hp/{k}"] = fv
                        break

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["domain", "model", "seed"])
    summary_path = out_dir / "summary_results_last_epoch.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[INFO] Saved summary to {summary_path}")

    # 2) 도메인×모델별 통계 저장
    group_stats = (
        summary
        .groupby(["domain", "model"])[["mAP50", "mAP5095", "precision", "recall"]]
        .agg(["mean", "std"])
    )

    stats_path = out_dir / "group_stats.json"
    group_stats_json = json.loads(group_stats.to_json(orient="split"))
    with stats_path.open("w") as f:
        json.dump(group_stats_json, f, indent=2)
    print(f"[INFO] Saved group stats to {stats_path}")

    print("\n=== Grouped stats (domain × model) ===")
    print(group_stats)

    # 3) Pivot 테이블 저장
    pivot = (
        summary
        .pivot_table(
            index=["domain", "seed"],
            columns="model",
            values="mAP5095",
        )
    )

    pivot_path = out_dir / "pivot_mAP5095_domain_seed.csv"
    pivot.to_csv(pivot_path)
    print(f"[INFO] Saved pivot table to {pivot_path}")

    print("\n=== Pivot: mAP50-95 per domain/seed (base vs moe) ===")
    print(pivot)

    # 4) 그래프들
    plot_domain_model_bar(summary, out_dir / "mAP5095_domain_model_bar.png")
    plot_seed_boxplot(summary, out_dir / "mAP5095_domain_seed_boxplot.png")

    print(f"\n[DONE] Analysis finished. See folder: {out_dir}")


if __name__ == "__main__":
    default_project_root = "/ultralytics/runs/single_moe_benchmark"
    default_out_dir = "/ultralytics/outputs/analysis_moe_vs_single"
    run_analysis(default_project_root, default_out_dir)
