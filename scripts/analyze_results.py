import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import yaml


# =========================
# 유틸 함수
# =========================
def parse_run_name(run_name: str):
    """
    새로운 네이밍 규칙을 지원하는 파서.
    
    MoE 예:
        MoE_trial3_multi_BAL2.00_ENT0.08_AUX0.030_NS0.015_s11
    Baseline 예:
        YOLOn_multi_s11
    """

    # Baseline pattern
    m = re.match(r"^(YOLO[a-z])_(\w+)_s(\d+)$", run_name)
    if m:
        return ("base", m.group(2), int(m.group(3)))

    # MoE pattern
    m = re.match(
    r"^MoE_(trial\d+_)?(\w+)_BAL([0-9.]+)_ENT([0-9.]+)_AUX([0-9.]+)_NS([0-9.]+)"
    r"(?:_SCHED_[A-Za-z0-9_\.]+)?_s(\d+)$",
    run_name
    )
    if m:
        domain = m.group(2)
        seed = int(m.group(7))
        return ("moe", domain, seed)

    # Unknown pattern
    print(f"[WARN] Cannot parse run name: {run_name}")
    return None, None, None


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

def load_moe_schedule(run_dir: Path) -> dict | None:
    sched_path = run_dir / "moe_schedule.json"
    if not sched_path.exists():
        return None
    try:
        return json.loads(sched_path.read_text())
    except Exception as e:
        print(f"[WARN] Failed to load {sched_path}: {e}")
        return None

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

def analyze_expert_heads(analysis_root):
    df = pd.read_csv(analysis_root / "expert_usage_ratio.csv")

    # 1) epoch 평균 / seed 평균 버전 생성
    mean_df = (
        df.groupby(["run", "model", "domain", "expert"])
        [["global_p", "nonmoving_p", "rider_p", "delta"]]
        .mean()
        .reset_index()
    )

    mean_df.to_csv(analysis_root / "expert_usage_mean_by_run.csv", index=False)

    # 2) expert × model pivot — 사람이 보기 쉬운 형태
    pivot = mean_df.pivot_table(
        index=["model", "expert"],
        values=["nonmoving_p", "rider_p", "delta"],
        aggfunc="mean"
    )

    pivot.to_csv(analysis_root / "expert_usage_pivot.csv")

    # 3) 도메인 분리 스칼라 요약
    summary = (
        mean_df.groupby(["run"])
            .agg(
                mean_delta=("delta", "mean"),
                abs_delta=("delta", lambda x: x.abs().mean())
            )
    )

    summary.to_csv(analysis_root / "expert_usage_summary.csv")

def plot_expert_usage_over_epochs(analysis_root, run_filter=None, model_filter=None):
    df = pd.read_csv(analysis_root / "expert_usage_ratio.csv")

    # 필요하면 run 또는 model 필터링
    if run_filter is not None:
        df = df[df["run"] == run_filter]
    if model_filter is not None:
        df = df[df["model"] == model_filter]

    # epoch 순 정렬
    df = df.sort_values(["epoch", "expert"])

    # 1) 도메인별 expert 비율
    for domain in df["domain"].unique():
        sub = df[df["domain"] == domain]

        plt.figure()
        for e in sorted(sub["expert"].unique()):
            s_e = sub[sub["expert"] == e]
            plt.plot(s_e["epoch"], s_e["nonmoving_p" if domain == "nonmoving" else "rider_p"],
                     label=f"expert {e}")
        plt.xlabel("epoch")
        plt.ylabel("usage ratio")
        plt.title(f"Expert usage over epochs (domain={domain})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(analysis_root / f"expert_usage_epochs_{domain}.png")
        plt.close()

    # 2) delta (도메인 분리 정도)
    plt.figure()
    for e in sorted(df["expert"].unique()):
        s_e = df[df["expert"] == e]
        plt.plot(s_e["epoch"], s_e["delta"], label=f"expert {e}")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("epoch")
    plt.ylabel("delta (nonmoving_p - rider_p)")
    plt.title("Expert domain-split (delta) over epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_root / "expert_delta_over_epochs.png")
    plt.close()

# =========================
# 공개 함수: 다른 스크립트에서 쓰는 진입점
# =========================
def run_analysis(results_root, analysis_root):
    """
    results_root: runs가 들어있는 디렉토리 (str 또는 Path)
    analysis_root:     분석 결과를 저장할 디렉토리 (str 또는 Path)
    """
    analysis_root.mkdir(parents=True, exist_ok=True)

    # 1) run 스캔 및 summary 테이블 생성
    rows = []
    usage_rows = []

    for run_dir in sorted(results_root.glob("*")):
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

        schedule = load_moe_schedule(run_dir)
        
        if schedule is None:
            schedule_tag = "NOSCHED"
            schedule_param = None
            schedule_min = None
            schedule_max = None
        else:
            schedule_tag = schedule.get("tag", "UNKNOWN")

            # 어떤 파라미터가 스케줄링 대상인지 자동 추론
            schedule_param = None
            schedule_min = None
            schedule_max = None
            for k in ("lambda_balance", "lambda_entropy", "gumbel_scale"):
                if k in schedule:
                    cfg = schedule[k]
                    if cfg.get("m_min") != 1.0 or cfg.get("m_max") != 1.0:
                        schedule_param = k
                        schedule_min = cfg.get("m_min")
                        schedule_max = cfg.get("m_max")
                        break

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
            "schedule_tag": schedule_tag,
            "schedule_param": schedule_param,
            "schedule_min": schedule_min,
            "schedule_max": schedule_max,
            "has_schedule": schedule is not None,
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

        usage_file = run_dir / "moe_usage.json"
        if usage_file.exists():
            usage_data = json.loads(usage_file.read_text())

            for rec in usage_data:
                epoch = rec["epoch"]
                g = np.array(rec["global_usage"])        # shape (num_scales, num_experts)
                n = np.array(rec["nonmoving_usage"])     # shape (num_experts,)
                r = np.array(rec["rider_usage"])         # shape (num_experts,)

                # global: 스케일 합 → (E,)
                g_sum = g.sum(axis=0)

                # 정규화 (/sum) → 확률 분포
                g_p = g_sum / g_sum.sum()
                n_p = n / n.sum()
                r_p = r / r.sum()

                for expert_id in range(len(g_p)):
                    usage_rows.append(
                        {
                            "run": run_dir.name,
                            "model": model,
                            "domain": domain,
                            "seed": seed,
                            "epoch": epoch,
                            "expert": expert_id,
                            "global_p": float(g_p[expert_id]),
                            "nonmoving_p": float(n_p[expert_id]),
                            "rider_p": float(r_p[expert_id]),
                            "delta": float(n_p[expert_id] - r_p[expert_id]),
                        }
                    )


    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["domain", "model", "seed"])
    summary_path = analysis_root / "summary_results_last_epoch.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[INFO] Saved summary to {summary_path}")

    # 2) 도메인×모델별 통계 저장
    group_stats = (
        summary
        .groupby(["domain", "model"])[["mAP50", "mAP5095", "precision", "recall"]]
        .agg(["mean", "std"])
    )

    stats_path = analysis_root / "group_stats.json"
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

    pivot_path = analysis_root / "pivot_mAP5095_domain_seed.csv"
    pivot.to_csv(pivot_path)
    print(f"[INFO] Saved pivot table to {pivot_path}")

    print("\n=== Pivot: mAP50-95 per domain/seed (base vs moe) ===")
    print(pivot)

    # expert_usage_count = run_dir / "moe_usage.json"
    # usage_data = json.loads(expert_usage_count.read_text())

    # usage_rows = []
    # for rec in data:
    #     epoch = rec["epoch"]
    #     g = np.array(rec["global"])        # shape (num_scales, num_experts)
    #     n = np.array(rec["nonmoving"])     # shape (num_experts,)
    #     r = np.array(rec["rider"])         # shape (num_experts,)

    #     # 2) global: 스케일 차원 합치고 expert 축만 남기기
    #     g_sum = g.sum(axis=0)  # (E,)

    #     # 3) 정규화 (/sum) → 확률 분포
    #     g_p = g_sum / g_sum.sum()
    #     n_p = n / n.sum()
    #     r_p = r / r.sum()

    #     # 4) 테이블 형태로 정리
    #     for expert_id in range(len(g_p)):
    #         usage_rows.append(
    #             {
    #                 "epoch": epoch,
    #                 "expert": expert_id,
    #                 "global_p": float(g_p[expert_id]),
    #                 "nonmoving_p": float(n_p[expert_id]),
    #                 "rider_p": float(r_p[expert_id]),
    #                 "delta": float(n_p[expert_id] - r_p[expert_id])
    #             }
    #         )
    if usage_rows:
        expert_usage = pd.DataFrame(usage_rows)
        expert_usage_path = analysis_root / "expert_usage_ratio.csv"
        expert_usage.to_csv(expert_usage_path)
        print(f"[INFO] Saved expert usage ratio to {expert_usage_path}")
        print(expert_usage[["epoch", "expert", "delta"]])
    else:
        print("[INFO] No moe_usage.json found in runs → skip expert usage export.")

    # 4) 그래프들
    plot_domain_model_bar(summary, analysis_root / "mAP5095_domain_model_bar.png")
    # plot_seed_boxplot(summary, analysis_root / "mAP5095_domain_seed_boxplot.png")
    analyze_expert_heads(analysis_root)
    plot_expert_usage_over_epochs(analysis_root)
    print(f"\n[DONE] Analysis finished. See folder: {analysis_root}")


if __name__ == "__main__":
    default_analysis_root = Path(
        "/ultralytics/outputs/scheduling_test_cosine_with_peak_debug"
    )
    default_results_root = default_analysis_root / "results"
    run_analysis(default_results_root, default_analysis_root)
