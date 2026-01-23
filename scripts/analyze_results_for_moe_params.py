import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# 선택적 의존성: seaborn, sklearn
try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    RandomForestRegressor = None


# -------------------------------
# 1) run name parser
# -------------------------------
def parse_run_name(run_name: str):
    # Baseline: YOLOn_multi_s11
    m = re.match(r"^(YOLO[a-z])_(\w+)_s(\d+)$", run_name)
    if m:
        return ("base", m.group(2), int(m.group(3)))

    # MoE: MoE_trial0_multi_BAL2.00_ENT0.10_AUX0.020_NS0.010_s11
    m = re.match(
        r"^MoE_(trial\d+_)?(\w+)_BAL([0-9.]+)_ENT([0-9.]+)_AUX([0-9.]+)_NS([0-9.]+)"
        r"(?:_SCHED_[A-Za-z0-9_\.]+)?_s(\d+)$",
        run_name
    )
    if m:
        domain = m.group(2)
        seed = int(m.group(7))
        return ("moe", domain, seed)

    print(f"[WARN] parse failed: {run_name}")
    return None, None, None


def load_moe_params_if_exists(run_dir: Path):
    f = run_dir / "moe_params.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text())

def load_moe_schedule(run_dir: Path) -> dict | None:
    sched_path = run_dir / "moe_schedule.json"
    if not sched_path.exists():
        return None
    try:
        return json.loads(sched_path.read_text())
    except Exception as e:
        print(f"[WARN] Failed to load {sched_path}: {e}")
        return None

def load_args_yaml(run_dir: Path):
    for name in ("args.yaml", "hyp.yaml", "opt.yaml"):
        f = run_dir / name
        if f.exists():
            return yaml.safe_load(f.read_text())
    return {}


# -------------------------------
# 2) Param-level analysis main
# -------------------------------

def run_param_analysis(
    results_root: Path,
    analysis_root: Path,
    top_k: int | None = None,
    make_plots: bool = True,
):
    """
    MoE 하이퍼패러미터 랜덤 서치 결과를 한 번에 분석하는 함수.

    - 모든 run 의 마지막 epoch 지표 + moe_params.json을 테이블로 저장
    - MoE run 들만 모아서:
      * 단변량 scatter (mAP vs 각 파라미터)
      * 상관계수 히트맵
      * pairplot (샘플 수가 충분할 때)
      * RandomForest 기반 feature importance
      * (lambda_balance, lambda_entropy) 2D 색깔 scatter
      * trial 별 테이블 및 top-k 결과
    """
    analysis_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    # ---------------------------
    # 2-1) 모든 run 스캔
    # ---------------------------
    for run_dir in sorted(results_root.glob("*")):
        results_file = run_dir / "results.csv"
        if not results_file.exists():
            continue

        model, domain, seed = parse_run_name(run_dir.name)
        if model is None:
            continue

        df = pd.read_csv(results_file)
        if df.empty:
            continue
        last = df.iloc[-1]

        base_row = {
            "run": run_dir.name,
            "model": model,
            "domain": domain,
            "seed": seed,
            "epoch": int(last["epoch"]),
            "mAP50": last.get("metrics/mAP50(B)", np.nan),
            "mAP5095": last.get("metrics/mAP50-95(B)", np.nan),
            "precision": last.get("metrics/precision(B)", np.nan),
            "recall": last.get("metrics/recall(B)", np.nan),
        }

        # MoE 파라미터 (있으면)
        moe_params = load_moe_params_if_exists(run_dir)
        for k, v in moe_params.items():
            base_row[k] = v

        rows.append(base_row)


    #####################

        schedule = load_moe_schedule(run_dir)

        if schedule is None:
            base_row.update({
                "has_schedule": False,
                "schedule_tag": "NOSCHED",
                "schedule_param": None,
                "schedule_min": None,
                "schedule_max": None,
            })
        else:
            schedule_tag = schedule.get("tag", "UNKNOWN")

            schedule_param = None
            schedule_min = None
            schedule_max = None

            for k in ("lambda_balance", "lambda_entropy", "noise_scale"):
                if k in schedule:
                    cfg = schedule[k]
                    if cfg.get("m_min") != 1.0 or cfg.get("m_max") != 1.0:
                        schedule_param = k
                        schedule_min = cfg.get("m_min")
                        schedule_max = cfg.get("m_max")
                        break

            base_row.update({
                "has_schedule": True,
                "schedule_tag": schedule_tag,
                "schedule_param": schedule_param,
                "schedule_min": schedule_min,
                "schedule_max": schedule_max,
            })

        rows.append(base_row)

#####
    df = pd.DataFrame(rows)
    summary_path = analysis_root / "moe_param_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"[INFO] Saved param summary → {summary_path}")

    # ---------------------------
    # 2-2) MoE run 만 필터링
    # ---------------------------
    df_moe = df[df["model"] == "moe"].copy()
    if df_moe.empty:
        print("[INFO] No MoE runs detected")
        return df

    # 공통 파라미터 컬럼
    param_cols = [
        "lambda_balance",
        "lambda_entropy",
        "aux_loss_weight",
        "noise_scale",
    ]
    param_cols_present = [p for p in param_cols if p in df_moe.columns]

    # ---------------------------
    # 3) 단변량 scatter: mAP vs 각 파라미터
    # ---------------------------
    if make_plots:
        for p in param_cols_present:
            plt.figure()
            plt.scatter(df_moe[p], df_moe["mAP5095"])
            plt.xlabel(p)
            plt.ylabel("mAP50-95")
            plt.title(f"mAP5095 vs {p}")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            out = analysis_root / f"scatter_mAP5095_vs_{p}.png"
            plt.savefig(out, dpi=200)
            plt.close()
            print(f"[INFO] Saved scatter: {out}")

    # ---------------------------
    # 4) 상관계수 히트맵 + pairplot
    # ---------------------------
    if make_plots and len(df_moe) >= 3 and param_cols_present:
        numeric_cols = ["mAP5095"] + param_cols_present
        df_num = df_moe[numeric_cols].dropna()

        # (a) Correlation heatmap
        corr = df_num.corr()
        plt.figure(figsize=(6, 5))
        if sns is not None:
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        else:
            plt.imshow(corr, cmap="coolwarm")
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
            plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation matrix (params & mAP5095)")
        plt.tight_layout()
        out = analysis_root / "corr_params_mAP5095.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[INFO] Saved correlation heatmap → {out}")

        # (b) Pairplot (샘플이 많을수록 의미가 큼)
        if sns is not None and len(df_num) >= 5:
            g = sns.pairplot(df_num)
            g.fig.suptitle("Pairplot of MoE params vs mAP5095", y=1.02)
            out = analysis_root / "pairplot_params_mAP5095.png"
            g.savefig(out, dpi=200)
            plt.close(g.fig)
            print(f"[INFO] Saved pairplot → {out}")

    # ---------------------------
    # 5) RandomForest 기반 feature importance
    # ---------------------------
    if (
        RandomForestRegressor is not None
        and len(df_moe) >= max(5, len(param_cols_present) + 1)
        and param_cols_present
    ):
        X = df_moe[param_cols_present].astype(float)
        y = df_moe["mAP5095"].astype(float)
        mask = ~X.isna().any(axis=1) & ~y.isna()
        X = X[mask]
        y = y[mask]

        if len(X) >= max(5, len(param_cols_present) + 1):
            rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                random_state=0,
            )
            rf.fit(X, y)
            importances = rf.feature_importances_

            imp_df = (
                pd.DataFrame(
                    {"param": param_cols_present, "importance": importances}
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )

            imp_csv = analysis_root / "moe_param_feature_importance.csv"
            imp_df.to_csv(imp_csv, index=False)
            print(f"[INFO] Saved feature importance → {imp_csv}")

            if make_plots:
                plt.figure()
                plt.bar(imp_df["param"], imp_df["importance"])
                plt.ylabel("importance")
                plt.title("RandomForest feature importance (mAP5095)")
                plt.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                out = analysis_root / "feature_importance_bar.png"
                plt.savefig(out, dpi=200)
                plt.close()
                print(f"[INFO] Saved feature importance bar plot → {out}")
        else:
            print("[INFO] Not enough samples for RandomForest importance.")
    else:
        if RandomForestRegressor is None:
            print("[INFO] sklearn not available → skip feature importance.")
        elif not param_cols_present:
            print("[INFO] No param columns found → skip feature importance.")

    # ---------------------------
    # 6) 2D 색깔 scatter (lambda_balance vs lambda_entropy)
    # ---------------------------
    if make_plots and len(df_moe) >= 3:
        if "lambda_balance" in df_moe.columns and "lambda_entropy" in df_moe.columns:
            df_2d = df_moe[["lambda_balance", "lambda_entropy", "mAP5095"]].dropna()
            if len(df_2d) >= 3:
                plt.figure()
                sc = plt.scatter(
                    df_2d["lambda_balance"],
                    df_2d["lambda_entropy"],
                    c=df_2d["mAP5095"],
                    cmap="viridis",
                )
                plt.colorbar(sc, label="mAP5095")
                plt.xlabel("lambda_balance")
                plt.ylabel("lambda_entropy")
                plt.title("2D param space (color = mAP5095)")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                out = analysis_root / "scatter2d_balance_entropy_mAP5095.png"
                plt.savefig(out, dpi=200)
                plt.close()
                print(f"[INFO] Saved 2D param scatter → {out}")

    # ---------------------------
    # 7) Trial-by-trial table
    # ---------------------------
    df_moe["trial"] = (
        df_moe["run"].str.extract(r"trial(\d+)_", expand=False).astype(float)
    )

    trial_table = df_moe.sort_values(["trial", "seed"])[
        ["run", "trial", "seed", "mAP5095"] + param_cols_present
    ]

    trial_table_path = analysis_root / "moe_trial_table.csv"
    trial_table.to_csv(trial_table_path, index=False)
    print(f"[INFO] Saved trial table → {trial_table_path}")


    df_sched = df[df["has_schedule"]]

    for param in ["lambda_balance", "lambda_entropy", "noise_scale"]:
        sub = df_sched[df_sched["schedule_param"] == param]
        if sub.empty:
            continue

        pivot = sub.pivot_table(
            index="schedule_min",
            values="mAP5095",
            aggfunc=["mean", "std"],
        )
        print(f"\n[SCHEDULE RESULT] {param}")
        print(pivot)


    # ---------------------------
    # *) Top-K parameter selection
    # ---------------------------
    if top_k is not None:
        top = df_moe.sort_values("mAP5095", ascending=False).head(top_k)
        top_path = analysis_root / f"top_{top_k}_params.csv"
        top.to_csv(top_path, index=False)
        print(f"[INFO] Saved top-{top_k} configs → {top_path}")

    print("\n[DONE] Param analysis complete.")
    return df


if __name__ == "__main__":
    default_analysis_root = Path(
        "/ultralytics/outputs/aux_weight_test_1"
    )
    default_results_root = default_analysis_root / "results"
    run_param_analysis(default_results_root, default_analysis_root)
