import json
from pathlib import Path
from types import SimpleNamespace

from ultralytics import YOLO
from ultralytics.nn.modules.head import MoEDetect

from custom_core import MoETrainer, create_dataset_config, install_freeze_warning_filter
from custom_core.config_loader import register_symlink_config_loader

from cfg.project_config import ProjectConfig
from cfg.sd_config import sd_symlink_config_loader
# from cfg.sd_config_single_nonmoving import sd_symlink_config_loader_n
# from cfg.sd_config_single_rider import sd_symlink_config_loader_r

from analyze_results import run_analysis
from analyze_results_for_moe_params import run_param_analysis
from random_param_search import MoEParams


def create_project_config() -> ProjectConfig:
    cfg =  ProjectConfig(
        name="multi_11n_moe_param_test_1",
        seed_list=[11, 59, 61],
        random_trial_counts=20,
        loader_pairs = [
            ("multi", sd_symlink_config_loader),
            # ("nonmoving", sd_symlink_config_loader_n),
            # ("rider", sd_symlink_config_loader_r),
            ],
        dataset_mode="TRAIN",
        param_mode="RANDOM",
        # dataset_mode="DEBUG",
        # param_mode="FIXED",
        max_train=3000,
        val_ratio=0.1,
        epochs=20,
        run_analysis=True,
    )
    if cfg.param_mode == "FIXED":
        cfg.moe_fixed_params = MoEParams.fixed(
        moe_aux_loss=0.03,
        lambda_entropy=0.08,
        lambda_balance=2.0,
        noise_scale=0.015,
        )
    return cfg

def params_sanity_check(args, moe_head, moe_params):
    for k, v in moe_params.to_dict().items():
        assert getattr(args, k) == v
    assert moe_head.lambda_entropy == moe_params.lambda_entropy
    assert moe_head.lambda_balance == moe_params.lambda_balance
    assert moe_head.noise_scale == moe_params.noise_scale  

def run_benchmark(
    cfg: ProjectConfig,
    moe_params: MoEParams,
    seed: int,
    trial_idx: int | None = None,
    dataset_path: Path | None = None,
    skip_baseline: bool = False,
    domain_name: str | None = None
    ):
    common_hp = dict(
        project=str(cfg.results_root),
        data=str(dataset_path),
        epochs=cfg.epochs,
        batch=16,
        imgsz=640,
        seed=seed,
    )

    # 2) baseline, MoE 모델 로드
    base = YOLO("yolo11n.pt")          # COCO pretrain 완료 모델
    moe  = YOLO("yolo11-moe.yaml")     # 구조만 정의된 MoE
    # base_s = YOLO("yolo11s.pt")

    # 3) ★ backbone 포함 전체 가중치를 MoE로 복사 (strict=False)
    moe.model.load_state_dict(base.model.state_dict(), strict=False)

    # 4) Detect head → MoE head 복사
    base_head = base.model.model[-1]
    moe_head  = moe.model.model[-1]
    assert isinstance(moe_head, MoEDetect)

    args = moe.model.args

    if isinstance(args, dict):
        args = SimpleNamespace(**args)
        moe.model.args = args  # overwrite

    # ---- apply MoE params ----
    for k, v in moe_params.to_dict().items():
        setattr(args, k, v)

    # detect head에는 필드 기반으로 주입
    moe_head.lambda_entropy = moe_params.lambda_entropy
    moe_head.lambda_balance = moe_params.lambda_balance
    moe_head.noise_scale = moe_params.noise_scale

    moe_head.init_from_detect(base_head)

    params_sanity_check(args, moe_head, moe_params)
    print(f"[Seed {seed}][{domain_name}] Random MoE Params = {moe_params}")

    moe.ckpt = True

    # Try baseline(YOLO11n) training only once
    if skip_baseline is False:
        results_base = base.train(
            **common_hp,
            name=f"YOLOn_{domain_name}_s{seed}",
        )
    
    # 7) MoE 학습
    tag = f"BAL{moe_params.lambda_balance:.2f}_ENT{moe_params.lambda_entropy:.2f}_" \
            f"AUX{moe_params.moe_aux_loss:.3f}_NS{moe_params.noise_scale:.3f}"

    trial_part = f"trial{trial_idx}_" if trial_idx is not None else ""
    moe_dir_name = f"MoE_{trial_part}{domain_name}_{tag}_s{seed}"

    results_moe = moe.train(
        trainer=MoETrainer,
        **common_hp,
        name=moe_dir_name,
    )
    run_dir = Path(cfg.results_root) / moe_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "moe_params.json").write_text(
        json.dumps(moe_params.to_dict(), indent=2)
    )
    # results_base_s = base_s.train(
    #     **common_hp,
    #     name=f"YOLOs_{domain_name}_s{seed}",
    # )

def run_random_param_search(cfg: ProjectConfig):
    search_seed = cfg.seed_list[0]
    trial_params = [MoEParams.random_params(seed=i) for i in range(cfg.random_trial_counts)]

    for domain_name, loader in cfg.loader_pairs:
        # 2) 도메인별 초기화
        register_symlink_config_loader(loader)
        # 2-1) dataset은 도메인+seed 조합마다 한 번만 생성
        if cfg.dataset_mode == "TRAIN":
            dataset_config_path = create_dataset_config(
                cfg.val_ratio, seed=search_seed,
                max_train=cfg.max_train, max_val=cfg.max_val
            )
        else:
            dataset_config_path = "/ultralytics/run/sd_moe/multihead_data.yaml"

        # 2-2) baseline은 (domain, seed) 조합에 대해 딱 한 번만
        base_run_dir = cfg.results_root / f"YOLOn_{domain_name}_s{search_seed}"
        skip_baseline = (base_run_dir / "results.csv").exists()

        # 3) trial별 MoE 학습
        for trial_idx, moe_params in enumerate(trial_params):
            run_benchmark(
                cfg=cfg,
                moe_params=moe_params,
                seed=search_seed,
                trial_idx=trial_idx,
                skip_baseline=skip_baseline,
                dataset_path=dataset_config_path,
                domain_name=domain_name,
            )
            # baseline은 한 번 돌고 나면 늘 True
            skip_baseline = True

    if cfg.run_analysis:
        run_param_analysis(cfg.results_root, cfg.analysis_root, None, False)

def run_param_fixed_eval(cfg: ProjectConfig):
    moe_params = cfg.moe_fixed_params
    for domain_name, loader in cfg.loader_pairs:
        register_symlink_config_loader(loader)

        for seed in cfg.seed_list:
            if cfg.dataset_mode == "TRAIN":
                dataset_config_path = create_dataset_config(
                    cfg.val_ratio, seed=seed,
                    max_train=cfg.max_train, max_val=cfg.max_val
                )
            else:
                dataset_config_path = "/ultralytics/run/sd_moe/multihead_data.yaml"

            base_run_dir = cfg.results_root / f"YOLOn_{domain_name}_s{seed}"
            skip_baseline = (base_run_dir / "results.csv").exists()

            run_benchmark(
                cfg=cfg,
                moe_params=moe_params,
                seed=seed,
                trial_idx=None,
                skip_baseline=skip_baseline,
                dataset_path=dataset_config_path,
                domain_name=domain_name
            )

    if cfg.run_analysis:
        run_analysis(cfg.results_root, cfg.analysis_root)

def main():
    cfg = create_project_config()
    install_freeze_warning_filter()
    # Random parameter sampling. Dataset fixed
    if cfg.param_mode == "RANDOM":
        run_random_param_search(cfg)
        # for trial_idx, moe_params in enumerate(trial_params):
        #     run_benchmark(cfg, moe_params, search_seed, trial_idx, skip_baseline)
        #     base_run_dir = cfg.results_root / f"YOLOn_{cfg.loader_pairs[0][0]}_s{search_seed}"
        #     if (base_run_dir / "results.csv").exists():
        #         skip_baseline = True
        # if cfg.run_analysis:
        #     run_param_analysis(cfg.results_root, cfg.analysis_root, None, False)

    # Random dataset sampling. MoE parameters fixed
    elif cfg.param_mode == "FIXED":
        run_param_fixed_eval(cfg)
        # moe_params = MoEParams.fixed(
        #     moe_aux_loss=0.03,
        #     lambda_entropy=0.08,
        #     lambda_balance=2.0,
        #     noise_scale=0.015
        # )
        # for seed in cfg.seed_list:
        #     run_benchmark(cfg, moe_params, seed)
        # if cfg.run_analysis:
        #     run_analysis(cfg.results_root, cfg.analysis_root)

if __name__ == "__main__":
    main()