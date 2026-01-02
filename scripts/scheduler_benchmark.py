import gc, torch, time
import multiprocessing
import json

from pathlib import Path

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


fixed_params_list = [
    {"aux_loss_weight": 0.03, "lambda_entropy": 0.08, "lambda_balance": 2.0, "noise_scale": 0.015, "gumbel_scale": 0.6},
    # {"aux_loss_weight": 0.01, "lambda_entropy": 0.08, "lambda_balance": 2.0, "noise_scale": 0.015, "gumbel_scale": 0.6},
]

single_param_plan = [
    ("gumbel_scale", {"m_min": 0.0, "grid": [0.4]}),
    # balance: 후반 바닥(m_min)을 올리는 게 collapse 방지에 핵심
    ("lambda_balance", {"m_max": 1.3, "grid": [0.85]}),

    # # entropy: 너무 크게 흔들 필요는 없고, 후반 잔여량(m_min)만 조금씩
    ("lambda_entropy", {"m_max": 1.3, "grid": [0.22]}),

    # noise: 초반 탐색(m_max)이 더 중요할 때가 많음 → 여기선 m_max를 grid로 돌리고 m_min은 고정
    # 구현상 통일을 위해 m_min=0.08 고정, m_max를 grid로
    # ("noise_scale", {"m_min": 0.08, "grid": [1.6, 2.2]}),
]

combo_plan = [
    ("B0.85_E0.22_G0.40", {
        "lambda_balance": {"m_min": 0.85, "m_max": 1.30},
        "lambda_entropy": {"m_min": 0.22, "m_max": 1.30},
        "gumbel_scale":   {"m_min": 0.0,  "m_max": 0.40, "mu":0.48, "sigma":0.10, "peak_gain":1.0},
        # aux는 고정 유지 권장
        "aux_loss_weight": {"m_min": 1.0, "m_max": 1.0},
        # temp는 필요 시만 (top-2면 우선 꺼도 됨)
        "temperature": {"min": 1.0, "max": 1.5, "peak_gain": 0.0},
    }),
]


def create_project_config() -> ProjectConfig:
    cfg =  ProjectConfig(
        name="scheduling_combo_test_1",
        seed_list=[11, 21, 47, 121],
        random_trial_counts=20,
        loader_pairs = [
            ("multi", sd_symlink_config_loader),
            # ("nonmoving", sd_symlink_config_loader_n),
            # ("rider", sd_symlink_config_loader_r),
            ],
        dataset_mode="TRAIN",
        param_mode="SCHEDULE",
        # dataset_mode="DEBUG",
        # param_mode="FIXED",
        max_train=3000,
        val_ratio=0.1,
        epochs=30,
        run_analysis=True,
    )
    if cfg.param_mode == "FIXED":
        cfg.moe_fixed_params = MoEParams.fixed(
        aux_loss_weight=0.03,
        lambda_entropy=0.08,
        lambda_balance=2.0,
        noise_scale=0.015,
        )
    return cfg

# dataset_mode:
#   "TRAIN": create new dataset symlink
#   "DEBUG": use recent dataset symlink

# param_mode:
#   "RANDOM": try random combinations of MoE parameters, single dataset
#   "FIXED": try single set of fixed MoE parameters, use multiple dataset
#   "FIXED_LIST": try multiple combinations of fixed MoE Parameters, single dataset

def params_sanity_check(moe_head, moe_params):
    assert moe_head.lambda_entropy == moe_params.lambda_entropy
    assert moe_head.lambda_balance == moe_params.lambda_balance
    assert moe_head.noise_scale == moe_params.noise_scale
    assert moe_head.aux_loss_weight == moe_params.aux_loss_weight
    

def run_benchmark(
    cfg: ProjectConfig,
    moe_params: MoEParams,
    seed: int,
    trial_idx: int | None = None,
    dataset_path: Path | None = None,
    skip_baseline: bool = False,
    domain_name: str | None = None,
    schedule_cfg: dict | None = None
    ):
    common_hp = dict(
        project=str(cfg.results_root),
        data=str(dataset_path),
        epochs=cfg.epochs,
        batch=16,
        imgsz=640,
        seed=seed,
        device=0,
        workers=2,
        cache='disk',
        plots=False,
        # format=None
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

    moe_head.init_from_detect(base_head, moe_params.noise_scale)

    
    print(f"[Seed {seed}][{domain_name}] Random MoE Params = {moe_params}")

    moe.ckpt = True

   
    # 7) MoE 학습
    tag = f"BAL{moe_params.lambda_balance:.2f}_ENT{moe_params.lambda_entropy:.2f}_" \
            f"AUX{moe_params.aux_loss_weight:.3f}_NS{moe_params.noise_scale:.3f}"



    trial_part = f"trial{trial_idx}_" if trial_idx is not None else ""
    # moe_dir_name = f"MoE_{trial_part}{domain_name}_{tag}_s{seed}"

    sched_tag = schedule_cfg.get("tag", "NOSCHED") if schedule_cfg else "NOSCHED"
    moe_dir_name = f"MoE_{trial_part}{domain_name}_{tag}_{sched_tag}_s{seed}"
    
    # MoETrainer.MOE_PARAMS = moe_params
    moe.model.moe_params = moe_params

    moe.model.moe_schedule = schedule_cfg or {}

    # params_sanity_check(moe_head, moe_params)
    
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

    (run_dir / "moe_schedule.json").write_text(json.dumps(schedule_cfg or {}, indent=2))
    # results_base_s = base_s.train(
    #     **common_hp,
    #     name=f"YOLOs_{domain_name}_s{seed}",
    # )

    # Try baseline(YOLO11n) training only once
    if skip_baseline is False:
        results_base = base.train(
            **common_hp,
            name=f"YOLOn_{domain_name}_s{seed}",
        )

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
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

            # baseline은 한 번 돌고 나면 늘 True
            skip_baseline = True

    if cfg.run_analysis:
        run_analysis(Path(cfg.results_root), Path(cfg.analysis_root))
        run_param_analysis(Path(cfg.results_root), Path(cfg.analysis_root), None, True)

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
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

    if cfg.run_analysis:
        run_analysis(Path(cfg.results_root), Path(cfg.analysis_root))


def run_param_fixed_list(cfg: ProjectConfig):
    search_seed = cfg.seed_list[0]
    for param_set in fixed_params_list:
        moe_params = MoEParams.fixed(
            aux_loss_weight = param_set["aux_loss_weight"],
            lambda_entropy = param_set["lambda_entropy"],
            lambda_balance = param_set["lambda_balance"],
            noise_scale = param_set["noise_scale"],
            )
                
        for domain_name, loader in cfg.loader_pairs:
            register_symlink_config_loader(loader)
            if cfg.dataset_mode == "TRAIN":
                dataset_config_path = create_dataset_config(
                    cfg.val_ratio, seed=search_seed,
                    max_train=cfg.max_train, max_val=cfg.max_val
                )
            else:
                dataset_config_path = "/ultralytics/run/sd_moe/multihead_data.yaml"

            base_run_dir = cfg.results_root / f"YOLOn_{domain_name}_s{search_seed}"
            skip_baseline = (base_run_dir / "results.csv").exists()

            run_benchmark(
                cfg=cfg,
                moe_params=moe_params,
                seed=search_seed,
                trial_idx=None,
                skip_baseline=skip_baseline,
                dataset_path=dataset_config_path,
                domain_name=domain_name
            )
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

    if cfg.run_analysis:
        run_analysis(Path(cfg.results_root), Path(cfg.analysis_root))
        run_param_analysis(Path(cfg.results_root), Path(cfg.analysis_root), None, True)

def make_schedule_single_param(
    which: str,
    m_min: float,
    m_max: float,
    *,
    temp_min: float = 1.0,
    temp_max: float = 1.5,
) -> dict:
    # 기본: 모두 고정(배율 1.0)
    sched = {
        "tag": f"SCHED_{which}_min{m_min:.2f}_max{m_max:.2f}",
        "lambda_balance": {"m_min": 1.0, "m_max": 1.0},
        "lambda_entropy": {"m_min": 1.0, "m_max": 1.0},
        "noise_scale": {"m_min": 1.0, "m_max": 1.0},
        "gumbel_scale": {"m_min": 0.0, "m_max": 1.0, "mu": 0.48, "sigma": 0.10, "peak_gain": 1.5},
        "aux_loss_weight": {"m_min": 1.0, "m_max": 1.0},
        "temperature": {"min": temp_min, "max": temp_max},  # 온도는 공통으로 켜도 되고, 끄려면 min=max=1.0
    }

    # 하나만 활성화
    if which not in ("lambda_balance", "lambda_entropy", "gumbel_scale"):
        raise ValueError(f"Unknown param: {which}")

    sched[which] = {"m_min": float(m_min), "m_max": float(m_max)}
    return sched

def make_schedule_combo(tag: str, spec: dict, *, temp_min=1.0, temp_max=1.5) -> dict:
    """
    spec 예시:
      {
        "lambda_balance": {"m_min": 0.85, "m_max": 1.30},
        "lambda_entropy": {"m_min": 0.22, "m_max": 1.30},
        "gumbel_scale":   {"m_min": 0.0,  "m_max": 0.40, "mu":0.48, "sigma":0.10, "peak_gain":1.0},
        # "aux_loss_weight": {"m_min": 1.0, "m_max": 1.0},  # 보통 고정
        # "temperature": {"min":1.0, "max":1.5, "peak_gain":0.0},
      }
    """
    sched = {
        "tag": f"SCHED_COMBO_{tag}",
        # 기본값(없으면 trainer에서 mult() default_min/max=1.0로 처리됨)
        "lambda_balance": {"m_min": 1.0, "m_max": 1.0},
        "lambda_entropy": {"m_min": 1.0, "m_max": 1.0},
        "gumbel_scale":   {"m_min": 1.0, "m_max": 1.0},  # trainer가 mult_peak로 읽을 것
        "aux_loss_weight": {"m_min": 1.0, "m_max": 1.0},
        "temperature": {"min": temp_min, "max": temp_max, "peak_gain": 0.0},
    }

    # 덮어쓰기
    for k, v in spec.items():
        sched[k] = v
    return sched

def run_param_schedule_list(cfg: ProjectConfig):
    for search_seed in cfg.seed_list:
        for param_set in fixed_params_list:
            moe_params = MoEParams.fixed(
                aux_loss_weight = param_set["aux_loss_weight"],
                lambda_entropy = param_set["lambda_entropy"],
                lambda_balance = param_set["lambda_balance"],
                noise_scale = param_set["noise_scale"],
                gumbel_scale = param_set["gumbel_scale"]
                )
            # for which, spec in single_param_plan:
            #     # 케이스 1) m_min을 grid로 돌리는 타입(balance, entropy)
            #     if "m_max" in spec:
            #         m_max = float(spec["m_max"])
            #         for m_min in spec["grid"]:
            #             schedule_cfg = make_schedule_single_param(
            #                 which=which,
            #                 m_min=float(m_min),
            #                 m_max=m_max,
            #                 temp_min=1.0,
            #                 temp_max=1.5,
            #             )

            #             for domain_name, loader in cfg.loader_pairs:
            #                 register_symlink_config_loader(loader)

            #                 if cfg.dataset_mode == "TRAIN":
            #                     dataset_config_path = create_dataset_config(
            #                         cfg.val_ratio, seed=search_seed,
            #                         max_train=cfg.max_train, max_val=cfg.max_val
            #                     )
            #                 else:
            #                     dataset_config_path = "/ultralytics/run/sd_moe/multihead_data.yaml"

            #                 base_run_dir = cfg.results_root / f"YOLOn_{domain_name}_s{search_seed}"
            #                 skip_baseline = (base_run_dir / "results.csv").exists()

            #                 run_benchmark(
            #                     cfg=cfg,
            #                     moe_params=moe_params,
            #                     schedule_cfg=schedule_cfg,   # ✅ 전달
            #                     seed=search_seed,
            #                     trial_idx=None,
            #                     skip_baseline=skip_baseline,
            #                     dataset_path=dataset_config_path,
            #                     domain_name=domain_name,
            #                 )
            #                 gc.collect()
            #                 torch.cuda.empty_cache()
            #                 time.sleep(2)

                # 케이스 2) m_max를 grid로 돌리는 타입(noise)
                # else:
                #     m_min = float(spec["m_min"])
                #     for m_max in spec["grid"]:
                #         schedule_cfg = make_schedule_single_param(
                #             which=which,
                #             m_min=m_min,
                #             m_max=float(m_max),
                #             temp_min=1.0,
                #             temp_max=1.5,
                #         )

                #         for domain_name, loader in cfg.loader_pairs:
                #             register_symlink_config_loader(loader)

                #             if cfg.dataset_mode == "TRAIN":
                #                 dataset_config_path = create_dataset_config(
                #                     cfg.val_ratio, seed=search_seed,
                #                     max_train=cfg.max_train, max_val=cfg.max_val
                #                 )
                #             else:
                #                 dataset_config_path = "/ultralytics/run/sd_moe/multihead_data.yaml"

                #             base_run_dir = cfg.results_root / f"YOLOn_{domain_name}_s{search_seed}"
                #             skip_baseline = (base_run_dir / "results.csv").exists()

                #             run_benchmark(
                #                 cfg=cfg,
                #                 moe_params=moe_params,
                #                 schedule_cfg=schedule_cfg,   # ✅ 전달
                #                 seed=search_seed,
                #                 trial_idx=None,
                #                 skip_baseline=skip_baseline,
                #                 dataset_path=dataset_config_path,
                #                 domain_name=domain_name,
                #             )
                #             gc.collect()
                #             torch.cuda.empty_cache()
                #             time.sleep(2)

            for combo_tag, combo_spec in combo_plan:
                schedule_cfg = make_schedule_combo(combo_tag, combo_spec, temp_min=1.0, temp_max=1.5)

                for domain_name, loader in cfg.loader_pairs:
                    register_symlink_config_loader(loader)
                    if cfg.dataset_mode == "TRAIN":
                        dataset_config_path = create_dataset_config(
                            cfg.val_ratio, seed=search_seed,
                            max_train=cfg.max_train, max_val=cfg.max_val
                        )
                    else:
                        dataset_config_path = "/ultralytics/run/sd_moe/multihead_data.yaml"

                    base_run_dir = cfg.results_root / f"YOLOn_{domain_name}_s{search_seed}"
                    skip_baseline = (base_run_dir / "results.csv").exists()

                    run_benchmark(
                        cfg=cfg,
                        moe_params=moe_params,
                        schedule_cfg=schedule_cfg,
                        seed=search_seed,
                        trial_idx=None,
                        skip_baseline=skip_baseline,
                        dataset_path=dataset_config_path,
                        domain_name=domain_name,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(2)

    if cfg.run_analysis:
        run_analysis(Path(cfg.results_root), Path(cfg.analysis_root))
        run_param_analysis(Path(cfg.results_root), Path(cfg.analysis_root), None, True)

def main():
    cfg = create_project_config()
    install_freeze_warning_filter()
    # Random parameter sampling. Dataset fixed
    if cfg.param_mode == "RANDOM":
        run_random_param_search(cfg)

    # Random dataset sampling. MoE parameters fixed
    elif cfg.param_mode == "FIXED":
        run_param_fixed_eval(cfg)

    elif cfg.param_mode == "FIXED_LIST":
        run_param_fixed_list(cfg)

    elif cfg.param_mode == "SCHEDULE":
        run_param_schedule_list(cfg)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()