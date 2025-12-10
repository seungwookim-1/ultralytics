from ultralytics import YOLO
from ultralytics.nn.modules.head import MoEDetect
from custom_core import MoETrainer, create_dataset_config, install_freeze_warning_filter
from custom_core.config_loader import register_symlink_config_loader
from cfg.sd_config import sd_symlink_config_loader
# from cfg.sd_config_single_nonmoving import sd_symlink_config_loader_n
# from cfg.sd_config_single_rider import sd_symlink_config_loader_r
from analyze_results import run_analysis


def main():
    _name = "multi_11n_moe_5k"
    project_root = f"/ultralytics/runs/{_name}"
    results_root = f"/ultralytics/runs/{_name}/results"
    analysis_root = project_root
    seed_list = [17, 23, 121]
    loader_pairs = [
        ("multi", sd_symlink_config_loader),
        # ("nonmoving", sd_symlink_config_loader_n),
        # ("rider", sd_symlink_config_loader_r),
    ]
    dataset_mode = "TRAIN"
    # dataset_mode = "DEBUG"
    max_train = 5000
    val_ratio = 0.1
    max_val = (int) (max_train * val_ratio)
    epochs = 50

    for seed in seed_list:
        for domain_name, loader in loader_pairs:
            install_freeze_warning_filter()
            # 0) 데이터 경로 등 인자 loader 등록
            register_symlink_config_loader(loader)

            # 1) dataset 고정
            if dataset_mode == "TRAIN":
                dataset_config_path = create_dataset_config(
                    val_ratio=val_ratio, seed=seed, max_train=max_train, max_val=max_val
                )
            # debug: 마지막으로 생성된 dataset 사용
            elif dataset_mode == "DEBUG":
                dataset_config_path = "/ultralytics/run/sd_moe/multihead_data.yaml"

            common_hp = dict(
                project=results_root,
                data=str(dataset_config_path),
                epochs=epochs,
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
            moe_head.init_from_detect(base_head, noise_scale=0.01) 

            moe.ckpt = True

            results_base = base.train(
                **common_hp,
                name=f"YOLOn_{domain_name}_s{seed}",
            )
            
            # 7) MoE 학습
            results_moe = moe.train(
                trainer=MoETrainer,   # 여기서 teacher를 model에 붙여주기만 함
                # moe_kd_weight=1.0,    # model.args로 들어가서 MoEDetectionLoss가 읽음
                # moe_kd_temp=1.0,
                **common_hp,
                name=f"MoE_{domain_name}_s{seed}",
            )

            # results_base_s = base_s.train(
            #     **common_hp,
            #     name=f"YOLOs_{domain_name}_s{seed}",
            # )   
    run_analysis(results_root, analysis_root)


if __name__ == "__main__":
    main()