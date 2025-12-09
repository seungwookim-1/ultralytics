from ultralytics import YOLO
from ultralytics.nn.modules.head import MoEDetect
from custom_core import MoETrainer, create_dataset_config, install_freeze_warning_filter
from custom_core.config_loader import register_symlink_config_loader
from cfg.sd_config import sd_symlink_config_loader
from analyze_results import run_analysis

def main():
    project_name = "/ultralytics/runs/multi_11n_11s_moe"
    result_dir = "/ultralytics/outputs/multi_11n_11s_moe"
    install_freeze_warning_filter()
    # 0) 데이터 경로 등 인자 loader 등록
    register_symlink_config_loader(sd_symlink_config_loader)

    # 1) dataset 고정
    dataset_config_path = create_dataset_config(
        val_ratio=0.1, seed=42, max_train=1000, max_val=100
    )
    dataset_config_path = "/ultralytics/run/sd_moe/multihead_data.yaml"

    common_hp = dict(
        data=str(dataset_config_path),
        epochs=30,
        batch=16,
        imgsz=640,
        seed=42,
    )

    # 2) baseline, MoE 모델 로드
    base = YOLO("yolo11n.pt")          # COCO pretrain 완료 모델
    moe  = YOLO("yolo11-moe.yaml")     # 구조만 정의된 MoE

    # 3) ★ backbone 포함 전체 가중치를 MoE로 복사 (strict=False)
    moe.model.load_state_dict(base.model.state_dict(), strict=False)

    # 4) Detect head → MoE head 복사
    base_head = base.model.model[-1]
    moe_head  = moe.model.model[-1]
    assert isinstance(moe_head, MoEDetect)
    moe_head.init_from_detect(base_head, noise_scale=0.01) 

    moe.ckpt = True

    # 7) MoE 학습

    results_moe = moe.train(
        trainer=MoETrainer,   # 여기서 teacher를 model에 붙여주기만 함
        # moe_kd_weight=1.0,    # model.args로 들어가서 MoEDetectionLoss가 읽음
        # moe_kd_temp=1.0,
        **common_hp,
    )

    # Analyze expert specialization
    print("Expert usage (global):", moe_head.expert_counts)

    if hasattr(moe_head, "expert_counts_nonmoving"):
        print("Expert usage (nonmoving):", moe_head.expert_counts_nonmoving)
    if hasattr(moe_head, "expert_counts_rider"):
        print("Expert usage (rider):", moe_head.expert_counts_rider)

    run_analysis(project_name, result_dir)

if __name__ == "__main__":
    main()