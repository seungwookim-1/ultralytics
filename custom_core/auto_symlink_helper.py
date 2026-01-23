from pathlib import Path
import random
import shutil

from custom_core.data_splitter import build_maps_and_splits
from .config_loader import get_symlink_config
from .dataset_yaml_creator import write_dataset_yaml
from .global_schema_builder import build_global_schema
from .symlink_maker import create_safe_symlink


def _prepare_virtual_dirs():
    sym_cfg = get_symlink_config()
    for split_dir in (sym_cfg.images_train, sym_cfg.images_val, sym_cfg.labels_train, sym_cfg.labels_val):
        if split_dir.exists():
            print(f"[SymlinkHelper] 기존 split 삭제: {split_dir}")
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def create_dataset_config(val_ratio: float = 0.1,
                          seed: int = 42,
                          max_train: int | None = None,
                          max_val: int | None = None,
                          **kwargs) -> Path:
    """
    /mnt/nas/AI_hub 기반 멀티헤드용 train/val split 및 data.yaml 생성.

    Args:
        val_ratio (float): 전체 이미지 중 validation 비율
        seed (int): 랜덤 시드
        **kwargs: (옛날 DB 버전과의 호환용, 현재는 사용 안함)

    Returns:
        Path: 생성된 data.yaml 경로 (/mnt/nas/AI_hub/multihead_data.yaml)
    """
    sym_cfg = get_symlink_config()
    schema, local_structs = build_global_schema(sym_cfg)
    
    (domain_image_maps, domain_label_maps, train_stems, val_stems) = build_maps_and_splits(sym_cfg, val_ratio, seed, max_train, max_val)

    _prepare_virtual_dirs()

    train_count = create_safe_symlink(
        train_stems,
        domain_image_maps=domain_image_maps,
        domain_label_maps=domain_label_maps,
        schema=schema,
        split="train",
    )
    val_count = create_safe_symlink(
        val_stems,
        domain_image_maps=domain_image_maps,
        domain_label_maps=domain_label_maps,
        schema=schema,
        split="val",
    )
    
    print(f"[SymlinkHelper] 링크 생성 결과: train {train_count}개, val {val_count}개")
    DATASET_YAML = write_dataset_yaml(sym_cfg=sym_cfg, schema=schema, local_structs=local_structs)

    return DATASET_YAML
