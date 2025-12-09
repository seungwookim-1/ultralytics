from pathlib import Path
from custom_core.dataclass.domain_config import DomainConfig
from custom_core.dataclass.symlink_config import SymlinkConfig


DATA_ROOT = Path("/mnt/nas/AI_hub")
VIRTUAL_ROOT = Path("/ultralytics/run/sd_moe")

VIRTUAL_IMAGES_TRAIN = VIRTUAL_ROOT / "images" / "train"
VIRTUAL_IMAGES_VAL   = VIRTUAL_ROOT / "images" / "val"
VIRTUAL_LABELS_TRAIN = VIRTUAL_ROOT / "labels" / "train"
VIRTUAL_LABELS_VAL = VIRTUAL_ROOT / "labels" / "val"

DATASET_YAML = VIRTUAL_ROOT / "multihead_data.yaml"

DOMAIN_CONFIGS: list[DomainConfig] = [
    DomainConfig(
        name="nonmoving",
        root = DATA_ROOT / "Camera-nonmoving",
        class_list_file = DATA_ROOT / "Camera-nonmoving" / "classes_nonmoving.txt",
        max_images = 10000
    ),
    DomainConfig(
        name="rider",
        root = DATA_ROOT / "Camera-rider",
        class_list_file = DATA_ROOT / "Camera-rider" / "classes.txt",
        max_images = 10000
    )
]

def sd_symlink_config_loader() -> SymlinkConfig:
    return SymlinkConfig(
        virtual_root=VIRTUAL_ROOT,
        images_train=VIRTUAL_IMAGES_TRAIN,
        images_val=VIRTUAL_IMAGES_VAL,
        labels_train=VIRTUAL_LABELS_TRAIN,
        labels_val=VIRTUAL_LABELS_VAL,
        domain_configs=DOMAIN_CONFIGS,
        dataset_yaml=DATASET_YAML
    )
