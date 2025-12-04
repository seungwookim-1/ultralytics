from dataclasses import dataclass
from pathlib import Path
from typing import List
from .domain_config import DomainConfig

@dataclass
class SymlinkConfig:
    virtual_root: Path
    images_train: Path
    images_val: Path
    labels_train: Path
    labels_val: Path
    dataset_yaml: Path
    domain_configs: List[DomainConfig]
    