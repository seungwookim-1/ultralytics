from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple
from cfg.sd_config import sd_symlink_config_loader
from random_param_search import MoEParams

@dataclass
class ProjectConfig:
    name: str = "multi_11n_moe_param_test_0"
    root_dir: str = "/ultralytics/outputs"

    seed_list: List[int] = None
    random_trial_counts: int = 10
    loader_pairs: List[Tuple[str, Any]] = None

    dataset_mode: str = "TRAIN"   # "TRAIN" | "DEBUG"
    param_mode: str = "RANDOM"    # "RANDOM" | "FIXED"

    max_train: int = 5000
    val_ratio: float = 0.1
    epochs: int = 15
    
    run_analysis: bool = False

    moe_fixed_params: Optional[MoEParams] = None

    def __post_init__(self):
        if self.seed_list is None:
            self.seed_list = [1]
        if self.loader_pairs is None:
            self.loader_pairs = [("multi", sd_symlink_config_loader)]

    @property
    def project_root(self) -> Path:
        return Path(self.root_dir) / self.name

    @property
    def results_root(self) -> Path:
        return self.project_root / "results"

    @property
    def analysis_root(self) -> Path:
        return self.project_root

    @property
    def max_val(self) -> int:
        return int(self.max_train * self.val_ratio)
