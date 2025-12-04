from dataclasses import dataclass
from pathlib import Path


@dataclass
class DomainConfig:
    name: str
    root: Path
    class_list_file: Path
    max_images: int | None = None
    augmentation: float | None = None
