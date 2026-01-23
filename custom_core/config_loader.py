from typing import Callable, Optional
from .dataclass.symlink_config import SymlinkConfig


_CONFIG_LOADER: Optional[Callable[[], SymlinkConfig]] = None

def register_symlink_config_loader(loader: Callable[[], SymlinkConfig]) -> None:
    """
    외부(scripts)에서 SymlinkConfig를 제공하는 loader를 등록.
    """
    global _CONFIG_LOADER
    _CONFIG_LOADER = loader

def get_symlink_config() -> SymlinkConfig:
    if _CONFIG_LOADER is None:
        raise RuntimeError("SymlinkConfig loader가 등록되지 않았습니다. "
                           "register_symlink_config_loader(...)를 먼저 호출하세요.")
    return _CONFIG_LOADER()