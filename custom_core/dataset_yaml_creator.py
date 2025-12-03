from pathlib import Path
from dataclasses import dataclass
from auto_symlink_helper import LabelConfig

@dataclass
class LocalClasses:
    label_name: str
    local_names: list[str]
    @property
    def nc(self) -> int:
        return len(self.local_names)


def _build_local_label_class(cfg: LabelConfig):
    local_names = []
    for line in cfg.class_list_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split(maxsplit=1)

        if len(parts) == 1:
            token = parts[0]
            # line without class name (only index)
            if token.isdigit():
                name = "unused"
            # line without class index (only name)
            else:
                name = token
        else:
            # "0 Column" → name="Column"
            _, name = parts

        local_names.append(name)

    return LocalClasses(label_name = cfg.name, local_names = local_names)


def _build_local_structs(label_configs: list[LabelConfig]) -> list[LocalClasses]:
    local_structs_list = []
    for cfg in label_configs:
        local_structs_list.append(_build_local_label_class(cfg))
    return local_structs_list

def data_yaml_writer(VIRTUAL_ROOT: Path, LABEL_CONFIGS: list[LabelConfig]):
    
    DATASET_YAML = VIRTUAL_ROOT / "multihead_data.yaml"

    local_structs_list = _build_local_structs(LABEL_CONFIGS)
    nc_sum = sum(ls.nc for ls in local_structs_list)

    _content: list[str] = []
    _content.append(f"path: {VIRTUAL_ROOT}")
    _content.append("train: images/train")
    _content.append("val: images/val")
    _content.append("")
    _content.append(f"nc: {nc_sum}")
    _content.append("names:")
    
    _idx = 0
    for local_struct in local_structs_list:
        for name in local_struct.local_names:
            _content.append(f"    {_idx}: {name}")
            _idx += 1
    
    for local_struct in local_structs_list:
        _content.append(f"nc_{local_struct.label_name}: {local_struct.nc}")
    
    for local_struct in local_structs_list:
        _content.append(f"names_{local_struct.label_name}:")
        for i, name in enumerate(local_struct.local_names):
            _content.append(f"    {i}: {name}")

    _content.append("channels: 3")

    DATASET_YAML.write_text("\n".join(_content) + "\n", encoding="utf-8")
    print(f"[SymlinkHelper] data.yaml 생성 완료: {DATASET_YAML}")
    return DATASET_YAML