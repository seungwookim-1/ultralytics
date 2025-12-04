from custom_core.dataclass.global_schema import GlobalSchema
from custom_core.dataclass.symlink_config import SymlinkConfig
from .dataclass.local_classes import LocalClasses


def write_dataset_yaml(sym_cfg: SymlinkConfig, schema: GlobalSchema, local_structs=list[LocalClasses]):
    _content: list[str] = []
    _content.append(f"path: {sym_cfg.virtual_root}")
    _content.append("train: images/train")
    _content.append("val: images/val")
    _content.append("")
    _content.append(f"nc: {len(schema.global_names)}")
    _content.append("names:")
    
    for g_id, name in enumerate(schema.global_names):
        _content.append(f"    {g_id}: {name}")
    _content.append("")

    for domain in local_structs:
        _content.append(f"nc_{domain.domain_name}: {domain.nc}")
    _content.append("")

    for domain in local_structs:
        _content.append(f"names_{domain.domain_name}:")
        for l_id, name in zip(domain.local_ids, domain.local_names):
            _content.append(f"    {l_id}: {name}")
        _content.append("")
        
    _content.append("channels: 3")

    sym_cfg.dataset_yaml.write_text("\n".join(_content) + "\n", encoding="utf-8")
    print(f"[SymlinkHelper] data.yaml 생성 완료: {sym_cfg.dataset_yaml}")
    return sym_cfg.dataset_yaml
