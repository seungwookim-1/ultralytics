from custom_core.dataclass.symlink_config import SymlinkConfig
from .dataclass.domain_config import DomainConfig
from .dataclass.global_schema import GlobalSchema
from .dataclass.local_classes import LocalClasses


def _build_local_domain_class(cfg: DomainConfig):
    local_ids = []
    local_names = []
    for line in cfg.class_list_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split(maxsplit=1)

        if parts[0].isdigit():
            local_id = int(parts[0])
            # line without class name (only index)
            if len(parts) == 1 or not parts[1].strip():
                name = "unused"
            else:
                name = parts[1].strip()

        # line without class index (only name)
        else:
            local_id = len(local_ids)
            name = line
        
        local_ids.append(local_id)
        local_names.append(name)

    return LocalClasses(domain_name = cfg.name, local_ids = local_ids, local_names = local_names)


def _build_local_structs(domain_configs: list[DomainConfig]) -> list[LocalClasses]:
    local_structs = []
    for cfg in domain_configs:
        local_structs.append(_build_local_domain_class(cfg))
    return local_structs


def build_global_schema(sym_cfg: SymlinkConfig) -> tuple[GlobalSchema, list[LocalClasses]]:
    local_structs = _build_local_structs(sym_cfg.domain_configs)

    global_names: list[str] = []
    mapping: dict[tuple[str, int], int] = {}
    per_domain_nc: dict[str, int] = {}

    for domain in local_structs:
        per_domain_nc[domain.domain_name] = domain.nc
        for l_id, name in zip(domain.local_ids, domain.local_names):
            g_id = len(global_names)
            mapping[(domain.domain_name, l_id)] = g_id
            global_names.append(name)

    schema = GlobalSchema(
        global_names=global_names,
        mapping=mapping,
        per_domain_nc=per_domain_nc,
    )
    return schema, local_structs
