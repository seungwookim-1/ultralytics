from dataclasses import dataclass

@dataclass
class GlobalSchema:
    global_names: list[str]                             # global_id -> name
    mapping: dict[tuple[str, int], int]                 # (domain_name, local_id) -> global_id
    per_domain_nc: dict[str, int]                       # "nonmoving" -> 35, "rider" -> 19
    