from dataclasses import dataclass


@dataclass
class LocalClasses:
    domain_name: str
    local_ids: list[int]
    local_names: list[str]
    @property
    def nc(self) -> int:
        return len(self.local_ids)