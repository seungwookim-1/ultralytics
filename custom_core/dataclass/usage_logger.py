from dataclasses import dataclass
from typing import Any, List, Optional

@dataclass
class MoEUsageRecord:
    epoch: int
    global_usage: List[List[float]]
    nonmoving_usage: Optional[List[float]] = None
    rider_usage: Optional[List[float]] = None
