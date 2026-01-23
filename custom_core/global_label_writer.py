from pathlib import Path
from .dataclass.global_schema import GlobalSchema


def remap_and_write_label(
    domain_name: str,
    src: Path,
    dst: Path,
    schema: GlobalSchema,
) -> None:
    lines_out: list[str] = []

    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            local_id = int(parts[0])
        except ValueError:
            # class id가 숫자가 아니면 스킵 or 경고
            continue

        key = (domain_name, local_id)
        if key not in schema.mapping:
            # 정의되지 않은 라벨 → 스킵 or 경고
            continue

        global_id = schema.mapping[key]
        parts[0] = str(global_id)
        lines_out.append(" ".join(parts))

    if lines_out:
        dst.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
