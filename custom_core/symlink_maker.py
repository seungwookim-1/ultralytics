from pathlib import Path

from custom_core.dataclass.global_schema import GlobalSchema
from .config_loader import get_symlink_config
from .global_label_writer import remap_and_write_label


def create_safe_symlink(
    stems: list[str],
    domain_image_maps: dict[str, dict[str, Path]],
    domain_label_maps: dict[str, dict[str, Path]],
    schema: GlobalSchema,
    split: str,
):
    sym_cfg = get_symlink_config()

    image_root = sym_cfg.images_train if split == "train" else sym_cfg.images_val
    label_root = sym_cfg.labels_train if split == "train" else sym_cfg.labels_val

    domain_list = [cfg.name for cfg in sym_cfg.domain_configs]
    count = 0

    for stem in stems:
        # 이미지 src 결정
        image_src: Path | None = None
        for domain_name in domain_list:
            image_map = domain_image_maps.get(domain_name, {})
            if stem in image_map:
                image_src = image_map[stem]
                break
        if image_src is None:
            continue

        image_dst = image_root / f"{stem}{image_src.suffix}"
        image_dst.symlink_to(image_src)

        label_src: Path | None = None
        label_domain: str | None = None
        for domain_name in domain_list:
            label_map = domain_label_maps.get(domain_name, {})
            if stem in label_map:
                label_src = label_map[stem]
                label_domain = domain_name
                break

        if label_src is not None and label_domain is not None:
            label_dst = label_root / f"{stem}{label_src.suffix}"
            remap_and_write_label(label_domain, label_src, label_dst, schema)
            count += 1

    return count