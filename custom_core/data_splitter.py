from pathlib import Path
import random
from custom_core.dataclass.symlink_config import SymlinkConfig
from .data_pair_collector import collect_image_label_pairs
from .data_balancer import select_balanced_stems


def build_maps_and_splits(
    sym_cfg: SymlinkConfig,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_train: int | None = None,
    max_val: int | None = None
    ):

    rng = random.Random(seed)
    # 0) 이번에 실제로 필요한 페어 개수 계산
    total_needed = 0
    if max_train:
        total_needed += max_train
    if max_val:
        total_needed += max_val
    if total_needed == 0:
        total_needed = 1100  # 디폴트 디버그용

    # 여유 버퍼 (조금 더 많이 확보해두고 그 안에서 train/val 나눔)
    buffered = int(total_needed * 1.5)

    # 1) 라벨 기반으로 페어 샘플링
    domain_image_maps : dict[str, dict[str, Path]] = {}
    domain_label_maps : dict[str, dict[str, Path]] = {}

    for cfg in sym_cfg.domain_configs:
        image_dir = cfg.root / "images"
        label_dir = cfg.root / "labels"

        domain_target = buffered
        if cfg.max_images is not None:
            domain_target = min(buffered, cfg.max_images)
        
        image_map, label_map = collect_image_label_pairs(
            image_dir,
            label_dir,
            target_pairs = domain_target,
            rng = rng,
            max_scan_labels=20000)
        domain_image_maps[cfg.name] = image_map
        domain_label_maps[cfg.name] = label_map

    # 2) 완전히 비었다면 에러
    if all(not map for map in domain_image_maps.values()):
        raise RuntimeError("어떠한 domain에서도 이미지-라벨 페어를 찾지 못했습니다.")

    # 3) union stem으로 전체 풀 구성
    all_stem_sets = [set(map.keys()) for map in domain_image_maps.values() if map]
    all_stems = sorted(set().union(*all_stem_sets))
    rng.shuffle(all_stems)

    # 4) train / val split
    if max_val is not None:
        num_val = min(max_val, max(1, int(len(all_stems) * val_ratio)))
    else:
        num_val = max(1, int(len(all_stems) * val_ratio))

    val_stems_raw = all_stems[:num_val]
    train_stems_raw = all_stems[num_val:]

    if max_train is not None:
        train_stems = select_balanced_stems(train_stems_raw, domain_image_maps, max_train, rng)
    else:
        train_stems = train_stems_raw

    if max_val is not None:
        val_stems = select_balanced_stems(val_stems_raw, domain_image_maps, max_val, rng)
    else:
        val_stems = val_stems_raw

    print(
        f"[SymlinkHelper] split: train {len(train_stems)}개, val {len(val_stems)}개 "
        f"(val_ratio={val_ratio}, max_train={max_train}, max_val={max_val})"
    )

    return domain_image_maps, domain_label_maps, train_stems, val_stems
