from pathlib import Path
import random
import shutil

DATA_ROOT = Path("/mnt/nas/AI_hub")
VIRTUAL_ROOT = Path("/ultralytics/run/sd_single")

VIRTUAL_IMAGES_TRAIN = VIRTUAL_ROOT / "images" / "train"
VIRTUAL_IMAGES_VAL   = VIRTUAL_ROOT / "images" / "val"
VIRTUAL_LABELS_TRAIN = VIRTUAL_ROOT / "labels" / "train"
VIRTUAL_LABELS_VAL = VIRTUAL_ROOT / "labels" / "val"

# 멀티헤드용 data.yaml 경로
DATASET_YAML = VIRTUAL_ROOT / "data.yaml"

# 원본 nonmoving 이미지/라벨 디렉토리
NONMOVING_IMAGE_DIR = DATA_ROOT / "Camera-nonmoving" / "images"
NONMOVING_LABEL_DIR = DATA_ROOT / "Camera-nonmoving" / "labels"

def _prepare_virtual_dirs():
    for split_dir in (VIRTUAL_IMAGES_TRAIN, VIRTUAL_IMAGES_VAL, VIRTUAL_LABELS_TRAIN, VIRTUAL_LABELS_VAL):
        if split_dir.exists():
            print(f"[Yolo] 기존 split 삭제: {split_dir}")
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def _create_safe_symlink(
    stems,
    nonmoving_image_map,
    nonmoving_label_map,
    split: str
):
    image_root = VIRTUAL_IMAGES_TRAIN if split == "train" else VIRTUAL_IMAGES_VAL
    label_root = VIRTUAL_LABELS_TRAIN if split == "train" else VIRTUAL_LABELS_VAL

    count = 0
    for stem in stems:
        # 이미지 src 결정
        if stem in nonmoving_image_map:
            image_src = nonmoving_image_map[stem]
        else:
            continue  # stem 엉킨 케이스

        image_dst = image_root / f"{stem}{image_src.suffix}"
        image_dst.symlink_to(image_src)

        # 라벨 src 결정
        label_src = None
        if stem in nonmoving_label_map:
            label_src = nonmoving_label_map[stem]

        if label_src:
            label_dst = label_root / f"{stem}{label_src.suffix}"
            label_dst.symlink_to(label_src)
            count += 1
        else:
            pass

    return count


def balanced_sample(group_A, group_B, group_C, target, rng: random.Random):
    # 추천 비율: C 50%, A 25%, B 25%
    c_target = int(target * 0.5)
    a_target = int(target * 0.25)
    b_target = int(target * 0.25)

    sampled_C = group_C[:c_target]
    sampled_A = group_A[:a_target]
    sampled_B = group_B[:b_target]

    # 부족분 채우기
    remaining = target - (len(sampled_C) + len(sampled_A) + len(sampled_B))

    if remaining > 0:
        pool = group_C[c_target:] + group_A[a_target:] + group_B[b_target:]
        rng.shuffle(pool)
        sampled_extra = pool[:remaining]
    else:
        sampled_extra = []

    return sampled_C + sampled_A + sampled_B + sampled_extra


def _sample_paired_from_labels(
    image_root: Path,
    label_root: Path,
    target_pairs: int,
    rng: random.Random,
    max_scan_labels: int = 20000,
    exts=(".jpg", ".jpeg", ".png"),
):
    """
    라벨 디렉토리에서 최대 max_scan_labels개까지만 훑고,
    그 안에서 이미지-라벨 쌍이 실제로 존재하는 stem을 골라
    최대 target_pairs개까지만 반환.
    """
    label_paths: list[Path] = []
    # 라벨은 어차피 .txt만 본다고 가정
    for i, p in enumerate(label_root.rglob("*.txt")):
        label_paths.append(p)
        if i + 1 >= max_scan_labels:
            break

    if not label_paths:
        print(f"[Yolo] 라벨 디렉토리 비어있음: {label_root}")
        return {}, {}

    rng.shuffle(label_paths)

    image_map: dict[str, Path] = {}
    label_map: dict[str, Path] = {}

    for lp in label_paths:
        stem = lp.stem

        img_path = None
        for ext in exts:
            cand = image_root / f"{stem}{ext}"
            if cand.exists():
                img_path = cand
                break

        if img_path is None:
            continue  # 이 라벨과 짝인 이미지는 현재 root에 없음

        image_map[stem] = img_path
        label_map[stem] = lp

        if len(image_map) >= target_pairs:
            break

    print(
        f"[Yolo] {image_root.name}: 라벨 기반 페어 {len(image_map)}개 확보 "
        f"(target_pairs={target_pairs}, scan_labels={len(label_paths)})"
    )
    return image_map, label_map

def create_dataset_config(val_ratio: float = 0.1,
                          seed: int = 42,
                          max_train: int | None = None,
                          max_val: int | None = None,
                          **kwargs) -> Path:
    """
    /mnt/nas/AI_hub 기반 멀티헤드용 train/val split 및 data.yaml 생성.

    Args:
        val_ratio (float): 전체 이미지 중 validation 비율
        seed (int): 랜덤 시드
        **kwargs: (옛날 DB 버전과의 호환용, 현재는 사용 안함)

    Returns:
        Path: 생성된 data.yaml 경로 (/mnt/nas/AI_hub/multihead_data.yaml)
    """
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

    # 1) nonmoving 라벨 기반으로 페어 샘플링
    nonmoving_image_map, nonmoving_label_map = _sample_paired_from_labels(
        NONMOVING_IMAGE_DIR, NONMOVING_LABEL_DIR,
        target_pairs=buffered,
        rng=rng,
        max_scan_labels=20000,  # 디버깅용이면 더 줄여도 됨(예: 5000)
    )

    # 2) 둘 다 완전히 비었다면 에러
    if not nonmoving_image_map:
        raise RuntimeError("nonmoving에서 이미지-라벨 페어를 전혀 찾지 못했습니다.")

    # 3) union stem으로 전체 풀 구성
    all_stems = sorted(set(nonmoving_image_map.keys()))
    rng.shuffle(all_stems)

    # 4) train / val split
    if max_val is not None:
        num_val = min(max_val, max(1, int(len(all_stems) * val_ratio)))
    else:
        num_val = max(1, int(len(all_stems) * val_ratio))

    val_stems_raw = all_stems[:num_val]
    train_stems_raw = all_stems[num_val:]

    # (선택) balanced_sample 유지하고 싶으면 group_A/B/C는 여기서 다시 구성
    group_A, group_B, group_C = [], [], []
    for stem in all_stems:
        has_nm = stem in nonmoving_image_map
        if has_nm:
            group_C.append(stem)

    rng.shuffle(group_C)

    if max_train is not None:
        train_stems = group_C[:max_train]
    else:
        train_stems = group_C

    rest = [s for s in group_C if s not in train_stems]
    rng.shuffle(rest)
    if max_val is not None:
        val_stems = rest[:max_val]
    else:
        val_stems = rest[:int(len(all_stems)*val_ratio)]

    print(
        f"[Yolo] split: train {len(train_stems)}개, val {len(val_stems)}개 "
        f"(val_ratio={val_ratio}, max_train={max_train}, max_val={max_val})"
    )

    # 5) 여기서부터는 기존 로직 그대로
    _prepare_virtual_dirs()

    train_count = _create_safe_symlink(
        train_stems,
        nonmoving_image_map,
        nonmoving_label_map,
        split="train",
    )
    val_count = _create_safe_symlink(
        val_stems,
        nonmoving_image_map,
        nonmoving_label_map,
        split="val",
    )

    print(f"[Yolo] 링크 생성 결과: train {train_count}개, val {val_count}개")

    # 4) 멀티헤드용 data.yaml 생성
    yaml_content = f"""
path: {VIRTUAL_ROOT}
train: images/train
val: images/val

nc: 35

names:
    0: Background  # 혹은 원래 정의에 맞는 이름(비어 있으면 임시로 Background 정도)
    1: Column
    2: Horizontal member
    3: Guidepost
    4: Seagull sign
    5: Signpost
    6: Obstacle target sign
    7: Structure paint & Hatching sign
    8: Guide post
    9: Lighting facility
    10: Road reflector
    11: Speed bump
    12: Median barrier
    13: Guardrail
    14: Impact absorbing facility
    15: Rockfall net
    16: Rockfall barrier
    17: Rockfall prevention wall
    18: Vegetation method
    19: Bridge
    20: Tunnel
    21: Underpass
    22: Overpass
    23: Interchange
    24: Underground passage
    25: Footbridge
    26: Station
    27: Traffic signal
    28: Road sign
    29: Safety sign
    30: Road name sign
    31: Emergency contact facility
    32: CCTV
    33: Road electronic sign
    34: Road mileage sign

multi_heads:
  nonmoving:
    class_ids: [{", ".join(str(i) for i in range(1, 35))}]
    
channels: 3
"""
    DATASET_YAML.write_text(yaml_content.strip() + "\n", encoding="utf-8")
    print(f"[Yolo] data.yaml 생성 완료: {DATASET_YAML}")

    return DATASET_YAML
