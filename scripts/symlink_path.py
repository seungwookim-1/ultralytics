from pathlib import Path
import random
import shutil

DATA_ROOT = Path("/mnt/nas/AI_hub")
VIRTUAL_ROOT = Path("/ultralytics/run/sd_moe")

VIRTUAL_IMAGES_TRAIN = VIRTUAL_ROOT / "images" / "train"
VIRTUAL_IMAGES_VAL   = VIRTUAL_ROOT / "images" / "val"
VIRTUAL_LABELS_TRAIN = VIRTUAL_ROOT / "labels" / "train"
VIRTUAL_LABELS_VAL = VIRTUAL_ROOT / "labels" / "val"

# 멀티헤드용 data.yaml 경로
DATASET_YAML = VIRTUAL_ROOT / "multihead_data.yaml"

# 원본 nonmoving / rider 이미지/라벨 디렉토리
NONMOVING_IMAGE_DIR = DATA_ROOT / "Camera-nonmoving" / "images"
NONMOVING_LABEL_DIR = DATA_ROOT / "Camera-nonmoving" / "labels"

RIDER_IMAGE_DIR = DATA_ROOT / "Camera-rider" / "images"
RIDER_LABEL_DIR = DATA_ROOT / "Camera-rider" / "labels"


def _prepare_virtual_dirs():
    for split_dir in (VIRTUAL_IMAGES_TRAIN, VIRTUAL_IMAGES_VAL, VIRTUAL_LABELS_TRAIN, VIRTUAL_LABELS_VAL):
        if split_dir.exists():
            print(f"[SymlinkHelper] 기존 split 삭제: {split_dir}")
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)



def _scan_files(root: Path, exts=(".jpg", ".jpeg", ".png", ".txt"), max_files: int | None = None) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    if not root.exists():
        return mapping

    count = 0
    print(f"[SymlinkHelper] 스캔 시작: {root}")
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts:
            mapping[p.stem] = p
        if max_files is not None and len(mapping) >= max_files:
            print(
                f"[SymlinkHelper] 스캔 조기 종료: {root}, "
                f"매핑 {len(mapping)}개 (max_files={max_files})"
            )
            break

        count += 1
        if count % 10000 == 0:
            print(f"[SymlinkHelper] {root}: {count}개 파일 검사 중...")

    print(f"[SymlinkHelper] 스캔 완료: {root}, 매핑 {len(mapping)}개")
    return mapping


def _create_safe_symlink(
    stems,
    nonmoving_image_map, rider_image_map,
    nonmoving_label_map, rider_label_map,
    split: str
):
    image_root = VIRTUAL_IMAGES_TRAIN if split == "train" else VIRTUAL_IMAGES_VAL
    label_root = VIRTUAL_LABELS_TRAIN if split == "train" else VIRTUAL_LABELS_VAL

    count = 0
    for stem in stems:
        # 이미지 src 결정
        if stem in nonmoving_image_map:
            image_src = nonmoving_image_map[stem]
        elif stem in rider_image_map:
            image_src = rider_image_map[stem]
        else:
            continue  # stem 엉킨 케이스

        image_dst = image_root / f"{stem}{image_src.suffix}"
        image_dst.symlink_to(image_src)

        # 라벨 src 결정
        label_src = None
        if stem in nonmoving_label_map:
            label_src = nonmoving_label_map[stem]
        elif stem in rider_label_map:
            label_src = rider_label_map[stem]

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
        print(f"[SymlinkHelper] 라벨 디렉토리 비어있음: {label_root}")
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
        f"[SymlinkHelper] {image_root.name}: 라벨 기반 페어 {len(image_map)}개 확보 "
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

    # 1) nonmoving / rider 각각 라벨 기반으로 페어 샘플링
    nonmoving_image_map, nonmoving_label_map = _sample_paired_from_labels(
        NONMOVING_IMAGE_DIR, NONMOVING_LABEL_DIR,
        target_pairs=buffered,
        rng=rng,
        max_scan_labels=20000,  # 디버깅용이면 더 줄여도 됨(예: 5000)
    )

    rider_image_map, rider_label_map = _sample_paired_from_labels(
        RIDER_IMAGE_DIR, RIDER_LABEL_DIR,
        target_pairs=buffered,
        rng=rng,
        max_scan_labels=20000,
    )

    # 2) 둘 다 완전히 비었다면 에러
    if not nonmoving_image_map and not rider_image_map:
        raise RuntimeError("nonmoving/rider에서 이미지-라벨 페어를 전혀 찾지 못했습니다.")

    # 3) union stem으로 전체 풀 구성
    all_stems = sorted(set(nonmoving_image_map.keys()) | set(rider_image_map.keys()))
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
        has_rd = stem in rider_image_map
        if has_nm and has_rd:
            group_C.append(stem)
        elif has_nm:
            group_A.append(stem)
        elif has_rd:
            group_B.append(stem)

    rng.shuffle(group_A)
    rng.shuffle(group_B)
    rng.shuffle(group_C)

    if max_train is not None:
        train_stems = balanced_sample(group_A, group_B, group_C, max_train, rng)
    else:
        train_stems = train_stems_raw

    if max_val is not None:
        val_stems = balanced_sample(group_A, group_B, group_C, max_val, rng)
    else:
        val_stems = val_stems_raw

    print(
        f"[SymlinkHelper] split: train {len(train_stems)}개, val {len(val_stems)}개 "
        f"(val_ratio={val_ratio}, max_train={max_train}, max_val={max_val})"
    )

    # 5) 여기서부터는 기존 로직 그대로
    _prepare_virtual_dirs()

    train_count = _create_safe_symlink(
        train_stems,
        nonmoving_image_map, rider_image_map,
        nonmoving_label_map, rider_label_map,
        split="train",
    )
    val_count = _create_safe_symlink(
        val_stems,
        nonmoving_image_map, rider_image_map,
        nonmoving_label_map, rider_label_map,
        split="val",
    )

    print(f"[SymlinkHelper] 링크 생성 결과: train {train_count}개, val {val_count}개")

    # 4) 멀티헤드용 data.yaml 생성
    yaml_content = f"""
path: {VIRTUAL_ROOT}
train: images/train
val: images/val

nc: 54

names:
    0: unused
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

    35: ambulance
    36: bicycle
    37: bus
    38: constructionGuide
    39: motorcycle
    40: otherCar
    41: pedestrian
    42: policeCar
    43: rider
    44: rubberCone
    45: safetyZone
    46: schoolBus
    47: trafficDrum
    48: trafficLight
    49: trafficSign
    50: truck
    51: twoWheeler
    52: vehicle
    53: warningTriangle

# 멀티헤드용 별도 nc
nc_nonmoving: 35
nc_rider: 19

names_nonmoving:
    0: unused_background
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


names_rider:
    0: ambulance
    1: bicycle
    2: bus
    3: constructionGuide
    4: motorcycle
    5: otherCar
    6: pedestrian
    7: policeCar
    8: rider
    9: rubberCone
    10: safetyZone
    11: schoolBus
    12: trafficDrum
    13: trafficLight
    14: trafficSign
    15: truck
    16: twoWheeler
    17: vehicle
    18: warningTriangle

multi_heads:
  nonmoving:
    class_ids: [{", ".join(str(i) for i in range(1, 35))}]
  rider:
    class_ids: [{", ".join(str(i) for i in range(35, 54))}]

channels: 3
"""
    DATASET_YAML.write_text(yaml_content.strip() + "\n", encoding="utf-8")
    print(f"[SymlinkHelper] data.yaml 생성 완료: {DATASET_YAML}")

    return DATASET_YAML
