from pathlib import Path, PureWindowsPath
import os
import random
import shutil
from SD_db_Library import ImageLabelDataset

# ===========================
#  Global Config
# ===========================
VIRTUAL_ROOT = Path("/ultralytics/run/sd_nas_area_2")

VIRTUAL_IMAGES_TRAIN = VIRTUAL_ROOT / "images" / "train"
VIRTUAL_IMAGES_VAL   = VIRTUAL_ROOT / "images" / "val"

VIRTUAL_LABELS_TRAIN = VIRTUAL_ROOT / "labels" / "train"
VIRTUAL_LABELS_VAL   = VIRTUAL_ROOT / "labels" / "val"
DATASET_YAML = VIRTUAL_ROOT / "dataset-temp.yaml"

NAS_IMG_ROOT = Path("/mnt/nas/source_images")
NAS_label_ROOT = Path("/mnt/nas/labeling_data")


# 경로 정규화 함수
def _normalize_path(raw_path: str) -> Path:
    """
    DB에서 넘어오는 경로를 실제 NAS 경로(/mnt/nas/...)로 정규화.
    - Windows 경로(X:\source_images\...) → /mnt/nas/source_images/...
    - 이미 /mnt/nas/... 형태면 그대로 사용
    """
    raw_path = (raw_path or "").strip()
    if not raw_path:
        return None

    # 이미 리눅스 경로이면 그대로
    if raw_path.startswith("/mnt/nas"):
        return Path(raw_path)

    # Windows 경로일 가능성 (X:\...)
    if ":" in raw_path and "\\" in raw_path:
        win = PureWindowsPath(raw_path)
        # drive (예: 'X:') 제거 후 나머지 부분만 사용
        parts = list(win.parts[1:])  # ['source_images', 'Date_serial', ...]
        return Path("/mnt/nas").joinpath(*parts)

    # 그 외엔 상대경로라고 보고 NAS 아래로 붙여줌 (필요시 조정)
    return Path("/mnt/nas").joinpath(raw_path.lstrip("\\/"))

# 가상 디렉토리 초기화
def _prepare_virtual_dirs():
    """
    이전에 만든 심볼릭 링크/폴더들 싹 지우고,
    train/val용 디렉터리 재생성.
    """
    if VIRTUAL_ROOT.exists():
        print(f"기존 VIRTUAL_ROOT 제거: {VIRTUAL_ROOT}")
        shutil.rmtree(VIRTUAL_ROOT)

    # train / val 공통 구조
    for split in ("train", "val"):
        (VIRTUAL_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (VIRTUAL_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

def _create_safe_symlink(sample: dict, split: str) -> bool:
    """
    sample: ImageLabelDataset[i]에서 얻은 dict
      - image_path: str (필수)
      - label_paths: list[str] (optional)
    split: 'train' 또는 'val'
    return: 실제로 링크를 만든 경우 True, 스킵하면 False
    """
    raw_img = sample["image_path"]
    real_img = _normalize_path(raw_img)
    if not real_img or not real_img.is_file():
        return False

    raw_label = sample["label_paths"][0] if sample.get("label_paths") else None
    real_label = _normalize_path(raw_label) if raw_label else None
    if not real_label or not real_label.is_file():
        return False   # 현재 구조상 라벨 없는 경우는 dataset에서 이미 필터됨

    # 심볼릭 링크 생성
    target_img = VIRTUAL_ROOT / "images" / split / real_img.name
    target_label = VIRTUAL_ROOT / "labels" / split / (real_img.stem + ".txt")

    target_img.symlink_to(real_img)
    target_label.symlink_to(real_label)

    return True

# Main : Symlink batch 생성
def create_dataset_config(data_conditions=None, val_ratio: float = 0.1, seed: int = 42) -> Path:
    from dotenv import load_dotenv
    load_dotenv('/ultralytics/.env', override=True)
    try:
        from SD_db_Library import SDDatabase
        DB_AVAILABLE = True
        print("SD_db_Library imported successfully")
    except ImportError as e:
        print(f"Warning: SD_db_Library not available - {e}")
        DB_AVAILABLE = False
        return
    if DB_AVAILABLE:
        try:
            db = SDDatabase()
            print("Database connection established for training")
        except Exception as e:
            print(f"Database connection failed: {e}")
            DB_AVAILABLE = False
            return

    dataset = ImageLabelDataset(**(data_conditions or {}))
    total_data_count = len(dataset)

    if total_data_count == 0:
        raise RuntimeError("조건에 맞는 이미지가 없습니다.")

    print(f"DB에서 {total_data_count}개의 row를 로딩했습니다.")

    # symlink 디렉토리 초기화
    _prepare_virtual_dirs()

    indices = list(range(total_data_count))
    random.Random(seed).shuffle(indices)

    n_val = max(1, int(total_data_count * val_ratio))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    print(f"split: train {len(train_indices)}개, val {len(val_indices)}개 (val_ratio={val_ratio})")

    # train split 링크 생성
    train_count = 0
    for idx in train_indices:
        sample = dataset[idx]
        if _create_safe_symlink(sample, split="train"):
            train_count += 1

    # val split 링크 생성
    val_count = 0
    for idx in val_indices:
        sample = dataset[idx]
        if _create_safe_symlink(sample, split="val"):
            val_count += 1

    print(f"링크 생성 결과: train {train_count}개, val {val_count}개")

    # ===========================
    # YAML 생성
    # ===========================
    yaml_content = f"""
path: {VIRTUAL_ROOT}
train: images/train
val: images/val

names:
  0: Myspace
  1: Leftspace
  2: Rightspace
  3: Nodrivingzone
  4: Uturnspace
"""
    DATASET_YAML.write_text(yaml_content.strip(), encoding="utf-8")
    print(f"YAML 생성 완료: {DATASET_YAML}")

    return DATASET_YAML
