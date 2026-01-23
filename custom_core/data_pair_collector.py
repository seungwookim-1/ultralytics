from pathlib import Path
import random


def collect_image_label_pairs(
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