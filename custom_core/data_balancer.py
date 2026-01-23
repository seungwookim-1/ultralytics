from pathlib import Path
import random
from .config_loader import get_symlink_config


def select_balanced_stems(
    all_stems: list[str],
    domain_image_maps: dict[str, dict[str, Path]],
    target: int,
    rng: random.Random,
) -> list[str]:
    """
    도메인 개수에 상관없이, 각 도메인에 비슷한 수의 샘플이 배분되도록 stem을 뽑는다.
    - target: 최종적으로 원하는 stem 개수
    - domain_image_maps: domain_name -> {stem -> image_path}
    """
    if target is None or target >= len(all_stems):
        # 전체 쓰는 게 목표보다 적으면 그냥 다 쓰기
        return list(all_stems)

    sym_cfg = get_symlink_config()
    domain_names = [cfg.name for cfg in sym_cfg.domain_configs]

    # stem 별로 어떤 도메인에 속하는지 membership 계산
    membership: dict[str, list[str]] = {}
    for stem in all_stems:
        doms = [d for d in domain_names if stem in domain_image_maps.get(d, {})]
        membership[stem] = doms

    # 단일 도메인 전용 stem 풀(single) + 다중 도메인 공유 stem 풀(multi)
    single_pools: dict[str, list[str]] = {d: [] for d in domain_names}
    multi_pool: list[str] = []

    for stem, doms in membership.items():
        if len(doms) == 1:
            single_pools[doms[0]].append(stem)
        elif len(doms) > 1:
            multi_pool.append(stem)
        # 0개인 stem은 애초에 all_stems에 안 들어왔다고 가정

    for d in domain_names:
        rng.shuffle(single_pools[d])
    rng.shuffle(multi_pool)

    selected: list[str] = []
    selected_set: set[str] = set()
    domain_counts: dict[str, int] = {d: 0 for d in domain_names}

    # 도메인별로 맞추고 싶은 대략 목표치
    target_per_domain = max(1, target // len(domain_names))

    # 1단계: 각 도메인별 single stem으로 1차 채우기
    for d in domain_names:
        pool = single_pools[d]
        i = 0
        while i < len(pool) and domain_counts[d] < target_per_domain and len(selected) < target:
            stem = pool[i]
            if stem not in selected_set:
                selected.append(stem)
                selected_set.add(stem)
                for d2 in membership[stem]:
                    domain_counts[d2] += 1
            i += 1

    # 2단계: multi_pool + 남은 single들을 섞어서 남은 자리를 채우기
    remaining_needed = target - len(selected)
    if remaining_needed > 0:
        remaining_pool: list[str] = []
        remaining_pool.extend(multi_pool)
        for d in domain_names:
            remaining_pool.extend([s for s in single_pools[d] if s not in selected_set])

        rng.shuffle(remaining_pool)

        for stem in remaining_pool:
            if len(selected) >= target:
                break
            if stem in selected_set:
                continue
            selected.append(stem)
            selected_set.add(stem)
            for d2 in membership[stem]:
                domain_counts[d2] += 1

    return selected