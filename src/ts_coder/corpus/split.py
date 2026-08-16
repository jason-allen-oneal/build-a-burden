"""Cluster-stable deterministic splitting."""

import hashlib


def split_for_cluster(
    cluster: str, seed: int = 42, train: float = 0.9, validation: float = 0.05
) -> str:
    value = int(hashlib.sha256(f"{seed}:{cluster}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if value < train else "validation" if value < train + validation else "test"


def assign_splits(
    records: list[dict],
    seed: int = 42,
    train: float = 0.9,
    validation: float = 0.05,
    *,
    group_by_repository: bool = False,
) -> None:
    if group_by_repository:
        _assign_repository_family_splits(records, seed, train, validation)
        return
    clusters = sorted({record["dedup_cluster"] for record in records if record["included"]})
    assignments = {
        cluster: split_for_cluster(cluster, seed, train, validation) for cluster in clusters
    }
    # Tiny controlled corpora can hash entirely into train. Preserve cluster
    # isolation while guaranteeing usable validation/test partitions.
    if len(clusters) >= 2 and "validation" not in assignments.values():
        assignments[clusters[-1]] = "validation"
    if len(clusters) >= 3 and "test" not in assignments.values():
        candidate = clusters[-2] if assignments[clusters[-2]] == "train" else clusters[0]
        assignments[candidate] = "test"
    for record in records:
        record["split"] = assignments[record["dedup_cluster"]] if record["included"] else "excluded"


def _assign_repository_family_splits(
    records: list[dict], seed: int, train: float, validation: float
) -> None:
    repositories = sorted({record["repository_id"] for record in records if record["included"]})
    parent = {repository: repository for repository in repositories}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    cluster_repositories: dict[str, set[str]] = {}
    for record in records:
        if record["included"]:
            cluster_repositories.setdefault(record["dedup_cluster"], set()).add(
                record["repository_id"]
            )
    for members in cluster_repositories.values():
        ordered = sorted(members)
        for repository in ordered[1:]:
            union(ordered[0], repository)

    families: dict[str, list[str]] = {}
    for repository in repositories:
        families.setdefault(find(repository), []).append(repository)
    family_ids = sorted(families)
    family_weights = {family: 0 for family in family_ids}
    for record in records:
        if record["included"]:
            family_weights[find(record["repository_id"])] += 1
    ranked = sorted(family_ids, key=lambda family: (-family_weights[family], family))
    if len(ranked) <= 3:
        assignments = {ranked[0]: "train"} if ranked else {}
        if len(ranked) >= 2:
            assignments[ranked[1]] = "validation"
        if len(ranked) == 3:
            assignments[ranked[2]] = "test"
    else:
        assignments = {
            family: split_for_cluster(f"repository-family:{family}", seed, train, validation)
            for family in family_ids
        }
        if "validation" not in assignments.values():
            assignments[ranked[-1]] = "validation"
        if "test" not in assignments.values():
            candidate = ranked[-2] if assignments[ranked[-2]] == "train" else ranked[-1]
            assignments[candidate] = "test"
    repository_assignments = {
        repository: assignments[find(repository)] for repository in repositories
    }
    for record in records:
        record["split"] = (
            repository_assignments[record["repository_id"]] if record["included"] else "excluded"
        )
