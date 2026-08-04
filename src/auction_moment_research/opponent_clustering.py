from __future__ import annotations

"""Research-only unsupervised clustering study for auction opponent behavior.

This program deliberately separates two questions that are often conflated:

* Full five-round trajectory clusters are retrospective descriptions only.  They
  use completed-round labels and must never be treated as live features.
* Two-round prefix clusters use only the first two settled rounds.  Their
  out-of-time test predicts the fifth round of a fully observed game, which is
  a valid *after round two* research question, not a current-round bid model.

Neither clustering pass receives player identity or player event code.  The
clusters therefore describe an observed session pattern, not a persistent
player personality or intent.
"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from .opponent_archetype import (
    ARCHETYPE_LABELS,
    classify_completed_trajectory,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESEARCH_ROOT = REPOSITORY_ROOT / "data" / "v1" / "opponents"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "opponent-clustering-v1"
TRAIN_END_DATE = "2026-07-31"
FULL_ROUND_COUNT = 5
HIGH_RELATIVE_FLOOR = 0.80
LOW_RELATIVE_CEILING = 0.50
CANDIDATE_CLUSTER_COUNTS = tuple(range(2, 7))
MIN_CLUSTER_SHARE = 0.05
MIN_STABILITY_ARI = 0.80
STABILITY_SAMPLE_FRACTION = 0.80
STABILITY_REPLICATES = 30
BOOTSTRAP_REPLICATES = 2_000
KMEANS_N_INIT = 50
KMEANS_MAX_ITER = 500
MODEL_SEED = 20_260_803
STABILITY_SEED = 20_260_804
BOOTSTRAP_SEED = 20_260_805
SMOOTHING_STRENGTH = 10.0

FULL_FEATURE_NAMES = (
    "round1_relative",
    "round5_relative",
    "mean_relative",
    "std_relative",
    "round5_minus_round1",
    "range_relative",
    "zero_share",
)
PREFIX_FEATURE_NAMES = (
    "round1_relative",
    "round2_relative",
)


class StudyError(RuntimeError):
    """Raised when the frozen research contract is not met."""


@dataclass(frozen=True)
class StudyConfig:
    research_root: Path
    output_root: Path
    train_end_date: str = TRAIN_END_DATE
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES
    stability_replicates: int = STABILITY_REPLICATES
    overwrite: bool = False


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: object) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    _write_text_atomic(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise StudyError("无法对空样本计算分位数")
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def _date(group: Sequence[dict]) -> str:
    """Use the first observed round as the session's temporal split boundary.

    A real auction can legitimately cross midnight.  In that case all its
    rounds remain in the train or holdout side determined at session start;
    rejecting it or splitting a single player-session would be worse.
    """
    date = str(group[0].get("observed_at") or "")[:10]
    if not date:
        raise StudyError("玩家会话首轮缺少观察日期")
    return date


def _relative(row: dict) -> float:
    value = float(row["label_relative_to_round_max"])
    if not 0.0 <= value <= 1.0:
        raise StudyError(f"相对报价不在 [0, 1] 范围内：{value}")
    return value


def _is_zero(row: dict) -> int:
    return int(int(row["label_bid"]) == 0)


def _first_round_state(group: Sequence[dict]) -> str:
    if _is_zero(group[0]):
        return "zero"
    relative = _relative(group[0])
    if relative < LOW_RELATIVE_CEILING:
        return "low"
    if relative < HIGH_RELATIVE_FLOOR:
        return "middle"
    return "high"


def _input_data_quality(rows: Sequence[dict]) -> dict:
    """Validate the frozen source before any grouping or model fitting."""
    keys = [
        (str(row.get("player_key")), str(row.get("session_id")), int(row.get("round")))
        for row in rows
    ]
    key_counts = Counter(keys)
    duplicate_key_rows = sum(count - 1 for count in key_counts.values() if count > 1)
    invalid_relative_rows = sum(
        not 0.0 <= float(row.get("label_relative_to_round_max")) <= 1.0
        for row in rows
    )
    invalid_bid_state_rows = sum(
        str(row.get("label_bid_state"))
        != ("explicit_zero" if int(row.get("label_bid")) == 0 else "positive")
        for row in rows
    )
    non_strict_source_rows = sum(
        row.get("source") != "trusted_protocol_confirmed" for row in rows
    )
    missing_observed_at_rows = sum(not str(row.get("observed_at") or "") for row in rows)
    if duplicate_key_rows:
        raise StudyError(f"严格玩家轮次键重复：{duplicate_key_rows} 行")
    if invalid_relative_rows:
        raise StudyError(f"相对报价越界：{invalid_relative_rows} 行")
    if invalid_bid_state_rows:
        raise StudyError(f"协议报价状态与金额不一致：{invalid_bid_state_rows} 行")
    if non_strict_source_rows:
        raise StudyError(f"存在非严格可信来源：{non_strict_source_rows} 行")
    if missing_observed_at_rows:
        raise StudyError(f"缺少观察时间：{missing_observed_at_rows} 行")
    return {
        "strict_player_round_rows": len(rows),
        "unique_player_session_round_keys": len(key_counts),
        "duplicate_player_session_round_rows": duplicate_key_rows,
        "non_strict_source_rows": non_strict_source_rows,
        "relative_bid_out_of_range_rows": invalid_relative_rows,
        "bid_state_amount_mismatch_rows": invalid_bid_state_rows,
        "missing_observed_at_rows": missing_observed_at_rows,
        "explicit_zero_rows": sum(_is_zero(row) for row in rows),
        "positive_bid_with_zero_relative_rows": sum(
            int(row["label_bid"]) > 0 and _relative(row) == 0.0 for row in rows
        ),
        "zero_semantics": (
            "zero_share uses protocol label_bid == 0; a relative ratio of zero is not "
            "treated as an explicit zero bid"
        ),
        "observation_dates": sorted({str(row["observed_at"])[:10] for row in rows}),
    }


def _build_groups(rows: Sequence[dict]) -> tuple[list[list[dict]], dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("source") != "trusted_protocol_confirmed":
            raise StudyError("存在非严格可信玩家-轮次行")
        grouped[(str(row["player_key"]), str(row["session_id"]))].append(row)

    groups: list[list[dict]] = []
    for group in grouped.values():
        group.sort(key=lambda row: int(row["round"]))
        rounds = [int(row["round"]) for row in group]
        if rounds != list(range(1, len(rounds) + 1)):
            raise StudyError("发现非连续、重复或非首轮开始的玩家会话轨迹")
        _date(group)
        groups.append(group)

    counts = Counter(len(group) for group in groups)
    cross_midnight = sum(
        len({str(row.get("observed_at") or "")[:10] for row in group}) > 1
        for group in groups
    )
    return groups, {
        "player_session_trajectories": len(groups),
        "round_count_distribution": {str(key): value for key, value in sorted(counts.items())},
        "all_groups_start_at_round_one": True,
        "all_groups_have_contiguous_rounds": True,
        "groups_crossing_calendar_midnight_assigned_by_first_round": cross_midnight,
        "fully_observed_five_round_trajectories": counts[FULL_ROUND_COUNT],
    }


def _require_five_rounds(group: Sequence[dict]) -> None:
    if len(group) != FULL_ROUND_COUNT:
        raise StudyError("此聚类协议只接受完整的五轮玩家会话")
    if [int(row["round"]) for row in group] != [1, 2, 3, 4, 5]:
        raise StudyError("完整五轮轨迹的轮次序列错误")


def full_trajectory_features(group: Sequence[dict]) -> list[float]:
    """Return threshold-free, retrospective five-round behavior features."""
    _require_five_rounds(group)
    values = np.asarray([_relative(row) for row in group], dtype=float)
    return [
        float(values[0]),
        float(values[-1]),
        float(np.mean(values)),
        float(np.std(values)),
        float(values[-1] - values[0]),
        float(np.max(values) - np.min(values)),
        float(np.mean([_is_zero(row) for row in group])),
    ]


def prefix_features(group: Sequence[dict]) -> list[float]:
    """Return the only two behavior values visible after round two settles."""
    _require_five_rounds(group)
    return [_relative(group[0]), _relative(group[1])]


def _fit_kmeans(features: np.ndarray, cluster_count: int, seed: int) -> KMeans:
    return KMeans(
        n_clusters=cluster_count,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
        random_state=seed + cluster_count,
    ).fit(features)


def _candidate_cluster_counts(
    scaled_features: np.ndarray,
    *,
    stability_replicates: int,
) -> tuple[list[dict], int]:
    """Choose K before seeing holdout outcomes, using separation and stability."""
    if len(scaled_features) <= max(CANDIDATE_CLUSTER_COUNTS):
        raise StudyError("训练样本不足以比较预注册的聚类数量")
    candidates: list[dict] = []
    for cluster_count in CANDIDATE_CLUSTER_COUNTS:
        model = _fit_kmeans(scaled_features, cluster_count, MODEL_SEED)
        labels = model.labels_
        shares = np.bincount(labels, minlength=cluster_count) / len(labels)
        try:
            silhouette = float(silhouette_score(scaled_features, labels))
        except ValueError as exc:
            raise StudyError(f"K={cluster_count} 的 silhouette 无法计算：{exc}") from exc

        rng = np.random.default_rng(STABILITY_SEED + cluster_count)
        subset_size = max(cluster_count * 2, round(len(scaled_features) * STABILITY_SAMPLE_FRACTION))
        stability_scores: list[float] = []
        for replicate in range(stability_replicates):
            indexes = rng.choice(len(scaled_features), size=subset_size, replace=False)
            replica = KMeans(
                n_clusters=cluster_count,
                n_init=max(20, KMEANS_N_INIT // 2),
                max_iter=KMEANS_MAX_ITER,
                random_state=STABILITY_SEED + cluster_count * 100 + replicate,
            ).fit(scaled_features[indexes])
            stability_scores.append(
                float(adjusted_rand_score(labels, replica.predict(scaled_features)))
            )

        row = {
            "cluster_count": cluster_count,
            "silhouette": round(silhouette, 8),
            "minimum_train_cluster_share": round(float(np.min(shares)), 8),
            "train_cluster_shares_sorted": [round(float(value), 8) for value in sorted(shares)],
            "subsample_stability_ari_mean": round(_mean(stability_scores), 8),
            "subsample_stability_ari_p05": round(_quantile(stability_scores, 0.05), 8),
        }
        row["eligible"] = bool(
            row["minimum_train_cluster_share"] >= MIN_CLUSTER_SHARE
            and row["subsample_stability_ari_mean"] >= MIN_STABILITY_ARI
        )
        candidates.append(row)

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise StudyError("没有簇数同时通过最小簇占比和重采样稳定性门")
    selected = max(
        eligible,
        key=lambda candidate: (candidate["silhouette"], -candidate["cluster_count"]),
    )
    selected_count = int(selected["cluster_count"])
    for candidate in candidates:
        candidate["selected"] = candidate["cluster_count"] == selected_count
    return candidates, selected_count


def _canonical_labels(
    labels: np.ndarray,
    centers_raw: np.ndarray,
    *,
    feature_names: Sequence[str],
    mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Give KMeans labels a deterministic order without adding semantic names."""
    indexes = {name: position for position, name in enumerate(feature_names)}
    if mode == "full":
        order = sorted(
            range(len(centers_raw)),
            key=lambda index: (
                float(centers_raw[index, indexes["mean_relative"]]),
                float(centers_raw[index, indexes["round5_minus_round1"]]),
                float(centers_raw[index, indexes["round1_relative"]]),
                index,
            ),
        )
    elif mode == "prefix":
        order = sorted(
            range(len(centers_raw)),
            key=lambda index: (
                float(np.mean(centers_raw[index])),
                float(centers_raw[index, 1] - centers_raw[index, 0]),
                index,
            ),
        )
    else:
        raise StudyError(f"未知聚类模式：{mode}")
    old_to_new = {old: new for new, old in enumerate(order)}
    canonical = np.asarray([old_to_new[int(label)] for label in labels], dtype=int)
    return canonical, centers_raw[order], old_to_new


def _cluster_id(label: int) -> str:
    return f"cluster_{label + 1:02d}"


def _independent_holdout_center_distance(
    scaled_holdout: np.ndarray,
    train_centers_scaled: np.ndarray,
    cluster_count: int,
) -> float:
    """Refit only the holdout features and compare its best centre matching."""
    independent = _fit_kmeans(scaled_holdout, cluster_count, MODEL_SEED + 10_000)
    holdout_centers = independent.cluster_centers_
    best = math.inf
    for order in permutations(range(cluster_count)):
        squared = [
            float(np.sum((train_centers_scaled[index] - holdout_centers[order[index]]) ** 2))
            for index in range(cluster_count)
        ]
        best = min(best, math.sqrt(_mean(squared)))
    return float(best)


def _full_cluster_result(
    train_groups: Sequence[Sequence[dict]],
    holdout_groups: Sequence[Sequence[dict]],
    config: StudyConfig,
) -> tuple[dict, list[dict], list[dict], np.ndarray]:
    train_features = np.asarray([full_trajectory_features(group) for group in train_groups])
    holdout_features = np.asarray([full_trajectory_features(group) for group in holdout_groups])
    scaler = StandardScaler().fit(train_features)
    scaled_train = scaler.transform(train_features)
    scaled_holdout = scaler.transform(holdout_features)
    candidates, cluster_count = _candidate_cluster_counts(
        scaled_train, stability_replicates=config.stability_replicates
    )
    model = _fit_kmeans(scaled_train, cluster_count, MODEL_SEED)
    centers_raw = scaler.inverse_transform(model.cluster_centers_)
    train_labels, canonical_centers, mapping = _canonical_labels(
        model.labels_, centers_raw, feature_names=FULL_FEATURE_NAMES, mode="full"
    )
    holdout_raw_labels = model.predict(scaled_holdout)
    holdout_labels = np.asarray([mapping[int(label)] for label in holdout_raw_labels], dtype=int)
    train_shares = np.bincount(train_labels, minlength=cluster_count) / len(train_labels)
    holdout_shares = np.bincount(holdout_labels, minlength=cluster_count) / len(holdout_labels)

    combined_groups = list(train_groups) + list(holdout_groups)
    combined_labels = np.concatenate((train_labels, holdout_labels))
    archetypes = [classify_completed_trajectory(list(group)) for group in combined_groups]
    profiles: list[dict] = []
    crosstab: list[dict] = []
    for label in range(cluster_count):
        indexes = [index for index, value in enumerate(combined_labels) if value == label]
        values = np.asarray([full_trajectory_features(combined_groups[index]) for index in indexes])
        rule_counts = Counter(archetypes[index] for index in indexes)
        profiles.append({
            "cluster_id": _cluster_id(label),
            "trajectories": len(indexes),
            "all_data_share": round(len(indexes) / len(combined_groups), 8),
            **{
                f"mean_{name}": round(float(values[:, feature_index].mean()), 8)
                for feature_index, name in enumerate(FULL_FEATURE_NAMES)
            },
        })
        for archetype in sorted(ARCHETYPE_LABELS):
            count = rule_counts[archetype]
            crosstab.append({
                "cluster_id": _cluster_id(label),
                "rule_archetype": archetype,
                "rule_label": ARCHETYPE_LABELS[archetype],
                "trajectories": count,
                "within_cluster_share": round(count / len(indexes), 8),
            })

    result = {
        "scope": "retrospective_completed_five_round_trajectory_only",
        "feature_contract": {
            "features": list(FULL_FEATURE_NAMES),
            "player_identity_used": False,
            "player_event_code_used": False,
            "uses_post_round_outcomes": True,
            "live_feature_allowed": False,
        },
        "model_selection": {
            "candidate_cluster_counts": list(CANDIDATE_CLUSTER_COUNTS),
            "minimum_cluster_share": MIN_CLUSTER_SHARE,
            "minimum_subsample_stability_ari": MIN_STABILITY_ARI,
            "selection_rule": "highest silhouette among candidates passing both predeclared gates; lower K breaks ties",
            "candidates": candidates,
            "selected_cluster_count": cluster_count,
        },
        "cluster_centers": [
            {
                "cluster_id": _cluster_id(label),
                **{
                    name: round(float(canonical_centers[label, feature_index]), 8)
                    for feature_index, name in enumerate(FULL_FEATURE_NAMES)
                },
            }
            for label in range(cluster_count)
        ],
        "cross_time_structure": {
            "train_trajectories": len(train_groups),
            "holdout_trajectories": len(holdout_groups),
            "train_cluster_shares": {
                _cluster_id(label): round(float(train_shares[label]), 8)
                for label in range(cluster_count)
            },
            "holdout_assigned_cluster_shares": {
                _cluster_id(label): round(float(holdout_shares[label]), 8)
                for label in range(cluster_count)
            },
            "train_holdout_share_total_variation": round(
                float(0.5 * np.sum(np.abs(train_shares - holdout_shares))), 8
            ),
            "independent_holdout_center_rms_distance_standardized": round(
                _independent_holdout_center_distance(
                    scaled_holdout, model.cluster_centers_, cluster_count
                ),
                8,
            ),
        },
        "rule_archetype_adjusted_mutual_information": round(
            float(adjusted_mutual_info_score(combined_labels, archetypes)), 8
        ),
        "interpretation": (
            "The clusters are a descriptive segmentation of completed sessions. "
            "They cannot identify a player's intent or a stable player type."
        ),
    }
    return result, profiles, crosstab, combined_labels


def _repeat_cluster_stability(
    groups: Sequence[Sequence[dict]], labels: np.ndarray
) -> dict:
    by_player: dict[str, list[tuple[Sequence[dict], int]]] = defaultdict(list)
    for group, label in zip(groups, labels):
        by_player[str(group[0]["player_key"])].append((group, int(label)))
    pairs: list[tuple[int, int]] = []
    for player_groups in by_player.values():
        player_groups.sort(key=lambda item: (_date(item[0]), str(item[0][0]["session_id"])))
        pairs.extend(
            (earlier[1], later[1]) for earlier, later in zip(player_groups, player_groups[1:])
        )
    if not pairs:
        return {
            "players_with_two_or_more_five_round_sessions": 0,
            "consecutive_session_pairs": 0,
            "same_cluster_pair_rate": None,
            "independent_marginal_expected_same_rate": None,
            "excess_over_independent_marginal": None,
            "interpretation": "no repeated five-round player sessions",
        }
    previous = Counter(pair[0] for pair in pairs)
    later = Counter(pair[1] for pair in pairs)
    expected = sum(
        (previous[label] / len(pairs)) * (later[label] / len(pairs))
        for label in set(previous) | set(later)
    )
    observed = _mean([float(earlier == later) for earlier, later in pairs])
    return {
        "players_with_two_or_more_five_round_sessions": sum(
            len(value) >= 2 for value in by_player.values()
        ),
        "consecutive_session_pairs": len(pairs),
        "same_cluster_pair_rate": round(observed, 8),
        "independent_marginal_expected_same_rate": round(expected, 8),
        "excess_over_independent_marginal": round(observed - expected, 8),
        "interpretation": (
            "descriptive retention only; sparse repeated-player coverage cannot support "
            "a persistent personality claim"
        ),
    }


def _brier(targets: Sequence[float], predictions: Sequence[float]) -> float:
    return _mean([(target - prediction) ** 2 for target, prediction in zip(targets, predictions)])


def _category_probabilities(
    categories: Sequence[int | str],
    targets: Sequence[float],
) -> tuple[float, dict[int | str, float], Counter]:
    global_rate = _mean(targets)
    by_category: dict[int | str, list[float]] = defaultdict(list)
    for category, target in zip(categories, targets):
        by_category[category].append(float(target))
    probabilities = {
        category: (
            sum(values) + SMOOTHING_STRENGTH * global_rate
        ) / (len(values) + SMOOTHING_STRENGTH)
        for category, values in by_category.items()
    }
    return global_rate, probabilities, Counter(categories)


def _bootstrap_prefix_scores(scored: Sequence[dict], replicates: int) -> dict:
    by_session: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        by_session[str(row["session_id"])].append(row)
    sessions = sorted(by_session)
    if not sessions:
        raise StudyError("留出集评分为空")
    rng = random.Random(BOOTSTRAP_SEED)
    baseline_minus_cluster: list[float] = []
    first_state_minus_cluster: list[float] = []
    for _ in range(replicates):
        sample = [
            row
            for session in [rng.choice(sessions) for _ in sessions]
            for row in by_session[session]
        ]
        target = [float(row["target_round5_high"]) for row in sample]
        baseline = _brier(target, [float(row["prediction_baseline"]) for row in sample])
        first_state = _brier(
            target, [float(row["prediction_first_round_state"]) for row in sample]
        )
        cluster = _brier(
            target, [float(row["prediction_prefix_cluster"]) for row in sample]
        )
        baseline_minus_cluster.append(baseline - cluster)
        first_state_minus_cluster.append(first_state - cluster)

    def interval(values: Sequence[float]) -> dict:
        return {
            "ci95_low": round(_quantile(values, 0.025), 8),
            "ci95_high": round(_quantile(values, 0.975), 8),
            "probability_positive": round(_mean([float(value > 0) for value in values]), 8),
        }

    return {
        "cluster": "session_id",
        "replicates": replicates,
        "baseline_minus_prefix_cluster": interval(baseline_minus_cluster),
        "first_round_state_minus_prefix_cluster": interval(first_state_minus_cluster),
    }


def _prefix_cluster_result(
    train_groups: Sequence[Sequence[dict]],
    holdout_groups: Sequence[Sequence[dict]],
    config: StudyConfig,
) -> tuple[dict, list[dict], list[dict]]:
    train_features = np.asarray([prefix_features(group) for group in train_groups])
    holdout_features = np.asarray([prefix_features(group) for group in holdout_groups])
    scaler = StandardScaler().fit(train_features)
    scaled_train = scaler.transform(train_features)
    scaled_holdout = scaler.transform(holdout_features)
    candidates, cluster_count = _candidate_cluster_counts(
        scaled_train, stability_replicates=config.stability_replicates
    )
    model = _fit_kmeans(scaled_train, cluster_count, MODEL_SEED)
    centers_raw = scaler.inverse_transform(model.cluster_centers_)
    train_labels, canonical_centers, mapping = _canonical_labels(
        model.labels_, centers_raw, feature_names=PREFIX_FEATURE_NAMES, mode="prefix"
    )
    holdout_raw_labels = model.predict(scaled_holdout)
    holdout_labels = np.asarray([mapping[int(label)] for label in holdout_raw_labels], dtype=int)

    train_high = [float(_relative(group[-1]) >= HIGH_RELATIVE_FLOOR) for group in train_groups]
    train_zero = [float(_is_zero(group[-1])) for group in train_groups]
    holdout_high = [float(_relative(group[-1]) >= HIGH_RELATIVE_FLOOR) for group in holdout_groups]
    holdout_zero = [float(_is_zero(group[-1])) for group in holdout_groups]
    baseline_high, cluster_high_probability, _ = _category_probabilities(train_labels, train_high)
    baseline_zero, cluster_zero_probability, _ = _category_probabilities(train_labels, train_zero)
    train_states = [_first_round_state(group) for group in train_groups]
    _, first_state_probability, _ = _category_probabilities(train_states, train_high)

    scored: list[dict] = []
    for group, label, target_high, target_zero in zip(
        holdout_groups, holdout_labels, holdout_high, holdout_zero
    ):
        state = _first_round_state(group)
        scored.append({
            "session_id": str(group[0]["session_id"]),
            "date": _date(group),
            "prefix_cluster_id": _cluster_id(int(label)),
            "first_round_state": state,
            "target_round5_high": int(target_high),
            "target_round5_zero": int(target_zero),
            "prediction_baseline": round(baseline_high, 12),
            "prediction_first_round_state": round(first_state_probability[state], 12),
            "prediction_prefix_cluster": round(cluster_high_probability[int(label)], 12),
        })
    holdout_targets = [float(row["target_round5_high"]) for row in scored]
    metrics = {
        "baseline_brier": _brier(
            holdout_targets, [float(row["prediction_baseline"]) for row in scored]
        ),
        "first_round_state_brier": _brier(
            holdout_targets,
            [float(row["prediction_first_round_state"]) for row in scored],
        ),
        "prefix_cluster_brier": _brier(
            holdout_targets,
            [float(row["prediction_prefix_cluster"]) for row in scored],
        ),
    }
    metrics["baseline_minus_prefix_cluster"] = (
        metrics["baseline_brier"] - metrics["prefix_cluster_brier"]
    )
    metrics["first_round_state_minus_prefix_cluster"] = (
        metrics["first_round_state_brier"] - metrics["prefix_cluster_brier"]
    )

    calibration: list[dict] = []
    for label in range(cluster_count):
        train_indexes = [index for index, value in enumerate(train_labels) if value == label]
        holdout_indexes = [index for index, value in enumerate(holdout_labels) if value == label]
        calibration.append({
            "prefix_cluster_id": _cluster_id(label),
            "center_round1_relative": round(float(canonical_centers[label, 0]), 8),
            "center_round2_relative": round(float(canonical_centers[label, 1]), 8),
            "center_round2_minus_round1": round(
                float(canonical_centers[label, 1] - canonical_centers[label, 0]), 8
            ),
            "train_trajectories": len(train_indexes),
            "train_round5_high_rate": round(
                _mean([train_high[index] for index in train_indexes]), 8
            ),
            "train_round5_zero_rate": round(
                _mean([train_zero[index] for index in train_indexes]), 8
            ),
            "trained_round5_high_probability": round(cluster_high_probability[label], 8),
            "trained_round5_zero_probability": round(cluster_zero_probability[label], 8),
            "holdout_trajectories": len(holdout_indexes),
            "holdout_round5_high_rate": round(
                _mean([holdout_high[index] for index in holdout_indexes]), 8
            ),
            "holdout_round5_zero_rate": round(
                _mean([holdout_zero[index] for index in holdout_indexes]), 8
            ),
        })

    result = {
        "scope": "after_two_settled_rounds_predict_round5_of_a_fully_observed_game",
        "feature_contract": {
            "features": list(PREFIX_FEATURE_NAMES),
            "feature_time": "after round 2 ranking is settled",
            "player_identity_used": False,
            "player_event_code_used": False,
            "future_rounds_used_as_features": False,
            "current_round_bid_prediction": False,
        },
        "model_selection": {
            "candidate_cluster_counts": list(CANDIDATE_CLUSTER_COUNTS),
            "minimum_cluster_share": MIN_CLUSTER_SHARE,
            "minimum_subsample_stability_ari": MIN_STABILITY_ARI,
            "selection_rule": "highest silhouette among candidates passing both predeclared gates; lower K breaks ties",
            "candidates": candidates,
            "selected_cluster_count": cluster_count,
        },
        "training": {
            "trajectories": len(train_groups),
            "round5_high_rate": round(baseline_high, 8),
            "round5_zero_rate": round(baseline_zero, 8),
        },
        "holdout": {
            "trajectories": len(holdout_groups),
            "sessions": len({str(group[0]["session_id"]) for group in holdout_groups}),
            "round5_high_rate": round(_mean(holdout_high), 8),
            "round5_zero_rate": round(_mean(holdout_zero), 8),
        },
        "metrics": {name: round(value, 8) for name, value in metrics.items()},
        "bootstrap": _bootstrap_prefix_scores(scored, config.bootstrap_replicates),
        "benchmark_note": (
            "first_round_state is a thresholded, first-round-only benchmark; it is not "
            "an input to the unsupervised prefix clustering"
        ),
    }
    return result, calibration, scored


def _render_markdown(summary: dict) -> str:
    full = summary["full_trajectory_clustering"]
    prefix = summary["two_round_prefix_clustering"]
    repeat = summary["repeat_cluster_stability"]
    full_selected = next(
        row for row in full["model_selection"]["candidates"] if row["selected"]
    )
    prefix_selected = next(
        row for row in prefix["model_selection"]["candidates"] if row["selected"]
    )
    calibration = summary["prefix_cluster_calibration"]
    source_quality = summary["quality"]["input"]
    lines = [
        "# 对手行为无监督聚类研究 v1",
        "",
        "## 结论",
        "",
        "- **完整五轮轨迹按本次运行预先固定的选择规则只支持 "
        f"{full['model_selection']['selected_cluster_count']} 个稳健簇。** "
        f"其 train silhouette 为 {full_selected['silhouette']:.3f}，80% 子样本 ARI 为 "
        f"{full_selected['subsample_stability_ari_mean']:.3f}。",
        "- **这不是稳定的玩家人格分类。** "
        f"仅 {repeat['players_with_two_or_more_five_round_sessions']} 名玩家贡献了 "
        f"{repeat['consecutive_session_pairs']} 个相邻完整会话对；同簇率 "
        f"{repeat['same_cluster_pair_rate']:.1%}，独立边际期望 "
        f"{repeat['independent_marginal_expected_same_rate']:.1%}。",
        "- **“晚轮抬价”没有被自动发现为通过选择门的独立簇。** "
        "更高 K 可以切出更细形状，但未赢过本次固定的分离度/最小簇门；因此不能把它提升为已证实的独立类型。",
        "- **第二轮结束后，前缀簇对第 5 轮高相对出价有时间外信息，但只是会话状态。** "
        f"留出 Brier 从 {prefix['metrics']['baseline_brier']:.6f} 降至 "
        f"{prefix['metrics']['prefix_cluster_brier']:.6f}（改善 "
        f"{prefix['metrics']['baseline_minus_prefix_cluster']:.6f}）。",
        "- **它尚未证明优于简单的一轮状态桶。** "
        f"相对一轮状态桶的 Brier 额外改善仅 "
        f"{prefix['metrics']['first_round_state_minus_prefix_cluster']:.6f}，"
        f"簇级 bootstrap 95% CI "
        f"[{prefix['bootstrap']['first_round_state_minus_prefix_cluster']['ci95_low']:.6f}, "
        f"{prefix['bootstrap']['first_round_state_minus_prefix_cluster']['ci95_high']:.6f}]，"
        "跨过 0。",
        "",
        "## 聚类输入与时间边界",
        "",
        "- 数据：仅协议确认、完整观察的五轮对手轨迹；训练截止 2026-07-31，留出为 2026-08-01 至 2026-08-02。",
        "- 全轨迹簇使用报价的首/末轮、均值、波动、变化、范围和协议明确零报价占比；这些全是**赛后描述**，不可进入实时模型。",
        "- 前缀簇只使用已结算的第 1、2 轮相对报价；不使用玩家 ID、事件码、未来轮或同轮未知结果。",
        f"- 输入质量：{source_quality['strict_player_round_rows']} 条严格行、复合键重复 0、相对报价越界 0；"
        f"有 {source_quality['positive_bid_with_zero_relative_rows']} 条正数报价的相对比为 0，"
        "所以零报价特征严格取协议金额 `bid=0`，不按相对比补零。",
        "",
        "## 全轨迹簇（按平均相对报价从低到高编号）",
        "",
    ]
    for profile in summary["full_cluster_profiles"]:
        lines.append(
            f"- `{profile['cluster_id']}`：{profile['trajectories']} 条，"
            f"均值 {profile['mean_mean_relative']:.3f}，"
            f"首轮 {profile['mean_round1_relative']:.3f}，"
            f"第 5 轮 {profile['mean_round5_relative']:.3f}，"
            f"第 5 轮减首轮 {profile['mean_round5_minus_round1']:+.3f}，"
            f"零报价占比 {profile['mean_zero_share']:.1%}。"
        )
    lines.extend([
        "",
        "## 第二轮后的前缀簇与第 5 轮（时间外）",
        "",
    ])
    for row in calibration:
        lines.append(
            f"- `{row['prefix_cluster_id']}`：第 1/2 轮中心 "
            f"{row['center_round1_relative']:.3f}/{row['center_round2_relative']:.3f}；"
            f"训练第 5 轮高相对概率 {row['trained_round5_high_probability']:.1%}，"
            f"留出实测 {row['holdout_round5_high_rate']:.1%}（n={row['holdout_trajectories']}）。"
        )
    lines.extend([
        "",
        "## 使用边界",
        "",
        "- 簇是**本局已观察行为的压缩描述**，不是激进、保守或埋伏的永久人格判定。",
        "- 这份结果是冻结历史样本的研究证据，未接入建议、自动出价或当前轮建模。",
        "- K 候选与选择门在本次最终运行前固定，但本研究不是外部事前注册实验；需要新采集的未触碰时间块复核。",
        "- 若未来要尝试使用前缀簇，必须先登记新的时间边界，在新样本上再次做两段时间外验证。",
        "",
    ])
    return "\n".join(lines)


def run_study(config: StudyConfig) -> dict:
    if config.output_root.exists() and any(config.output_root.iterdir()) and not config.overwrite:
        raise StudyError(f"输出目录非空：{config.output_root}")
    input_manifest_path = config.research_root / "manifest.json"
    rows_path = config.research_root / "player_rounds.jsonl"
    if not input_manifest_path.is_file() or not rows_path.is_file():
        raise StudyError("缺少严格玩家行为研究输入")
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if not input_manifest.get("research_only") or input_manifest.get("production_enabled"):
        raise StudyError("拒绝非 research-only 或已启用生产的输入")

    rows = _read_jsonl(rows_path)
    input_quality = _input_data_quality(rows)
    groups, trajectory_quality = _build_groups(rows)
    five_round_groups = [group for group in groups if len(group) == FULL_ROUND_COUNT]
    train_groups = [group for group in five_round_groups if _date(group) <= config.train_end_date]
    holdout_groups = [group for group in five_round_groups if _date(group) > config.train_end_date]
    if not train_groups or not holdout_groups:
        raise StudyError("完整五轮样本没有同时覆盖训练和留出时间段")
    if len(train_groups) <= max(CANDIDATE_CLUSTER_COUNTS) or len(holdout_groups) <= max(CANDIDATE_CLUSTER_COUNTS):
        raise StudyError("完整五轮训练或留出样本不足")

    full_result, full_profiles, crosstab, all_full_labels = _full_cluster_result(
        train_groups, holdout_groups, config
    )
    prefix_result, prefix_calibration, scored = _prefix_cluster_result(
        train_groups, holdout_groups, config
    )
    all_groups = list(train_groups) + list(holdout_groups)
    repeat = _repeat_cluster_stability(all_groups, all_full_labels)
    summary = {
        "schema_version": "opponent-clustering-study-v1",
        "research_only": True,
        "production_enabled": False,
        "advisory_enabled": False,
        "auto_bid_enabled": False,
        "methodology": {
            "algorithm": "StandardScaler plus KMeans",
            "sklearn_version": sklearn.__version__,
            "study_design": (
                "exploratory frozen-sample study; the final-run K selection rule was fixed "
                "before artifact generation but was not externally preregistered"
            ),
            "cluster_count_selection": (
                "train-only silhouette, subject to minimum 5% cluster share and "
                "mean 80%-subsample ARI >= 0.80"
            ),
            "fully_observed_game_requirement": "exactly rounds 1 through 5",
            "train_end_date": config.train_end_date,
            "holdout_dates": sorted({_date(group) for group in holdout_groups}),
            "no_player_identity_feature": True,
            "no_player_event_code_feature": True,
        },
        "quality": {
            "input": input_quality,
            "trajectory": trajectory_quality,
        },
        "cohort": {
            "five_round_trajectories": len(five_round_groups),
            "train_five_round_trajectories": len(train_groups),
            "holdout_five_round_trajectories": len(holdout_groups),
            "holdout_sessions": len({str(group[0]["session_id"]) for group in holdout_groups}),
        },
        "full_trajectory_clustering": full_result,
        "full_cluster_profiles": full_profiles,
        "two_round_prefix_clustering": prefix_result,
        "prefix_cluster_calibration": prefix_calibration,
        "repeat_cluster_stability": repeat,
        "critical_boundary": (
            "full-trajectory clusters are retrospective labels; prefix clusters are "
            "research-only after-two-round states and not current-round bid features"
        ),
    }

    config.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_root / "study_summary.json"
    quality_path = config.output_root / "data_quality.json"
    full_candidates_path = config.output_root / "full_trajectory_k_candidates.jsonl"
    full_profiles_path = config.output_root / "full_trajectory_profiles.jsonl"
    crosstab_path = config.output_root / "full_cluster_rule_crosstab.jsonl"
    prefix_candidates_path = config.output_root / "prefix_k_candidates.jsonl"
    calibration_path = config.output_root / "prefix_cluster_calibration.jsonl"
    scores_path = config.output_root / "holdout_round5_scores.jsonl"
    report_path = config.output_root / "STUDY.md"
    _write_json(summary_path, summary)
    _write_json(quality_path, summary["quality"])
    _write_jsonl(
        full_candidates_path, full_result["model_selection"]["candidates"]
    )
    _write_jsonl(full_profiles_path, full_profiles)
    _write_jsonl(crosstab_path, crosstab)
    _write_jsonl(
        prefix_candidates_path, prefix_result["model_selection"]["candidates"]
    )
    _write_jsonl(calibration_path, prefix_calibration)
    _write_jsonl(scores_path, scored)
    _write_text_atomic(report_path, _render_markdown(summary))

    outputs = [
        summary_path,
        quality_path,
        full_candidates_path,
        full_profiles_path,
        crosstab_path,
        prefix_candidates_path,
        calibration_path,
        scores_path,
        report_path,
    ]
    manifest = {
        "schema_version": "opponent-clustering-study-v1",
        "research_only": True,
        "production_enabled": False,
        "advisory_enabled": False,
        "auto_bid_enabled": False,
        "input": {
            "research_manifest_sha256": _sha256(input_manifest_path),
            "player_rounds_sha256": _sha256(rows_path),
        },
        "config": {
            "train_end_date": config.train_end_date,
            "full_round_count": FULL_ROUND_COUNT,
            "candidate_cluster_counts": list(CANDIDATE_CLUSTER_COUNTS),
            "minimum_cluster_share": MIN_CLUSTER_SHARE,
            "minimum_stability_ari": MIN_STABILITY_ARI,
            "stability_replicates": config.stability_replicates,
            "bootstrap_replicates": config.bootstrap_replicates,
            "model_seed": MODEL_SEED,
            "stability_seed": STABILITY_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "summary": summary,
        "outputs": [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
    }
    _write_json(config.output_root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="对手行为无监督聚类研究（研究专用）")
    parser.add_argument("--research-root", default=str(DEFAULT_RESEARCH_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--train-end-date", default=TRAIN_END_DATE)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--stability-replicates", type=int, default=STABILITY_REPLICATES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        manifest = run_study(StudyConfig(
            research_root=Path(args.research_root).resolve(),
            output_root=Path(args.output_root).resolve(),
            train_end_date=args.train_end_date,
            bootstrap_replicates=args.bootstrap_replicates,
            stability_replicates=args.stability_replicates,
            overwrite=args.overwrite,
        ))
    except StudyError as exc:
        print(f"研究未生成：{exc}")
        return 2
    summary = manifest["summary"]
    print(json.dumps({
        "status": "study_generated",
        "output_root": str(args.output_root),
        "production_enabled": manifest["production_enabled"],
        "full_trajectory_cluster_count": summary["full_trajectory_clustering"]["model_selection"]["selected_cluster_count"],
        "prefix_cluster_count": summary["two_round_prefix_clustering"]["model_selection"]["selected_cluster_count"],
        "holdout_five_round_trajectories": summary["cohort"]["holdout_five_round_trajectories"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
