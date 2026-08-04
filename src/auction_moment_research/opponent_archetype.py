from __future__ import annotations

"""Research-only session-trajectory archetype study for auction opponents.

The labels below are descriptions of a completed player-session trajectory, not
claims about intent or persistent personality.  The only forward test uses the
first settled round to score a later round; it never uses the completed
trajectory label as a contemporaneous feature.
"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESEARCH_ROOT = REPOSITORY_ROOT / "data" / "v1" / "opponents"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "opponent-archetype-v1"
TRAIN_END_DATE = "2026-07-31"
SMOOTHING_STRENGTH = 10
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20_260_804
LOW_RELATIVE_CEILING = 0.50
HIGH_RELATIVE_FLOOR = 0.80
LATE_ESCALATION_MIN_DELTA = 0.40


class StudyError(RuntimeError):
    pass


@dataclass(frozen=True)
class StudyConfig:
    research_root: Path
    output_root: Path
    train_end_date: str = TRAIN_END_DATE
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES
    overwrite: bool = False


ARCHETYPE_LABELS = {
    "persistent_aggressive": "持续高相对出价",
    "persistent_conservative": "持续低相对或零出价",
    "late_escalator": "晚轮抬价轨迹",
    "mixed_or_contextual": "混合或情境型",
}


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
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _date(row: dict) -> str:
    return str(row.get("observed_at") or "")[:10]


def _relative(row: dict) -> float:
    return float(row["label_relative_to_round_max"])


def _first_round_state(row: dict) -> str:
    if int(row["label_bid"]) == 0:
        return "zero"
    relative = _relative(row)
    if relative < LOW_RELATIVE_CEILING:
        return "low"
    if relative < HIGH_RELATIVE_FLOOR:
        return "middle"
    return "high"


def classify_completed_trajectory(rows: list[dict]) -> str:
    """Classify a completed session trajectory; never use this live."""
    if len(rows) < 2:
        raise StudyError("原型需要至少两轮已结算行为")
    relative = [_relative(row) for row in rows]
    zero_rate = _mean([int(int(row["label_bid"]) == 0) for row in rows])
    high_share = _mean([int(value >= HIGH_RELATIVE_FLOOR) for value in relative])
    low_share = _mean([int(value <= LOW_RELATIVE_CEILING) for value in relative])
    if (
        relative[0] <= LOW_RELATIVE_CEILING
        and relative[-1] >= HIGH_RELATIVE_FLOOR
        and relative[-1] - relative[0] >= LATE_ESCALATION_MIN_DELTA
    ):
        return "late_escalator"
    if zero_rate == 0 and high_share >= 2 / 3:
        return "persistent_aggressive"
    if zero_rate >= 1 / 3 or low_share >= 2 / 3:
        return "persistent_conservative"
    return "mixed_or_contextual"


def _build_groups(rows: list[dict]) -> tuple[list[list[dict]], dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["player_key"]), str(row["session_id"]))].append(row)
    groups = []
    invalid_round_order = 0
    for group in grouped.values():
        group.sort(key=lambda row: int(row["round"]))
        rounds = [int(row["round"]) for row in group]
        if rounds != list(range(1, len(rounds) + 1)):
            invalid_round_order += 1
            continue
        groups.append(group)
    if invalid_round_order:
        raise StudyError(f"发现 {invalid_round_order} 个非连续或非首轮开始的玩家会话轨迹")
    return groups, {
        "all_player_session_groups": len(groups),
        "groups_with_one_round_excluded_from_trajectory_labels": sum(
            len(group) == 1 for group in groups
        ),
        "groups_with_two_or_more_rounds": sum(len(group) >= 2 for group in groups),
        "round_count_distribution": {
            str(length): count
            for length, count in sorted(Counter(len(group) for group in groups).items())
        },
        "all_groups_start_at_round_one": True,
        "all_groups_have_contiguous_rounds": True,
    }


def _archetype_summary(groups: list[list[dict]]) -> tuple[list[dict], list[dict]]:
    eligible = [group for group in groups if len(group) >= 2]
    by_type: dict[str, list[list[dict]]] = defaultdict(list)
    daily: dict[str, Counter[str]] = defaultdict(Counter)
    for group in eligible:
        archetype = classify_completed_trajectory(group)
        by_type[archetype].append(group)
        daily[_date(group[0])][archetype] += 1
    total = len(eligible)
    summary = []
    for archetype in sorted(ARCHETYPE_LABELS):
        members = by_type[archetype]
        if not members:
            continue
        summary.append({
            "archetype": archetype,
            "label": ARCHETYPE_LABELS[archetype],
            "player_session_trajectories": len(members),
            "share": round(len(members) / total, 6),
            "mean_rounds": round(_mean([len(group) for group in members]), 6),
            "mean_first_relative_bid": round(_mean([_relative(group[0]) for group in members]), 6),
            "mean_final_relative_bid": round(_mean([_relative(group[-1]) for group in members]), 6),
            "final_high_relative_rate": round(_mean([
                int(_relative(group[-1]) >= HIGH_RELATIVE_FLOOR)
                for group in members
            ]), 6),
            "final_zero_rate": round(_mean([
                int(int(group[-1]["label_bid"]) == 0)
                for group in members
            ]), 6),
        })
    daily_rows = []
    for date, counts in sorted(daily.items()):
        denominator = sum(counts.values())
        for archetype in sorted(ARCHETYPE_LABELS):
            daily_rows.append({
                "date": date,
                "archetype": archetype,
                "label": ARCHETYPE_LABELS[archetype],
                "trajectories": counts[archetype],
                "share": round(counts[archetype] / denominator, 6),
                "daily_total": denominator,
            })
    return summary, daily_rows


def _metrics(scored: list[dict]) -> dict[str, float]:
    result = {}
    for model in ("baseline", "first_round_state"):
        result[f"{model}_brier"] = _mean([
            (float(row["target_final_high"]) - float(row[f"prediction_{model}"])) ** 2
            for row in scored
        ])
    return result


def _forward_validation(groups: list[list[dict]], config: StudyConfig) -> tuple[dict, list[dict], list[dict]]:
    eligible = [group for group in groups if len(group) >= 2]
    train = [group for group in eligible if _date(group[0]) <= config.train_end_date]
    holdout = [group for group in eligible if _date(group[0]) > config.train_end_date]
    if not train or not holdout:
        raise StudyError("会话内前瞻验证没有足够的训练或留出轨迹")
    baseline = _mean([
        int(_relative(group[-1]) >= HIGH_RELATIVE_FLOOR)
        for group in train
    ])
    by_state: dict[str, list[list[dict]]] = defaultdict(list)
    for group in train:
        by_state[_first_round_state(group[0])].append(group)
    state_probability = {
        state: (
            sum(int(_relative(group[-1]) >= HIGH_RELATIVE_FLOOR) for group in state_groups)
            + SMOOTHING_STRENGTH * baseline
        ) / (len(state_groups) + SMOOTHING_STRENGTH)
        for state, state_groups in by_state.items()
    }
    scored = []
    for group in holdout:
        state = _first_round_state(group[0])
        scored.append({
            "session_id": str(group[0]["session_id"]),
            "date": _date(group[0]),
            "first_round_state": state,
            "round_count": len(group),
            "target_final_high": int(_relative(group[-1]) >= HIGH_RELATIVE_FLOOR),
            "target_final_zero": int(int(group[-1]["label_bid"]) == 0),
            "prediction_baseline": round(baseline, 12),
            "prediction_first_round_state": round(state_probability[state], 12),
        })
    metrics = _metrics(scored)
    by_session: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        by_session[row["session_id"]].append(row)
    sessions = sorted(by_session)
    rng = random.Random(BOOTSTRAP_SEED)
    differences = []
    for _ in range(config.bootstrap_replicates):
        sample = [
            row for session in [rng.choice(sessions) for _ in sessions]
            for row in by_session[session]
        ]
        score = _metrics(sample)
        differences.append(score["baseline_brier"] - score["first_round_state_brier"])
    transitions = []
    for state in ("zero", "low", "middle", "high"):
        state_rows = [row for row in scored if row["first_round_state"] == state]
        if state_rows:
            transitions.append({
                "first_round_state": state,
                "holdout_trajectories": len(state_rows),
                "final_high_rate": round(_mean([float(row["target_final_high"]) for row in state_rows]), 6),
                "final_zero_rate": round(_mean([float(row["target_final_zero"]) for row in state_rows]), 6),
                "trained_probability": round(state_probability[state], 6),
            })
    validation = {
        "target": "final_round_relative_bid_at_least_0_80",
        "feature": "first_settled_round_state_only",
        "train_end_date": config.train_end_date,
        "train_trajectories": len(train),
        "holdout_trajectories": len(holdout),
        "holdout_sessions": len(sessions),
        "baseline_final_high_rate_train": round(baseline, 8),
        "baseline_final_high_rate_holdout": round(_mean([float(row["target_final_high"]) for row in scored]), 8),
        "metrics": {key: round(value, 8) for key, value in metrics.items()},
        "brier_improvement_positive_is_better": round(
            metrics["baseline_brier"] - metrics["first_round_state_brier"], 8
        ),
        "bootstrap": {
            "cluster": "session_id",
            "replicates": config.bootstrap_replicates,
            "ci95_low": round(_quantile(differences, 0.025), 8),
            "ci95_high": round(_quantile(differences, 0.975), 8),
            "probability_positive": round(sum(value > 0 for value in differences) / len(differences), 6),
        },
    }
    return validation, scored, transitions


def _repeat_stability(groups: list[list[dict]]) -> dict:
    by_player: dict[str, list[list[dict]]] = defaultdict(list)
    for group in groups:
        if len(group) >= 2:
            by_player[str(group[0]["player_key"])].append(group)
    pairs = []
    for player_groups in by_player.values():
        player_groups.sort(key=lambda group: (group[0]["observed_at"], group[0]["session_id"]))
        for earlier, later in zip(player_groups, player_groups[1:]):
            pairs.append((
                classify_completed_trajectory(earlier),
                classify_completed_trajectory(later),
            ))
    previous = Counter(pair[0] for pair in pairs)
    later = Counter(pair[1] for pair in pairs)
    expected_same = sum(
        (previous[name] / len(pairs)) * (later[name] / len(pairs))
        for name in ARCHETYPE_LABELS
    ) if pairs else 0.0
    return {
        "players_with_two_or_more_multi_round_sessions": sum(
            len(player_groups) >= 2 for player_groups in by_player.values()
        ),
        "consecutive_session_pairs": len(pairs),
        "same_archetype_pair_rate": round(_mean([
            int(earlier == later) for earlier, later in pairs
        ]), 6) if pairs else None,
        "independent_marginal_expected_same_rate": round(expected_same, 6),
        "interpretation": (
            "descriptive retention only; repeated-player coverage is too small "
            "to claim persistent personal types"
        ),
    }


def _render_markdown(summary: dict) -> str:
    archetypes = {row["archetype"]: row for row in summary["archetypes"]}
    forward = summary["forward_validation"]
    stability = summary["repeat_stability"]
    return "\n".join([
        "# 对手会话内行为原型研究",
        "",
        "## 结论",
        "",
        "- **可以把对手划分为会话内行为原型，但不能把它称为稳定玩家人格。** "
        "这些标签描述一局结束后的轨迹，用于研究与回放，不是当前局的直接特征。",
        "- **用户所说的“埋伏型”存在为晚轮抬价轨迹，但比例不高且事后才能确认。** "
        f"它占 {archetypes['late_escalator']['share']:.1%} 的多轮轨迹；不能在首轮时断言某人就是埋伏型。",
        "- **第一轮结算后的状态对后续轮有强信号。** 仅用首轮状态预测末轮是否高相对出价，"
        f"留出 Brier 改善 {forward['brier_improvement_positive_is_better']:.6f}，"
        f"95% CI [{forward['bootstrap']['ci95_low']:.6f}, {forward['bootstrap']['ci95_high']:.6f}]。"
        "这说明应建模“本局已观察行为状态”，而不是给玩家贴永久标签。",
        "",
        "## 原型定义（全是赛后描述）",
        "",
        "- 持续高相对出价：无零报价，且至少三分之二轮次达到当轮最高报价的 80%。",
        "- 持续低相对或零出价：至少三分之一轮为零，或至少三分之二轮低于当轮最高报价的 50%。",
        "- 晚轮抬价轨迹（“埋伏候选”）：首轮低于 50%，末轮至少 80%，且上升至少 40 个百分点。",
        "- 其余为混合或情境型。",
        "",
        "## 重复玩家不足以证明稳定人格",
        "",
        f"- 有两局以上多轮轨迹的玩家仅 {stability['players_with_two_or_more_multi_round_sessions']} 个，"
        f"连续会话对仅 {stability['consecutive_session_pairs']} 对。",
        f"- 连续两局同原型比例 {stability['same_archetype_pair_rate']:.1%}，"
        f"独立边际期望为 {stability['independent_marginal_expected_same_rate']:.1%}；"
        "差距不足以支撑稳定类型声明。",
        "",
        "## 可接受的下一步",
        "",
        "1. 在第二轮开始后，仅将首轮已结算状态作为 shadow 特征，预测后续轮的对手高出价概率。",
        "2. 不把“埋伏型”当作开局判断；它只可作为复盘标签，直到获得更长的跨会话稳定性证据。",
        "3. 继续隔离所有赛后排名、未来轮报价和类型标签，不让它们进入同轮出价。",
        "",
    ])


def run_study(config: StudyConfig) -> dict:
    if config.output_root.exists() and any(config.output_root.iterdir()) and not config.overwrite:
        raise StudyError(f"输出目录非空：{config.output_root}")
    input_manifest = config.research_root / "manifest.json"
    rows_path = config.research_root / "player_rounds.jsonl"
    if not input_manifest.is_file() or not rows_path.is_file():
        raise StudyError("缺少严格玩家行为研究输入")
    source_manifest = json.loads(input_manifest.read_text(encoding="utf-8"))
    if not source_manifest.get("research_only") or source_manifest.get("production_enabled"):
        raise StudyError("拒绝非 research-only 输入")
    rows = _read_jsonl(rows_path)
    if any(row.get("source") != "trusted_protocol_confirmed" for row in rows):
        raise StudyError("存在非严格可信玩家-轮次行")
    groups, quality = _build_groups(rows)
    archetypes, daily = _archetype_summary(groups)
    forward, scored, transitions = _forward_validation(groups, config)
    stability = _repeat_stability(groups)
    summary = {
        "schema_version": "opponent-archetype-study-v1",
        "research_only": True,
        "production_enabled": False,
        "advisory_enabled": False,
        "auto_bid_enabled": False,
        "definitions": {
            "trajectory_grain": "player_key, session_id; at least two completed rounds",
            "relative_bid": "post_round_bid divided by post_round_max_bid",
            "low_relative_ceiling": LOW_RELATIVE_CEILING,
            "high_relative_floor": HIGH_RELATIVE_FLOOR,
            "late_escalation_min_delta": LATE_ESCALATION_MIN_DELTA,
            "critical_boundary": "completed trajectory labels are retrospective and not live features",
        },
        "quality": quality,
        "archetypes": archetypes,
        "daily_archetype_mix": daily,
        "forward_validation": forward,
        "first_round_to_final_transitions": transitions,
        "repeat_stability": stability,
    }
    config.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_root / "study_summary.json"
    archetype_path = config.output_root / "archetype_summary.jsonl"
    daily_path = config.output_root / "daily_archetype_mix.jsonl"
    transition_path = config.output_root / "first_round_transitions.jsonl"
    scores_path = config.output_root / "holdout_first_round_scores.jsonl"
    report_path = config.output_root / "STUDY.md"
    _write_json(summary_path, summary)
    _write_jsonl(archetype_path, archetypes)
    _write_jsonl(daily_path, daily)
    _write_jsonl(transition_path, transitions)
    _write_jsonl(scores_path, scored)
    _write_text_atomic(report_path, _render_markdown(summary))
    outputs = [summary_path, archetype_path, daily_path, transition_path, scores_path, report_path]
    manifest = {
        "schema_version": "opponent-archetype-study-v1",
        "research_only": True,
        "production_enabled": False,
        "advisory_enabled": False,
        "auto_bid_enabled": False,
        "input": {
            "research_manifest_sha256": _sha256(input_manifest),
            "player_rounds_sha256": _sha256(rows_path),
        },
        "config": {
            "train_end_date": config.train_end_date,
            "smoothing_strength": SMOOTHING_STRENGTH,
            "bootstrap_replicates": config.bootstrap_replicates,
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
    parser = argparse.ArgumentParser(description="对手会话内行为原型研究（研究专用）")
    parser.add_argument("--research-root", default=str(DEFAULT_RESEARCH_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--train-end-date", default=TRAIN_END_DATE)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        manifest = run_study(StudyConfig(
            research_root=Path(args.research_root).resolve(),
            output_root=Path(args.output_root).resolve(),
            train_end_date=args.train_end_date,
            bootstrap_replicates=args.bootstrap_replicates,
            overwrite=args.overwrite,
        ))
    except StudyError as exc:
        print(f"研究未生成：{exc}")
        return 2
    print(json.dumps({
        "status": "study_generated",
        "output_root": str(args.output_root),
        "trajectory_groups": manifest["summary"]["quality"]["groups_with_two_or_more_rounds"],
        "production_enabled": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
