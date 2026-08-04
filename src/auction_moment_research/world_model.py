from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


WORLD_MODEL_SCHEMA_VERSION = "world-model-v1"
EVIDENCE_BUNDLE_SCHEMA_VERSION = "evidence-bundle-v1"
QUALITY_ORDER = ("白", "蓝", "紫", "金", "彩")
NON_COLORED_QUALITIES = QUALITY_ORDER[:-1]


class WorldModelError(RuntimeError):
    pass


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def catalog_semantic_hash(catalog: Sequence[object]) -> str:
    rows = [
        {
            "catalog_id": str(getattr(item, "catalog_id")),
            "name": str(getattr(item, "name")),
            "quality": str(getattr(item, "quality")),
            "width": int(getattr(item, "width")),
            "height": int(getattr(item, "height")),
            "value": int(getattr(item, "value")),
        }
        for item in catalog
    ]
    rows.sort(key=lambda row: row["catalog_id"])
    return _sha256_json(rows)


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    maximum = max(values)
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(
        sum(math.exp(value - maximum) for value in values)
    )


def _normalize_positive(values: Sequence[float]) -> tuple[float, ...]:
    cleaned = [max(0.0, float(value)) for value in values]
    total = sum(cleaned)
    if total <= 0:
        return tuple(1.0 / len(cleaned) for _ in cleaned)
    return tuple(value / total for value in cleaned)


def _sample_categorical(
    rng: random.Random,
    values: Sequence[int],
    probabilities: Sequence[float],
) -> int:
    return int(rng.choices(values, weights=probabilities, k=1)[0])


def _dirichlet_draw(
    rng: random.Random,
    alpha: Sequence[float],
) -> tuple[float, ...]:
    draws = [
        rng.gammavariate(max(1e-9, float(value)), 1.0)
        for value in alpha
    ]
    return _normalize_positive(draws)


def _multinomial_draw(
    rng: random.Random,
    total: int,
    probabilities: Sequence[float],
) -> tuple[int, ...]:
    counts = [0] * len(probabilities)
    if total <= 0:
        return tuple(counts)
    population = list(range(len(probabilities)))
    for selected in rng.choices(
        population,
        weights=probabilities,
        k=int(total),
    ):
        counts[int(selected)] += 1
    return tuple(counts)


def _weighted_without_replacement_draw(
    rng: random.Random,
    total: int,
    probabilities: Sequence[float],
) -> tuple[int, ...]:
    counts = [0] * len(probabilities)
    available = list(range(len(probabilities)))
    for _ in range(min(max(0, int(total)), len(available))):
        weights = [probabilities[index] for index in available]
        selected = int(rng.choices(available, weights=weights, k=1)[0])
        counts[selected] += 1
        available.remove(selected)
    remaining = max(0, int(total) - sum(counts))
    if remaining:
        extra = _multinomial_draw(rng, remaining, probabilities)
        counts = [
            count + additional
            for count, additional in zip(counts, extra)
        ]
    return tuple(counts)


def _count_regime_bounds(max_count: int) -> tuple[tuple[int, int], ...]:
    candidates = ((1, 19), (20, 34), (35, 49), (50, int(max_count)))
    return tuple(
        (lower, min(upper, int(max_count)))
        for lower, upper in candidates
        if lower <= int(max_count)
    )


def dirichlet_multinomial_log_probability(
    counts: Sequence[int],
    alpha: Sequence[float],
) -> float:
    if len(counts) != len(alpha):
        raise WorldModelError("Dirichlet-multinomial 维度不一致")
    if any(int(value) < 0 for value in counts):
        return float("-inf")
    total = sum(int(value) for value in counts)
    alpha_total = sum(float(value) for value in alpha)
    if alpha_total <= 0 or any(float(value) <= 0 for value in alpha):
        raise WorldModelError("Dirichlet 参数必须为正")
    result = math.lgamma(total + 1)
    result += math.lgamma(alpha_total) - math.lgamma(
        alpha_total + total
    )
    for count, parameter in zip(counts, alpha):
        result -= math.lgamma(int(count) + 1)
        result += math.lgamma(float(parameter) + int(count))
        result -= math.lgamma(float(parameter))
    return result


def _smoothed_alpha(
    counts: Sequence[int],
    *,
    concentration: float,
    base_mass: float,
) -> tuple[float, ...]:
    if not counts:
        raise WorldModelError("不能拟合空分类分布")
    probabilities = _normalize_positive(
        [float(value) + float(base_mass) for value in counts]
    )
    return tuple(
        float(base_mass) + float(concentration) * probability
        for probability in probabilities
    )


@dataclass(frozen=True)
class WorldLogProbability:
    total: float
    components: Mapping[str, float]


class ProbabilisticWorldModel:
    """Hierarchical joint prior over a complete treasure multiset.

    The model deliberately keeps the parameterization small.  With only tens
    of independent maps, a high-capacity learner would memorize sessions.
    Counts are generated hierarchically:

    total count -> colored hurdle/count -> non-colored qualities
    -> size within quality -> identity within quality and size.

    Every catalog identity receives positive base mass, including identities
    not yet observed in reviewed maps.
    """

    def __init__(
        self,
        artifact: Mapping,
        catalog: Sequence[object],
    ) -> None:
        self.artifact = json.loads(_canonical_json(dict(artifact)))
        if (
            self.artifact.get("schema_version")
            != WORLD_MODEL_SCHEMA_VERSION
        ):
            raise WorldModelError("world model schema 不兼容")
        expected_hash = catalog_semantic_hash(catalog)
        if self.artifact.get("catalog_semantic_sha256") != expected_hash:
            raise WorldModelError("world model 图鉴语义哈希不匹配")
        self.catalog = tuple(catalog)
        self.catalog_hash = expected_hash
        self.max_count = int(self.artifact["support"]["max_count"])
        self.count_probabilities = tuple(
            float(value)
            for value in self.artifact["total_count_probabilities"]
        )
        if len(self.count_probabilities) != self.max_count + 1:
            raise WorldModelError("总件数概率表长度无效")
        self.colored_zero_probability = float(
            self.artifact["colored_hurdle"]["zero_probability"]
        )
        self.colored_positive_probabilities = tuple(
            float(value)
            for value in self.artifact["colored_hurdle"][
                "positive_count_probabilities"
            ]
        )
        self.non_colored_quality_categories = tuple(
            str(value)
            for value in self.artifact.get(
                "non_colored_quality_categories",
                NON_COLORED_QUALITIES,
            )
        )
        self.non_colored_quality_alpha = tuple(
            float(value)
            for value in self.artifact["non_colored_quality_alpha"]
        )
        self.quality_size_categories = {
            str(quality): tuple(str(value) for value in categories)
            for quality, categories in self.artifact[
                "quality_size_categories"
            ].items()
        }
        self.quality_size_alpha = {
            str(quality): tuple(float(value) for value in alpha)
            for quality, alpha in self.artifact[
                "quality_size_alpha"
            ].items()
        }
        self.identity_categories = {
            str(group): tuple(str(value) for value in categories)
            for group, categories in self.artifact[
                "identity_categories"
            ].items()
        }
        self.identity_alpha = {
            str(group): tuple(float(value) for value in alpha)
            for group, alpha in self.artifact["identity_alpha"].items()
        }
        self.count_regimes = tuple(
            {
                **dict(regime),
                "non_colored_quality_alpha": tuple(
                    float(value)
                    for value in regime[
                        "non_colored_quality_alpha"
                    ]
                ),
                "quality_size_alpha": {
                    str(quality): tuple(
                        float(value) for value in alpha
                    )
                    for quality, alpha in regime[
                        "quality_size_alpha"
                    ].items()
                },
                "colored_hurdle": {
                    "zero_probability": float(
                        regime["colored_hurdle"][
                            "zero_probability"
                        ]
                    ),
                    "positive_count_probabilities": tuple(
                        float(value)
                        for value in regime["colored_hurdle"][
                            "positive_count_probabilities"
                        ]
                    ),
                },
            }
            for regime in self.artifact.get("count_regimes") or []
        )
        self.colored_duplicate_extra_probability = float(
            self.artifact.get(
                "colored_identity_process", {}
            ).get("duplicate_extra_probability", 0.01)
        )
        self.row_probabilities = {
            int(row): float(probability)
            for row, probability in self.artifact[
                "row_probabilities"
            ].items()
        }
        self._index_by_id = {
            str(getattr(item, "catalog_id")): index
            for index, item in enumerate(self.catalog)
        }
        self._indices_by_quality_size: dict[
            tuple[str, str], tuple[int, ...]
        ] = {}
        for quality, sizes in self.quality_size_categories.items():
            for size in sizes:
                self._indices_by_quality_size[(quality, size)] = tuple(
                    index
                    for index, item in enumerate(self.catalog)
                    if str(getattr(item, "quality")) == quality
                    and f"{int(getattr(item, 'width'))}x"
                    f"{int(getattr(item, 'height'))}" == size
                )

    def _regime_for_count(self, total_count: int) -> Mapping | None:
        for regime in self.count_regimes:
            if (
                int(regime["minimum_count"])
                <= int(total_count)
                <= int(regime["maximum_count"])
            ):
                return regime
        return None

    @classmethod
    def fit(
        cls,
        catalog: Sequence[object],
        games: Sequence[object],
        *,
        auction_tier: str,
        max_count: int = 60,
        reviewed_prior_revision: str | None = None,
        training_fingerprint: str | None = None,
        quality_concentration: float = 18.0,
        size_concentration: float = 12.0,
        identity_concentration: float = 8.0,
        category_base_mass: float = 0.25,
        identity_base_mass: float = 0.15,
    ) -> "ProbabilisticWorldModel":
        if not catalog or not games:
            raise WorldModelError("联合先验需要非空图鉴和历史对局")
        max_count = max(
            int(max_count),
            max(int(getattr(game, "total_count")) for game in games),
        )
        catalog_rows = tuple(catalog)
        total_count_histogram = [0.05] * (max_count + 1)
        total_count_histogram[0] = 0.0
        row_histogram: Counter[int] = Counter()
        colored_zero = 0
        observed_colored_counts: Counter[int] = Counter()
        colored_duplicate_extras = 0
        colored_identity_exposures = 0
        non_colored_totals = Counter()
        quality_size_totals: defaultdict[
            str, Counter[str]
        ] = defaultdict(Counter)
        identity_totals: defaultdict[
            str, Counter[str]
        ] = defaultdict(Counter)
        training_rows: list[dict] = []

        for game in games:
            counts = tuple(
                int(value) for value in getattr(game, "item_counts")
            )
            total_count = sum(counts)
            if total_count <= 0 or total_count > max_count:
                continue
            total_count_histogram[total_count] += 1.0
            row_histogram[int(getattr(game, "rows"))] += 1
            colored_count = 0
            game_quality_counts: Counter[str] = Counter()
            game_quality_sizes: defaultdict[
                str, Counter[str]
            ] = defaultdict(Counter)
            for item, count in zip(catalog_rows, counts):
                quality = str(getattr(item, "quality"))
                size = (
                    f"{int(getattr(item, 'width'))}x"
                    f"{int(getattr(item, 'height'))}"
                )
                catalog_id = str(getattr(item, "catalog_id"))
                if quality == "彩":
                    colored_count += count
                    colored_identity_exposures += count
                    colored_duplicate_extras += max(0, count - 1)
                else:
                    non_colored_totals[quality] += count
                game_quality_counts[quality] += count
                game_quality_sizes[quality][size] += count
                quality_size_totals[quality][size] += count
                identity_totals[f"{quality}|{size}"][catalog_id] += count
            if colored_count == 0:
                colored_zero += 1
            else:
                observed_colored_counts[colored_count] += 1
            training_rows.append(
                {
                    "total_count": total_count,
                    "colored_count": colored_count,
                    "quality_counts": dict(game_quality_counts),
                    "quality_size_counts": {
                        quality: dict(values)
                        for quality, values
                        in game_quality_sizes.items()
                    },
                }
            )

        count_probabilities = _normalize_positive(total_count_histogram)
        maximum_observed_colored = max(
            observed_colored_counts, default=0
        )
        colored_positive_histogram = [0.0] * (max_count + 1)
        for count in range(1, max_count + 1):
            tail_distance = max(
                0, count - maximum_observed_colored
            )
            colored_positive_histogram[count] = (
                0.02 * (0.20**tail_distance)
            )
        for count, frequency in observed_colored_counts.items():
            colored_positive_histogram[count] += float(frequency)
        colored_positive_probabilities = _normalize_positive(
            colored_positive_histogram
        )
        present_qualities = tuple(
            quality
            for quality in QUALITY_ORDER
            if any(
                str(getattr(item, "quality")) == quality
                for item in catalog_rows
            )
        )
        colored_present = "彩" in present_qualities
        colored_zero_probability = (
            (colored_zero + 0.5) / (len(games) + 1.0)
            if colored_present
            else 1.0
        )
        non_colored_quality_categories = tuple(
            quality
            for quality in NON_COLORED_QUALITIES
            if quality in present_qualities
        )
        non_colored_quality_alpha = (
            _smoothed_alpha(
                [
                    non_colored_totals[quality]
                    for quality in non_colored_quality_categories
                ],
                concentration=quality_concentration,
                base_mass=category_base_mass,
            )
            if non_colored_quality_categories
            else ()
        )

        quality_size_categories: dict[str, list[str]] = {}
        quality_size_alpha: dict[str, list[float]] = {}
        identity_categories: dict[str, list[str]] = {}
        identity_alpha: dict[str, list[float]] = {}
        for quality in present_qualities:
            sizes = sorted(
                {
                    (
                        f"{int(getattr(item, 'width'))}x"
                        f"{int(getattr(item, 'height'))}"
                    )
                    for item in catalog_rows
                    if str(getattr(item, "quality")) == quality
                }
            )
            quality_size_categories[quality] = sizes
            quality_size_alpha[quality] = list(
                _smoothed_alpha(
                    [
                        quality_size_totals[quality][size]
                        for size in sizes
                    ],
                    concentration=size_concentration,
                    base_mass=category_base_mass,
                )
            )
            for size in sizes:
                group = f"{quality}|{size}"
                identities = sorted(
                    str(getattr(item, "catalog_id"))
                    for item in catalog_rows
                    if str(getattr(item, "quality")) == quality
                    and (
                        f"{int(getattr(item, 'width'))}x"
                        f"{int(getattr(item, 'height'))}"
                    )
                    == size
                )
                identity_categories[group] = identities
                identity_alpha[group] = list(
                    _smoothed_alpha(
                        [
                            identity_totals[group][catalog_id]
                            for catalog_id in identities
                        ],
                        concentration=identity_concentration,
                        base_mass=identity_base_mass,
                    )
                )

        count_regimes = []
        global_non_colored_total = sum(
            non_colored_totals.values()
        )
        for lower, upper in _count_regime_bounds(max_count):
            regime_rows = [
                row
                for row in training_rows
                if lower <= int(row["total_count"]) <= upper
            ]
            regime_quality_totals: Counter[str] = Counter()
            regime_quality_sizes: defaultdict[
                str, Counter[str]
            ] = defaultdict(Counter)
            regime_colored_counts: Counter[int] = Counter()
            regime_colored_zero = 0
            for row in regime_rows:
                regime_quality_totals.update(
                    {
                        quality: int(count)
                        for quality, count in row[
                            "quality_counts"
                        ].items()
                        if quality != "彩"
                    }
                )
                for quality, sizes in row[
                    "quality_size_counts"
                ].items():
                    regime_quality_sizes[quality].update(
                        {
                            size: int(count)
                            for size, count in sizes.items()
                        }
                    )
                colored_count = int(row["colored_count"])
                if colored_count == 0:
                    regime_colored_zero += 1
                else:
                    regime_colored_counts[colored_count] += 1

            global_colored_zero_probability = (
                colored_zero_probability
            )
            regime_zero_probability = (
                regime_colored_zero
                + 0.5
                + 3.0 * global_colored_zero_probability
            ) / (len(regime_rows) + 4.0)
            regime_positive_weights = [
                0.0
            ] * (max_count + 1)
            for count in range(1, max_count + 1):
                regime_positive_weights[count] = (
                    0.5 * colored_positive_probabilities[count]
                    + float(regime_colored_counts[count])
                )

            shrunken_quality_counts = []
            for quality in non_colored_quality_categories:
                global_probability = (
                    non_colored_totals[quality]
                    / global_non_colored_total
                    if global_non_colored_total
                    else 1.0
                    / max(
                        1,
                        len(non_colored_quality_categories),
                    )
                )
                shrunken_quality_counts.append(
                    float(regime_quality_totals[quality])
                    + 8.0 * global_probability
                )
            regime_quality_alpha = (
                _smoothed_alpha(
                    shrunken_quality_counts,
                    concentration=quality_concentration,
                    base_mass=category_base_mass,
                )
                if non_colored_quality_categories
                else ()
            )
            regime_size_alpha: dict[str, list[float]] = {}
            for quality, sizes in quality_size_categories.items():
                global_size_total = sum(
                    quality_size_totals[quality].values()
                )
                shrunken_size_counts = []
                for size in sizes:
                    global_probability = (
                        quality_size_totals[quality][size]
                        / global_size_total
                        if global_size_total
                        else 1.0 / max(1, len(sizes))
                    )
                    shrunken_size_counts.append(
                        float(
                            regime_quality_sizes[quality][size]
                        )
                        + 5.0 * global_probability
                    )
                regime_size_alpha[quality] = list(
                    _smoothed_alpha(
                        shrunken_size_counts,
                        concentration=size_concentration,
                        base_mass=category_base_mass,
                    )
                )
            count_regimes.append(
                {
                    "regime_id": f"{lower:02d}-{upper:02d}",
                    "minimum_count": lower,
                    "maximum_count": upper,
                    "training_game_count": len(regime_rows),
                    "colored_hurdle": {
                        "zero_probability": (
                            regime_zero_probability
                        ),
                        "positive_count_probabilities": list(
                            _normalize_positive(
                                regime_positive_weights
                            )
                        ),
                    },
                    "non_colored_quality_alpha": list(
                        regime_quality_alpha
                    ),
                    "quality_size_alpha": regime_size_alpha,
                }
            )

        maximum_row = max(max(row_histogram, default=10), 10)
        row_denominator = sum(row_histogram.values()) + 0.5 * maximum_row
        row_probabilities = {
            str(row): (
                row_histogram[row] + 0.5
            ) / row_denominator
            for row in range(1, maximum_row + 1)
        }
        artifact = {
            "schema_version": WORLD_MODEL_SCHEMA_VERSION,
            "auction_tier": str(auction_tier),
            "catalog_semantic_sha256": catalog_semantic_hash(catalog_rows),
            "reviewed_prior_revision": reviewed_prior_revision,
            "training_fingerprint": training_fingerprint,
            "training_session_count": len(games),
            "support": {
                "max_count": max_count,
                "maximum_observed_rows": maximum_row,
            },
            "hyperparameters": {
                "quality_concentration": quality_concentration,
                "size_concentration": size_concentration,
                "identity_concentration": identity_concentration,
                "category_base_mass": category_base_mass,
                "identity_base_mass": identity_base_mass,
                "colored_hurdle_beta_prior": [0.5, 0.5],
                "count_histogram_pseudocount": 0.5,
            },
            "total_count_probabilities": list(count_probabilities),
            "colored_hurdle": {
                "zero_probability": colored_zero_probability,
                "positive_count_probabilities": list(
                    colored_positive_probabilities
                ),
            },
            "present_qualities": list(present_qualities),
            "non_colored_quality_categories": list(
                non_colored_quality_categories
            ),
            "non_colored_quality_alpha": list(
                non_colored_quality_alpha
            ),
            "quality_size_categories": quality_size_categories,
            "quality_size_alpha": quality_size_alpha,
            "identity_categories": identity_categories,
            "identity_alpha": identity_alpha,
            "count_regimes": count_regimes,
            "colored_identity_process": {
                "sampling": (
                    "weighted_without_replacement_then_positive_"
                    "duplicate_tail"
                ),
                "duplicate_extra_probability": (
                    colored_duplicate_extras + 0.5
                )
                / (colored_identity_exposures + 1.0),
                "observed_duplicate_extras": (
                    colored_duplicate_extras
                ),
                "observed_identity_exposures": (
                    colored_identity_exposures
                ),
            },
            "row_probabilities": row_probabilities,
            "contracts": {
                "duplicates_allowed": True,
                "unseen_identity_probability": "strictly_positive",
                "settlement_used_for_identity_selection": False,
                "parameterization": (
                    "count + count-regime conditional colored "
                    "hurdle/quality/size + identity "
                    "Dirichlet-multinomial"
                ),
                "count_regime_conditioned": True,
                "colored_near_without_replacement": True,
            },
        }
        artifact["artifact_sha256"] = _sha256_json(
            {
                key: value
                for key, value in artifact.items()
                if key != "artifact_sha256"
            }
        )
        return cls(artifact, catalog_rows)

    @classmethod
    def load(
        cls,
        path: Path,
        catalog: Sequence[object],
    ) -> "ProbabilisticWorldModel":
        try:
            artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorldModelError(f"无法读取 world model：{path}: {exc}") from exc
        declared = str(artifact.get("artifact_sha256") or "")
        computed = _sha256_json(
            {
                key: value
                for key, value in artifact.items()
                if key != "artifact_sha256"
            }
        )
        if declared != computed:
            raise WorldModelError("world model artifact SHA-256 不匹配")
        return cls(artifact, catalog)

    def save(self, path: Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        # The published v1 artifact uses CRLF.  Write those bytes explicitly so
        # Windows and POSIX rebuilds produce the same release SHA-256.
        payload = (
            json.dumps(self.artifact, ensure_ascii=False, indent=2)
            .replace("\n", "\r\n")
            + "\r\n"
        ).encode("utf-8")
        temporary.write_bytes(payload)
        temporary.replace(destination)

    def summary(self) -> dict:
        return {
            "schema_version": WORLD_MODEL_SCHEMA_VERSION,
            "artifact_sha256": self.artifact["artifact_sha256"],
            "catalog_semantic_sha256": self.catalog_hash,
            "training_session_count": int(
                self.artifact["training_session_count"]
            ),
            "reviewed_prior_revision": self.artifact.get(
                "reviewed_prior_revision"
            ),
            "training_fingerprint": self.artifact.get(
                "training_fingerprint"
            ),
            "duplicates_allowed": True,
            "unseen_identity_probability": "strictly_positive",
            "count_regime_conditioned": bool(
                self.count_regimes
            ),
            "colored_duplicate_extra_probability": (
                self.colored_duplicate_extra_probability
            ),
        }

    def _row_log_likelihood(
        self,
        rows: int,
        *,
        rows_exact: bool,
    ) -> float:
        rows = max(1, int(rows))
        if rows_exact:
            probability = self.row_probabilities.get(rows)
            if probability is None:
                probability = 0.25 / (
                    sum(self.row_probabilities.values())
                    + 0.25 * (rows + 1)
                )
            return math.log(max(1e-300, probability))
        probability = sum(
            value
            for candidate, value in self.row_probabilities.items()
            if candidate >= rows
        )
        if rows <= 10:
            probability = 1.0
        return math.log(max(1e-12, probability))

    def log_probability(
        self,
        counts: Sequence[int],
        *,
        rows: int,
        rows_exact: bool,
    ) -> WorldLogProbability:
        if len(counts) != len(self.catalog):
            return WorldLogProbability(
                float("-inf"),
                {"dimension": float("-inf")},
            )
        normalized = tuple(int(value) for value in counts)
        if any(value < 0 for value in normalized):
            return WorldLogProbability(
                float("-inf"),
                {"negative_count": float("-inf")},
            )
        total_count = sum(normalized)
        if total_count <= 0 or total_count > self.max_count:
            return WorldLogProbability(
                float("-inf"),
                {"total_count": float("-inf")},
            )
        components: dict[str, float] = {}
        count_probability = self.count_probabilities[total_count]
        components["total_count"] = math.log(
            max(1e-300, count_probability)
        )

        quality_counts: Counter[str] = Counter()
        size_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        identity_counts: defaultdict[str, Counter[str]] = defaultdict(
            Counter
        )
        for item, count in zip(self.catalog, normalized):
            quality = str(getattr(item, "quality"))
            size = (
                f"{int(getattr(item, 'width'))}x"
                f"{int(getattr(item, 'height'))}"
            )
            catalog_id = str(getattr(item, "catalog_id"))
            quality_counts[quality] += count
            size_counts[quality][size] += count
            identity_counts[f"{quality}|{size}"][catalog_id] += count

        regime = self._regime_for_count(total_count)
        regime_colored_hurdle = (
            regime["colored_hurdle"]
            if regime is not None
            else {
                "zero_probability": self.colored_zero_probability,
                "positive_count_probabilities": (
                    self.colored_positive_probabilities
                ),
            }
        )
        regime_quality_alpha = (
            regime["non_colored_quality_alpha"]
            if regime is not None
            else self.non_colored_quality_alpha
        )
        regime_size_alpha = (
            regime["quality_size_alpha"]
            if regime is not None
            else self.quality_size_alpha
        )
        components["count_regime"] = 0.0
        colored_count = quality_counts["彩"]
        if colored_count == 0:
            components["colored_hurdle"] = math.log(
                max(
                    1e-300,
                    float(
                        regime_colored_hurdle[
                            "zero_probability"
                        ]
                    ),
                )
            )
        else:
            positive_probabilities = tuple(
                float(value)
                for value in regime_colored_hurdle[
                    "positive_count_probabilities"
                ]
            )
            if colored_count >= len(positive_probabilities):
                return WorldLogProbability(
                    float("-inf"),
                    {"colored_count": float("-inf")},
                )
            zero_probability = float(
                regime_colored_hurdle["zero_probability"]
            )
            components["colored_hurdle"] = math.log(
                max(1e-300, 1.0 - zero_probability)
            ) + math.log(
                max(
                    1e-300,
                    positive_probabilities[colored_count],
                )
            )

        non_colored_counts = [
            quality_counts[quality]
            for quality in self.non_colored_quality_categories
        ]
        if self.non_colored_quality_categories:
            components["non_colored_quality"] = (
                dirichlet_multinomial_log_probability(
                    non_colored_counts,
                    regime_quality_alpha,
                )
            )
        elif sum(
            quality_counts[quality]
            for quality in NON_COLORED_QUALITIES
        ):
            return WorldLogProbability(
                float("-inf"),
                {"non_colored_quality": float("-inf")},
            )
        for quality in self.quality_size_categories:
            sizes = self.quality_size_categories[quality]
            group_counts = [
                size_counts[quality][size] for size in sizes
            ]
            components[f"size:{quality}"] = (
                dirichlet_multinomial_log_probability(
                    group_counts,
                    regime_size_alpha[quality],
                )
            )
            for size in sizes:
                group = f"{quality}|{size}"
                identities = self.identity_categories[group]
                counts_for_group = [
                    identity_counts[group][catalog_id]
                    for catalog_id in identities
                ]
                components[f"identity:{group}"] = (
                    dirichlet_multinomial_log_probability(
                        counts_for_group,
                        self.identity_alpha[group],
                    )
                )
        colored_duplicate_extras = sum(
            max(0, int(count) - 1)
            for item, count in zip(self.catalog, normalized)
            if str(getattr(item, "quality")) == "彩"
        )
        if colored_count:
            duplicate_probability = min(
                0.49,
                max(
                    1e-6,
                    self.colored_duplicate_extra_probability,
                ),
            )
            components["colored_identity_duplicates"] = (
                colored_duplicate_extras
                * math.log(
                    duplicate_probability
                    / (1.0 - duplicate_probability)
                )
            )
        else:
            components["colored_identity_duplicates"] = 0.0
        components["row_evidence"] = self._row_log_likelihood(
            rows,
            rows_exact=rows_exact,
        )
        return WorldLogProbability(sum(components.values()), components)

    def sample_prior_counts(
        self,
        rng: random.Random,
    ) -> tuple[int, ...]:
        total_count = _sample_categorical(
            rng,
            range(1, self.max_count + 1),
            self.count_probabilities[1:],
        )
        regime = self._regime_for_count(total_count)
        regime_colored_hurdle = (
            regime["colored_hurdle"]
            if regime is not None
            else {
                "zero_probability": self.colored_zero_probability,
                "positive_count_probabilities": (
                    self.colored_positive_probabilities
                ),
            }
        )
        zero_probability = float(
            regime_colored_hurdle["zero_probability"]
        )
        positive_count_probabilities = tuple(
            float(value)
            for value in regime_colored_hurdle[
                "positive_count_probabilities"
            ]
        )
        if rng.random() < zero_probability:
            colored_count = 0
        else:
            positive_values = list(range(1, self.max_count + 1))
            positive_weights = list(
                positive_count_probabilities[1:]
            )
            feasible_values = [
                value for value in positive_values if value <= total_count
            ]
            feasible_weights = positive_weights[: len(feasible_values)]
            colored_count = (
                _sample_categorical(
                    rng, feasible_values, feasible_weights
                )
                if feasible_values
                else 0
            )
        remaining = total_count - colored_count
        if self.non_colored_quality_categories:
            quality_probabilities = _dirichlet_draw(
                rng,
                (
                    regime["non_colored_quality_alpha"]
                    if regime is not None
                    else self.non_colored_quality_alpha
                ),
            )
            non_colored_counts = _multinomial_draw(
                rng, remaining, quality_probabilities
            )
        else:
            non_colored_counts = ()
            if remaining:
                colored_count = total_count
        quality_counts = {quality: 0 for quality in QUALITY_ORDER}
        quality_counts.update(
            {
                quality: count
                for quality, count in zip(
                    self.non_colored_quality_categories,
                    non_colored_counts,
                )
            }
        )
        quality_counts["彩"] = colored_count

        result = [0] * len(self.catalog)
        for quality in self.quality_size_categories:
            sizes = self.quality_size_categories[quality]
            size_probabilities = _dirichlet_draw(
                rng,
                (
                    regime["quality_size_alpha"][quality]
                    if regime is not None
                    else self.quality_size_alpha[quality]
                ),
            )
            drawn_size_counts = _multinomial_draw(
                rng,
                quality_counts[quality],
                size_probabilities,
            )
            for size, group_total in zip(sizes, drawn_size_counts):
                group = f"{quality}|{size}"
                identities = self.identity_categories[group]
                identity_probabilities = _dirichlet_draw(
                    rng, self.identity_alpha[group]
                )
                drawn_identity_counts = (
                    _weighted_without_replacement_draw(
                        rng,
                        group_total,
                        identity_probabilities,
                    )
                    if quality == "彩"
                    else _multinomial_draw(
                        rng,
                        group_total,
                        identity_probabilities,
                    )
                )
                for catalog_id, count in zip(
                    identities,
                    drawn_identity_counts,
                ):
                    result[self._index_by_id[catalog_id]] = count
        return tuple(result)


def build_evidence_bundle(
    *,
    decision: Mapping,
    compiled: Mapping,
    evidence: Mapping,
    rows: int,
    rows_exact: bool,
) -> dict:
    """Return an immutable, hash-addressed inference evidence contract."""
    payload = {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "decision_id": str(decision.get("decision_id") or ""),
        "session_id": str(decision.get("session_id") or ""),
        "auction_tier": str(decision.get("auction_tier") or "S"),
        "round": int(decision.get("round") or 0),
        "map_extent": {
            "rows": int(rows),
            "rows_exact": bool(rows_exact),
        },
        "hard_constraints": [
            dict(spec)
            for spec in compiled.get("constraints") or []
            if not spec.get("informational")
        ],
        "random_reveal_observations": [
            dict(value)
            for value in compiled.get(
                "random_reveal_observations"
            )
            or []
        ],
        "visible_evidence": {
            key: evidence.get(key)
            for key in (
                "track_count",
                "identity_counts",
                "quality_counts",
                "exact_quality_counts",
                "size_counts",
                "exact_size_counts",
                "joint_counts",
                "tracks",
            )
            if evidence.get(key) not in (None, {}, [], 0)
        },
        "contracts": {
            "settlement_forbidden": True,
            "not_seen_is_not_absent": True,
            "hard_constraints_fail_closed": True,
        },
    }
    forbidden_keys = {
        "settlement",
        "settlement_value",
        "outcome",
        "official_total",
    }
    serialized = _canonical_json(payload).lower()
    if any(f'"{key}"' in serialized for key in forbidden_keys):
        raise WorldModelError("evidence bundle 混入赛后结算信息")
    payload["evidence_sha256"] = _sha256_json(payload)
    return payload


def write_world_model_package(
    model: ProbabilisticWorldModel,
    output: Path,
) -> dict:
    model.save(output)
    return {
        "schema_version": WORLD_MODEL_SCHEMA_VERSION,
        "path": str(Path(output).resolve()),
        **model.summary(),
    }
