from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from auction_moment_research.world_model import (
    WorldLogProbability,
    WorldModelError,
    catalog_semantic_hash,
)


WORLD_MODEL_V2_SCHEMA_VERSION = "world-model-v2"
GENERATION_FORMULA_VERSION = "treasure-generation-v1.9-20260803"
QUALITY_ORDER = ("白", "蓝", "紫", "金", "彩")
NON_COLORED_QUALITIES = QUALITY_ORDER[:-1]
DEFAULT_MINIMUM_QUOTAS = {
    "白": 1,
    "蓝": 1,
    "紫": 3,
    "金": 3,
    "彩": 0,
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _softmax(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise WorldModelError("softmax 不能为空")
    maximum = max(float(value) for value in values)
    weights = [math.exp(float(value) - maximum) for value in values]
    total = sum(weights)
    return tuple(value / total for value in weights)


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    cleaned = [max(0.0, float(value)) for value in values]
    total = sum(cleaned)
    if total <= 0.0:
        raise WorldModelError("概率权重之和必须为正")
    return tuple(value / total for value in cleaned)


def _multinomial_log_probability(
    counts: Sequence[int],
    probabilities: Sequence[float],
) -> float:
    if len(counts) != len(probabilities):
        raise WorldModelError("multinomial 维度不一致")
    normalized = tuple(int(value) for value in counts)
    if any(value < 0 for value in normalized):
        return float("-inf")
    total = sum(normalized)
    result = math.lgamma(total + 1)
    for count, probability in zip(normalized, probabilities):
        if count and probability <= 0.0:
            return float("-inf")
        result -= math.lgamma(count + 1)
        if count:
            result += count * math.log(probability)
    return result


def _sample_multinomial(
    rng: random.Random,
    total: int,
    probabilities: Sequence[float],
) -> tuple[int, ...]:
    counts = [0] * len(probabilities)
    for selected in rng.choices(
        range(len(probabilities)),
        weights=probabilities,
        k=max(0, int(total)),
    ):
        counts[int(selected)] += 1
    return tuple(counts)


def _polynomial(coefficients: Sequence[float], x: float) -> float:
    result = 0.0
    power = 1.0
    for coefficient in coefficients:
        result += float(coefficient) * power
        power *= float(x)
    return result


def _com_binomial_probabilities(
    maximum: int,
    mean: float,
    nu: float,
) -> tuple[float, ...]:
    maximum = int(maximum)
    if maximum < 0:
        raise WorldModelError("COM-binomial maximum 不能为负")
    if maximum == 0:
        return (1.0,)
    desired = min(float(maximum), max(0.0, float(mean)))
    log_choose = tuple(
        math.lgamma(maximum + 1)
        - math.lgamma(value + 1)
        - math.lgamma(maximum - value + 1)
        for value in range(maximum + 1)
    )

    def probabilities(theta: float) -> tuple[float, ...]:
        scores = tuple(
            float(nu) * log_choose[value] + theta * value
            for value in range(maximum + 1)
        )
        return _softmax(scores)

    low, high = -40.0, 40.0
    for _ in range(90):
        middle = (low + high) / 2.0
        candidate = probabilities(middle)
        expectation = sum(
            value * probability
            for value, probability in enumerate(candidate)
        )
        if expectation < desired:
            low = middle
        else:
            high = middle
    return probabilities((low + high) / 2.0)


class GenerativeWorldModelV2:
    """Formula-driven prior for complete S-tier treasure multisets.

    This model is deliberately separate from ``world-model-v1``.  It encodes
    the generation contracts recovered from the larger protocol-settlement
    corpus rather than fitting count regimes and quality/size Dirichlet layers
    to the small reviewed-image prior.
    """

    def __init__(
        self,
        artifact: Mapping,
        catalog: Sequence[object],
    ) -> None:
        self.artifact = json.loads(_canonical_json(dict(artifact)))
        if (
            self.artifact.get("schema_version")
            != WORLD_MODEL_V2_SCHEMA_VERSION
        ):
            raise WorldModelError("world-model-v2 schema 不兼容")
        declared = str(self.artifact.get("artifact_sha256") or "")
        computed = _sha256_json(
            {
                key: value
                for key, value in self.artifact.items()
                if key != "artifact_sha256"
            }
        )
        if declared != computed:
            raise WorldModelError("world-model-v2 artifact SHA-256 不匹配")
        self.catalog = tuple(catalog)
        self.catalog_hash = catalog_semantic_hash(self.catalog)
        if (
            self.artifact.get("catalog_semantic_sha256")
            != self.catalog_hash
        ):
            raise WorldModelError("world-model-v2 图鉴语义哈希不匹配")

        support = self.artifact.get("support") or {}
        self.min_count = int(support.get("min_count") or 10)
        self.max_count = int(support.get("max_count") or 60)
        if self.min_count != 10 or self.max_count != 60:
            raise WorldModelError("world-model-v2 件数支持必须为 10..60")
        self.minimum_quotas = {
            str(key): int(value)
            for key, value in (
                self.artifact.get("minimum_quotas") or {}
            ).items()
        }
        if self.minimum_quotas != DEFAULT_MINIMUM_QUOTAS:
            raise WorldModelError("world-model-v2 最低配额不匹配")

        self._indices_by_quality: dict[str, tuple[int, ...]] = {}
        self._index_by_id: dict[str, int] = {}
        for index, item in enumerate(self.catalog):
            catalog_id = str(getattr(item, "catalog_id"))
            if catalog_id in self._index_by_id:
                raise WorldModelError(f"图鉴 ID 重复：{catalog_id}")
            self._index_by_id[catalog_id] = index
        for quality in QUALITY_ORDER:
            indices = tuple(
                index
                for index, item in enumerate(self.catalog)
                if str(getattr(item, "quality")) == quality
            )
            if not indices:
                raise WorldModelError(f"图鉴缺少品质：{quality}")
            self._indices_by_quality[quality] = indices
        unknown_qualities = sorted(
            {
                str(getattr(item, "quality"))
                for item in self.catalog
            }
            - set(QUALITY_ORDER)
        )
        if unknown_qualities:
            raise WorldModelError(
                "图鉴含未知品质：" + ", ".join(unknown_qualities)
            )

        count_model = self.artifact.get("total_count_model") or {}
        self._count_probabilities = self._build_count_probabilities(
            count_model
        )
        self.count_probabilities = tuple(
            self._count_probabilities.get(value, 0.0)
            for value in range(self.max_count + 1)
        )
        quality_model = self.artifact.get("quality_count_model") or {}
        self.quality_logit_coefficients = {
            str(key): tuple(float(value) for value in coefficients)
            for key, coefficients in (
                quality_model.get("all_quality_logits") or {}
            ).items()
        }
        self.non_colored_logit_coefficients = {
            str(key): tuple(float(value) for value in coefficients)
            for key, coefficients in (
                quality_model.get("non_colored_logits_after_colored")
                or {}
            ).items()
        }
        if set(self.quality_logit_coefficients) != set(QUALITY_ORDER):
            raise WorldModelError("五品质 p_N 系数不完整")
        if set(self.non_colored_logit_coefficients) != set(
            NON_COLORED_QUALITIES
        ):
            raise WorldModelError("非彩四色 q_N 系数不完整")
        self.colored_count_nu = float(
            quality_model.get("colored_count_nu") or 1.40
        )

        identity_model = self.artifact.get("identity_model") or {}
        self.colored_identity_mode = str(
            identity_model.get("colored_mode")
            or "n_conditioned_value_power"
        )
        self.colored_global_exponent = float(
            identity_model.get("colored_global_exponent") or -0.32
        )
        self.colored_alpha = float(
            identity_model.get("colored_alpha") or -0.2863
        )
        self.colored_beta = float(
            identity_model.get("colored_beta") or -0.4489
        )
        if self.colored_identity_mode not in {
            "global_value_power",
            "n_conditioned_value_power",
        }:
            raise WorldModelError("未知彩色身份模型")
        if any(int(getattr(item, "value")) <= 0 for item in self.catalog):
            raise WorldModelError("身份价值必须为正")

        row_model = self.artifact.get("map_rows_count_update") or {}
        self.row_count_intercept = float(
            row_model.get("intercept") or -2.63
        )
        self.row_count_slope = float(row_model.get("slope") or 2.913)
        self.row_count_mae = float(row_model.get("heldout_mae") or 2.74)
        if self.row_count_mae <= 0.0:
            raise WorldModelError("map_rows 件数误差必须为正")

        # The MCMC proposal layer consumes the v1 category/alpha attributes.
        # Expose one quality-level group per quality.  These are proposal
        # weights only; exact v2 scores still use the N-dependent likelihood.
        self.identity_categories = {}
        self.identity_alpha = {}
        for quality, indices in self._indices_by_quality.items():
            group = f"quality:{quality}"
            self.identity_categories[group] = tuple(
                str(getattr(self.catalog[index], "catalog_id"))
                for index in indices
            )
            proposal = self._identity_probabilities(
                quality,
                total_count=35,
                force_global_colored=True,
            )
            self.identity_alpha[group] = tuple(
                0.05 + 64.0 * probability
                for probability in proposal
            )
        self._com_cache: dict[tuple[int, float, float], tuple[float, ...]] = {}

    @staticmethod
    def _build_count_probabilities(
        model: Mapping,
    ) -> dict[int, float]:
        low_spike = float(model.get("probability_n10") or 0.0842)
        high_spike = float(model.get("probability_n60") or 0.0649)
        middle_mass = float(model.get("middle_mass") or 0.8509)
        location = float(model.get("middle_location") or 29.57)
        deviation = float(model.get("middle_stddev") or 17.20)
        if (
            min(low_spike, high_spike, middle_mass) < 0.0
            or abs(low_spike + high_spike + middle_mass - 1.0) > 1e-6
            or deviation <= 0.0
        ):
            raise WorldModelError("总件数边界钟形参数无效")
        middle_weights = tuple(
            math.exp(-0.5 * ((value - location) / deviation) ** 2)
            for value in range(11, 60)
        )
        normalized = _normalize(middle_weights)
        result = {10: low_spike, 60: high_spike}
        result.update(
            {
                value: middle_mass * probability
                for value, probability in zip(range(11, 60), normalized)
            }
        )
        return result

    @classmethod
    def from_formula(
        cls,
        catalog: Sequence[object],
        *,
        auction_tier: str = "S",
        training_provenance: Mapping | None = None,
        colored_identity_mode: str = "n_conditioned_value_power",
        production_enabled: bool = False,
    ) -> "GenerativeWorldModelV2":
        if production_enabled:
            raise WorldModelError(
                "world-model-v2 尚无新鲜时间块证据，不能标记 production"
            )
        provenance = dict(training_provenance or {})
        artifact = {
            "schema_version": WORLD_MODEL_V2_SCHEMA_VERSION,
            "formula_version": GENERATION_FORMULA_VERSION,
            "auction_tier": str(auction_tier),
            "production_enabled": False,
            "promotion_status": (
                "research_only_requires_fresh_time_block"
            ),
            "catalog_semantic_sha256": catalog_semantic_hash(catalog),
            "training_provenance": provenance,
            "support": {"min_count": 10, "max_count": 60},
            "minimum_quotas": dict(DEFAULT_MINIMUM_QUOTAS),
            "total_count_model": {
                "kind": "boundary_spikes_plus_discrete_truncated_normal",
                "probability_n10": 0.0842,
                "probability_n60": 0.0649,
                "middle_mass": 0.8509,
                "middle_support": [11, 59],
                "middle_location": 29.57,
                "middle_stddev": 17.20,
            },
            "quality_count_model": {
                "kind": (
                    "hard_quotas_then_com_binomial_colored_then_"
                    "n_conditioned_multinomial"
                ),
                "all_quality_logits": {
                    "白": [0.3884, 0.7019, -0.1980, 0.0526],
                    "蓝": [1.2033, 0.5728, -0.3861, 0.3727],
                    "紫": [1.5354, 0.5380, -0.2987, 0.2753],
                    "金": [1.5074, 0.5718, -0.3209, 0.1357],
                    "彩": [0.0, 0.0, 0.0, 0.0],
                },
                "colored_count_nu": 1.40,
                "non_colored_logits_after_colored": {
                    "白": [-1.118934, 0.130317, 0.122754, -0.083346],
                    "蓝": [-0.304177, 0.001631, -0.064484, 0.235453],
                    "紫": [0.028032, -0.033568, 0.022486, 0.138962],
                    "金": [0.0, 0.0, 0.0, 0.0],
                },
            },
            "identity_model": {
                "non_colored": "uniform_with_replacement_within_quality",
                "colored": "weighted_with_replacement_across_quality",
                "colored_mode": str(colored_identity_mode),
                "colored_global_exponent": -0.32,
                "colored_alpha": -0.2863,
                "colored_beta": -0.4489,
                "shape_layer": "identity_carries_width_and_height",
            },
            "sequence_and_layout": {
                "index_order": "approximately_uniform_random_permutation",
                "columns": 10,
                "packing": "row_major_first_fit",
                "rotation": False,
                "map_rows": "maximum_occupied_row_after_packing",
            },
            "map_rows_count_update": {
                "kind": "generalized_likelihood_ratio_from_n_given_rows",
                "intercept": -2.63,
                "slope": 2.913,
                "heldout_mae": 2.74,
                "applies_only_when_height_exact": True,
            },
            "contracts": {
                "hard_minimum_quotas": True,
                "colored_underdispersed": True,
                "identities_with_replacement": True,
                "non_colored_uniform_within_quality": True,
                "shape_sampled_separately": False,
                "settlement_used_for_live_features": False,
                "fresh_validation_required": True,
            },
        }
        artifact["artifact_sha256"] = _sha256_json(artifact)
        return cls(artifact, catalog)

    @classmethod
    def load(
        cls,
        path: Path,
        catalog: Sequence[object],
    ) -> "GenerativeWorldModelV2":
        try:
            artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorldModelError(
                f"无法读取 world-model-v2：{path}: {exc}"
            ) from exc
        return cls(artifact, catalog)

    def save(self, path: Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    def summary(self) -> dict:
        provenance = self.artifact.get("training_provenance") or {}
        return {
            "schema_version": WORLD_MODEL_V2_SCHEMA_VERSION,
            "formula_version": self.artifact["formula_version"],
            "artifact_sha256": self.artifact["artifact_sha256"],
            "catalog_semantic_sha256": self.catalog_hash,
            "auction_tier": self.artifact["auction_tier"],
            "production_enabled": False,
            "promotion_status": self.artifact["promotion_status"],
            "training_source_kind": provenance.get("source_kind"),
            "training_session_count": provenance.get("session_count"),
            "training_min_session_id": provenance.get("min_session_id"),
            "training_max_session_id": provenance.get("max_session_id"),
            "training_fingerprint": provenance.get("fingerprint"),
            "hard_minimum_quotas": True,
            "colored_underdispersed": True,
            "identities_with_replacement": True,
            "count_regime_conditioned": False,
            "n_conditioned_quality_curve": True,
            "map_rows_count_update": True,
            "colored_identity_mode": self.colored_identity_mode,
        }

    def _quality_probabilities(
        self,
        total_count: int,
    ) -> tuple[float, ...]:
        x = (int(total_count) - 35.0) / 25.0
        return _softmax(
            [
                _polynomial(self.quality_logit_coefficients[quality], x)
                for quality in QUALITY_ORDER
            ]
        )

    def _non_colored_probabilities(
        self,
        total_count: int,
    ) -> tuple[float, ...]:
        x = (int(total_count) - 35.0) / 25.0
        return _softmax(
            [
                _polynomial(
                    self.non_colored_logit_coefficients[quality],
                    x,
                )
                for quality in NON_COLORED_QUALITIES
            ]
        )

    def _colored_count_probabilities(
        self,
        total_count: int,
    ) -> tuple[float, ...]:
        remaining = int(total_count) - sum(self.minimum_quotas.values())
        colored_probability = self._quality_probabilities(total_count)[-1]
        mean = remaining * colored_probability
        key = (
            remaining,
            round(mean, 12),
            round(self.colored_count_nu, 12),
        )
        cached = self._com_cache.get(key)
        if cached is None:
            cached = _com_binomial_probabilities(
                remaining,
                mean,
                self.colored_count_nu,
            )
            self._com_cache[key] = cached
        return cached

    def _identity_probabilities(
        self,
        quality: str,
        *,
        total_count: int,
        force_global_colored: bool = False,
    ) -> tuple[float, ...]:
        indices = self._indices_by_quality[quality]
        if quality != "彩":
            return tuple(1.0 / len(indices) for _ in indices)
        exponent = self.colored_global_exponent
        if (
            not force_global_colored
            and self.colored_identity_mode
            == "n_conditioned_value_power"
        ):
            exponent = self.colored_alpha + self.colored_beta * (
                (int(total_count) - 35.0) / 25.0
            )
        return _normalize(
            [
                float(getattr(self.catalog[index], "value")) ** exponent
                for index in indices
            ]
        )

    def _row_count_log_likelihood_ratio(
        self,
        total_count: int,
        rows: int,
        *,
        rows_exact: bool,
    ) -> float:
        if not rows_exact:
            return 0.0
        expected = min(
            float(self.max_count),
            max(
                float(self.min_count),
                self.row_count_intercept
                + self.row_count_slope * max(1, int(rows)),
            ),
        )
        conditional = _normalize(
            [
                math.exp(-abs(value - expected) / self.row_count_mae)
                for value in range(self.min_count, self.max_count + 1)
            ]
        )
        conditional_probability = conditional[
            int(total_count) - self.min_count
        ]
        prior_probability = self._count_probabilities[int(total_count)]
        return math.log(conditional_probability) - math.log(
            prior_probability
        )

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
        if total_count not in self._count_probabilities:
            return WorldLogProbability(
                float("-inf"),
                {"total_count": float("-inf")},
            )
        quality_counts = Counter()
        for item, count in zip(self.catalog, normalized):
            quality_counts[str(getattr(item, "quality"))] += count
        if any(
            quality_counts[quality] < minimum
            for quality, minimum in self.minimum_quotas.items()
        ):
            return WorldLogProbability(
                float("-inf"),
                {"minimum_quotas": float("-inf")},
            )

        components: dict[str, float] = {
            "total_count": math.log(
                self._count_probabilities[total_count]
            ),
            "minimum_quotas": 0.0,
        }
        extra_capacity = total_count - sum(self.minimum_quotas.values())
        colored_count = quality_counts["彩"]
        if colored_count > extra_capacity:
            return WorldLogProbability(
                float("-inf"),
                {"colored_count_given_n": float("-inf")},
            )
        colored_probabilities = self._colored_count_probabilities(
            total_count
        )
        components["colored_count_given_n"] = math.log(
            max(1e-300, colored_probabilities[colored_count])
        )

        non_colored_extras = tuple(
            quality_counts[quality] - self.minimum_quotas[quality]
            for quality in NON_COLORED_QUALITIES
        )
        if sum(non_colored_extras) != extra_capacity - colored_count:
            return WorldLogProbability(
                float("-inf"),
                {"non_colored_extra_quality_given_n": float("-inf")},
            )
        components["non_colored_extra_quality_given_n"] = (
            _multinomial_log_probability(
                non_colored_extras,
                self._non_colored_probabilities(total_count),
            )
        )
        for quality in QUALITY_ORDER:
            indices = self._indices_by_quality[quality]
            identity_counts = tuple(normalized[index] for index in indices)
            components[f"identity:{quality}"] = (
                _multinomial_log_probability(
                    identity_counts,
                    self._identity_probabilities(
                        quality,
                        total_count=total_count,
                    ),
                )
            )
        components["map_rows_count_likelihood_ratio"] = (
            self._row_count_log_likelihood_ratio(
                total_count,
                rows,
                rows_exact=rows_exact,
            )
        )
        return WorldLogProbability(sum(components.values()), components)

    def sample_prior_counts(
        self,
        rng: random.Random,
    ) -> tuple[int, ...]:
        total_count = int(
            rng.choices(
                list(self._count_probabilities),
                weights=list(self._count_probabilities.values()),
                k=1,
            )[0]
        )
        extra_capacity = total_count - sum(self.minimum_quotas.values())
        colored_probabilities = self._colored_count_probabilities(
            total_count
        )
        colored_count = int(
            rng.choices(
                range(extra_capacity + 1),
                weights=colored_probabilities,
                k=1,
            )[0]
        )
        non_colored_extras = _sample_multinomial(
            rng,
            extra_capacity - colored_count,
            self._non_colored_probabilities(total_count),
        )
        quality_counts = {
            quality: self.minimum_quotas[quality] + extra
            for quality, extra in zip(
                NON_COLORED_QUALITIES,
                non_colored_extras,
            )
        }
        quality_counts["彩"] = colored_count

        result = [0] * len(self.catalog)
        for quality in QUALITY_ORDER:
            indices = self._indices_by_quality[quality]
            identity_counts = _sample_multinomial(
                rng,
                quality_counts[quality],
                self._identity_probabilities(
                    quality,
                    total_count=total_count,
                ),
            )
            for index, count in zip(indices, identity_counts):
                result[index] = count
        if sum(result) != total_count:
            raise WorldModelError("world-model-v2 抽样件数不守恒")
        return tuple(result)


def load_world_model(
    path: Path,
    catalog: Sequence[object],
):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorldModelError(f"无法读取 world model：{path}: {exc}") from exc
    schema = str(payload.get("schema_version") or "")
    if schema == WORLD_MODEL_V2_SCHEMA_VERSION:
        return GenerativeWorldModelV2(payload, catalog)
    if schema == "world-model-v1":
        from probabilistic_world_model import ProbabilisticWorldModel

        return ProbabilisticWorldModel.load(path, catalog)
    raise WorldModelError(f"未知 world model schema：{schema or '<missing>'}")
