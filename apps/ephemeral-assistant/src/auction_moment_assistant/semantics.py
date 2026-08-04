from __future__ import annotations

import re


QUALITY_ALIASES = {
    "白": "白",
    "白色": "白",
    "蓝": "蓝",
    "蓝色": "蓝",
    "紫": "紫",
    "紫色": "紫",
    "金": "金",
    "金色": "金",
    "彩": "彩",
    "彩色": "彩",
}


def normalize_ocr_text(text: str) -> str:
    return (
        str(text)
        .strip()
        .replace(" ", "")
        .replace("×", "x")
        .replace("Ｘ", "x")
        .replace("，", ",")
        .replace("*", "x")
    )


def event_semantics(text: str) -> dict:
    normalized = normalize_ocr_text(text)
    result = {
        "effect": "unparsed",
        "parsed": False,
        "normalized_text": normalized,
    }

    match = re.search(r"随机完整揭示(\d+)件藏品", normalized)
    if match:
        result.update(
            effect="reveal_identity_random",
            parsed=True,
            count=int(match.group(1)),
        )
        return result

    match = re.search(r"随机显示(\d+)件藏品的?位置与品质", normalized)
    if match:
        result.update(
            effect="reveal_position_quality_random",
            parsed=True,
            count=int(match.group(1)),
        )
        return result

    match = re.search(r"显示所有规格为(\d+)x(\d+)的?藏品轮[廓廊]", normalized)
    if match:
        result.update(
            effect="reveal_outline_by_size",
            parsed=True,
            width=int(match.group(1)),
            height=int(match.group(2)),
        )
        return result

    match = re.search(r"随机显示(\d+)件藏品的位置$", normalized)
    if match:
        result.update(
            effect="reveal_position_random",
            parsed=True,
            count=int(match.group(1)),
        )
        return result

    match = re.search(
        r"显示本局藏品的最高品质(白色|蓝色|紫色|金色|彩色|白|蓝|紫|金|彩)$",
        normalized,
    )
    if match:
        result.update(
            effect="highest_quality",
            parsed=True,
            observed_quality=QUALITY_ALIASES[match.group(1)],
        )
        return result

    if re.search(r"显示本局占格数最多的藏品轮[廓廊]", normalized):
        result.update(
            effect="reveal_largest_area_item_outline",
            parsed=True,
        )
        return result

    match = re.search(r"显示本局单个单元格最高价值([\d,]+)", normalized)
    if match:
        result.update(
            effect="max_value_per_cell",
            parsed=True,
            observed_value=int(match.group(1).replace(",", "")),
        )
        return result

    match = re.search(r"显示本局单件价值最高的藏品价值([\d,]+)", normalized)
    if match:
        result.update(
            effect="max_item_value",
            parsed=True,
            observed_value=int(match.group(1).replace(",", "")),
        )
        return result

    quality_prefix = (
        r"计算本局所有品质为"
        r"(白色|蓝色|紫色|金色|彩色|白|蓝|紫|金|彩)"
        r"的藏品的"
    )
    match = re.search(quality_prefix + r"平均占用格数(\d+(?:\.\d+)?)", normalized)
    if match:
        result.update(
            effect="quality_average_area",
            parsed=True,
            quality=QUALITY_ALIASES[match.group(1)],
            observed_average_cells=float(match.group(2)),
        )
        return result

    match = re.search(quality_prefix + r"平均价值([\d,]+)", normalized)
    if match:
        result.update(
            effect="quality_average_value",
            parsed=True,
            quality=QUALITY_ALIASES[match.group(1)],
            observed_value=int(match.group(2).replace(",", "")),
        )
        return result

    match = re.search(quality_prefix + r"总占(?:用)?格数(\d+)", normalized)
    if match:
        result.update(
            effect="quality_total_area",
            parsed=True,
            quality=QUALITY_ALIASES[match.group(1)],
            observed_cells=int(match.group(2)),
        )
        return result

    match = re.search(r"计算本局所有藏品的平均占用格数(\d+(?:\.\d+)?)", normalized)
    if match:
        result.update(
            effect="average_item_area",
            parsed=True,
            observed_average_cells=float(match.group(1)),
        )
        return result

    match = re.search(r"计算本局所有藏品的总占格数(\d+)", normalized)
    if match:
        result.update(
            effect="total_occupied_cells",
            parsed=True,
            observed_cells=int(match.group(1)),
        )
        return result

    match = re.search(
        r"(?:显示|计算)本局(?:所有)?藏品(?:的)?(?:总数|总数量|数量)(\d+)",
        normalized,
    )
    if match:
        result.update(
            effect="total_item_count",
            parsed=True,
            observed_count=int(match.group(1)),
        )
        return result

    match = re.search(
        r"计算本局(白色|蓝色|紫色|金色|彩色|白|蓝|紫|金|彩)"
        r"(?:品质)?藏品总价值([\d,]+)",
        normalized,
    )
    if match:
        result.update(
            effect="quality_total_value",
            parsed=True,
            quality=QUALITY_ALIASES[match.group(1)],
            observed_value=int(match.group(2).replace(",", "")),
        )
        return result

    match = re.search(
        r"显示本局(白色|蓝色|紫色|金色|彩色|白|蓝|紫|金|彩)"
        r"(?:品质)?藏品(?:数|数量)(\d+)",
        normalized,
    )
    if match:
        result.update(
            effect="quality_item_count",
            parsed=True,
            quality=QUALITY_ALIASES[match.group(1)],
            observed_count=int(match.group(2)),
        )
        return result

    match = re.search(
        r"计算本局所有规格为(\d+)x(\d+)的?藏品的?"
        r"(平均价值|总价值|数量|平均占用格数|平均占格数|总占用格数|总占格数)"
        r"([\d,]+(?:\.\d+)?)",
        normalized,
    )
    if match:
        metric_map = {
            "平均价值": "size_average_value",
            "总价值": "size_total_value",
            "数量": "size_item_count",
            "平均占用格数": "size_average_area",
            "平均占格数": "size_average_area",
            "总占用格数": "size_total_area",
            "总占格数": "size_total_area",
        }
        effect = metric_map[match.group(3)]
        result.update(
            effect=effect,
            parsed=True,
            width=int(match.group(1)),
            height=int(match.group(2)),
        )
        value = match.group(4).replace(",", "")
        if effect == "size_average_area":
            result["observed_average_cells"] = float(value)
        elif effect == "size_total_area":
            result["observed_cells"] = int(value)
        elif effect == "size_item_count":
            result["observed_count"] = int(value)
        else:
            result["observed_value"] = int(value)
        return result

    match = re.search(
        r"显示所有(白色|蓝色|紫色|金色|彩色|白|蓝|紫|金|彩)"
        r"(?:品质)?藏品轮廓",
        normalized,
    )
    if match:
        result.update(
            effect="reveal_outline_by_quality",
            parsed=True,
            quality=QUALITY_ALIASES[match.group(1)],
        )
        return result

    match = re.search(
        r"显示所有(白色|蓝色|紫色|金色|彩色|白|蓝|紫|金|彩)"
        r"(?:品质)?藏品的?位置",
        normalized,
    )
    if match:
        result.update(
            effect="reveal_position_by_quality",
            parsed=True,
            quality=QUALITY_ALIASES[match.group(1)],
        )
        return result

    return result
