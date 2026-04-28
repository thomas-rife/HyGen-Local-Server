from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ValidationReport:
    ok: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def merge(self, other: "ValidationReport") -> "ValidationReport":
        self.ok = self.ok and other.ok
        self.warnings.extend(other.warnings)
        self.errors.extend(other.errors)
        self.metrics.update(other.metrics)
        return self

    def summary(self, prefix: str = "validation") -> str:
        parts = [
            f"{prefix}: ok={self.ok}",
            f"errors={len(self.errors)}",
            f"warnings={len(self.warnings)}",
        ]

        interesting = [
            "height_range",
            "playable_cells",
            "playable_mean_slope",
            "water_fraction",
            "river_cells",
            "lake_cells",
        ]

        for key in interesting:
            if key in self.metrics:
                parts.append(f"{key}={self.metrics[key]:.3f}")

        return " ".join(parts)


def _slope(height: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(height.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _prompt_text(params: Dict[str, Any]) -> str:
    return str(params.get("_matched_text") or params.get("_prompt") or "").lower()


def _mask(params: Dict[str, Any], name: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if name == "playable":
        m = params.get("_playable_mask")
        if isinstance(m, np.ndarray) and m.shape == shape:
            return m

    feature_masks = params.get("_feature_masks")
    if isinstance(feature_masks, dict):
        m = feature_masks.get(name)
        if isinstance(m, np.ndarray) and m.shape == shape:
            return m

    return None


def validate_macro(
    macro_raw: np.ndarray,
    adapted_params: Dict[str, Any],
    *,
    sea_level: int = 63,
) -> ValidationReport:
    """Validate macro terrain before img2img.

    This catches bad parser mapping, missing playable masks, missing water masks,
    and obviously unusable terrain before model time is spent.
    """
    report = ValidationReport()

    if macro_raw.ndim != 2:
        report.error(f"macro_raw must be 2D, got shape={macro_raw.shape}")
        return report

    h = macro_raw.astype(np.float32)
    shape = h.shape
    text = _prompt_text(adapted_params)
    base_shape = str(adapted_params.get("base_shape", "")).lower()

    h_min = float(np.min(h))
    h_max = float(np.max(h))
    h_range = h_max - h_min
    report.metrics["height_min"] = h_min
    report.metrics["height_max"] = h_max
    report.metrics["height_range"] = h_range

    if not np.isfinite(h).all():
        report.error("macro contains non-finite height values")

    if h_range < 1.0:
        report.warn("macro height range is very small; terrain may look flat or broken")

    if h_range > 220.0:
        report.warn("macro height range is extremely high; terrain may be too steep or clipped")

    playable = _mask(adapted_params, "playable", shape)
    if playable is None:
        report.error("missing playable mask")
    else:
        playable_bool = playable > 0.5
        playable_cells = int(np.sum(playable_bool))
        report.metrics["playable_cells"] = float(playable_cells)

        if playable_cells < 300:
            report.error(f"playable area too small: {playable_cells} cells")

        slope = _slope(h)
        if playable_cells > 0:
            mean_slope = float(np.mean(slope[playable_bool]))
            max_slope = float(np.percentile(slope[playable_bool], 95))
            min_height = float(np.min(h[playable_bool]))
            report.metrics["playable_mean_slope"] = mean_slope
            report.metrics["playable_p95_slope"] = max_slope
            report.metrics["playable_min_height"] = min_height

            if mean_slope > 2.0:
                report.warn(f"playable area mean slope is high: {mean_slope:.2f}")
            if max_slope > 5.0:
                report.warn(f"playable area p95 slope is high: {max_slope:.2f}")
            if min_height <= sea_level + 1:
                report.error(f"playable area is too close to/below water level: min={min_height:.2f}")

    water_hint = _mask(adapted_params, "water_hint", shape)
    lake = _mask(adapted_params, "lake", shape)
    river = _mask(adapted_params, "river", shape)

    if water_hint is not None:
        report.metrics["water_fraction"] = float(np.mean(water_hint > 0))

    wants_river = "river" in text or "stream" in text or "creek" in text or "river" in base_shape
    wants_lake = "lake" in text or "tarn" in text or "loch" in text or "lake" in base_shape

    if wants_river:
        if river is None:
            report.error("river prompt/base_shape but missing river feature mask")
        else:
            river_cells = int(np.sum(river > 0))
            report.metrics["river_cells"] = float(river_cells)
            if river_cells < 50:
                report.error(f"river feature mask too small: {river_cells}")

    if wants_lake:
        if lake is None:
            report.error("lake prompt/base_shape but missing lake feature mask")
        else:
            lake_cells = int(np.sum(lake > 0))
            report.metrics["lake_cells"] = float(lake_cells)
            if lake_cells < 100:
                report.error(f"lake feature mask too small: {lake_cells}")
            if lake_cells > h.size * 0.45:
                report.warn(f"lake feature mask very large: {lake_cells / h.size:.2%}")

    if "plain" in text or "plains" in text or base_shape == "plains":
        if h_range > 30.0 and "river" not in text:
            report.warn(f"plain/plains terrain relief may be too high: range={h_range:.2f}")

    return report


def validate_final_package_inputs(
    stitched_raw: np.ndarray,
    adapted_params: Dict[str, Any],
    *,
    sea_level: int = 63,
) -> ValidationReport:
    """Validate final heightmap before build_terrain_package."""
    report = ValidationReport()

    if stitched_raw.ndim != 2:
        report.error(f"stitched_raw must be 2D, got shape={stitched_raw.shape}")
        return report

    h = stitched_raw.astype(np.float32)
    shape = h.shape
    text = _prompt_text(adapted_params)

    if not np.isfinite(h).all():
        report.error("stitched_raw contains non-finite values")

    report.metrics["height_min"] = float(np.min(h))
    report.metrics["height_max"] = float(np.max(h))
    report.metrics["height_range"] = float(np.max(h) - np.min(h))

    playable = _mask(adapted_params, "playable", shape)
    if playable is not None:
        playable_bool = playable > 0.5
        report.metrics["playable_cells"] = float(np.sum(playable_bool))
        if np.any(playable_bool):
            s = _slope(h)
            report.metrics["playable_mean_slope"] = float(np.mean(s[playable_bool]))
            report.metrics["playable_p95_slope"] = float(np.percentile(s[playable_bool], 95))
            report.metrics["playable_min_height"] = float(np.min(h[playable_bool]))

            if report.metrics["playable_min_height"] <= sea_level + 1:
                report.error("final playable area is at/below water level")
            if report.metrics["playable_mean_slope"] > 2.5:
                report.warn("final playable area is still somewhat bumpy")

    river = _mask(adapted_params, "river", shape)
    lake = _mask(adapted_params, "lake", shape)

    if ("river" in text or "stream" in text or "creek" in text) and river is not None:
        river_cells = int(np.sum(river > 0))
        report.metrics["river_cells"] = float(river_cells)
        if river_cells < 50:
            report.error("final river feature mask too small")

    if ("lake" in text or "tarn" in text or "loch" in text) and lake is not None:
        lake_cells = int(np.sum(lake > 0))
        report.metrics["lake_cells"] = float(lake_cells)
        if lake_cells < 100:
            report.error("final lake feature mask too small")

    return report


def format_validation_lines(report: ValidationReport, prefix: str) -> List[str]:
    lines = [report.summary(prefix=prefix)]
    for err in report.errors:
        lines.append(f"{prefix} ERROR: {err}")
    for warn in report.warnings:
        lines.append(f"{prefix} WARN: {warn}")
    return lines
