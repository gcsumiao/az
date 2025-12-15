from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class Thresholds:
    fill_rate: float = 0.9
    outs_exposure: float = 200.0
    coverage: float = 1.0
    woh_min: float = 2.0
    woh_max: float = 12.0
    billbacks_pct: float = 0.08
    min_ly_rev_floor: float = 100.0


@dataclass(frozen=True)
class DashboardFilters:
    selected_weeks: List[int] = field(default_factory=list)
    selected_categories: List[str] = field(default_factory=list)
    selected_parts: List[str] = field(default_factory=list)
    sku_query: str = ""
    top_n: int = 15
    thresholds: Thresholds = field(default_factory=Thresholds)
    show_forecast_overlay: bool = True


def _as_int_list(values: Optional[Iterable[object]]) -> List[int]:
    if not values:
        return []
    out: List[int] = []
    for v in values:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def normalize_filters(raw: dict, *, available_weeks: Optional[List[int]] = None) -> DashboardFilters:
    available_weeks = sorted(available_weeks or [])

    selected_weeks = _as_int_list(raw.get("selected_weeks"))
    if not selected_weeks:
        selected_weeks = (available_weeks[-4:] if len(available_weeks) >= 4 else available_weeks)

    selected_categories = [str(x) for x in (raw.get("selected_categories") or []) if x is not None]
    selected_parts = [str(x) for x in (raw.get("selected_parts") or []) if x is not None]
    sku_query = (raw.get("sku_query") or "").strip()

    top_n = raw.get("top_n", 15)
    try:
        top_n = int(top_n)
    except Exception:
        top_n = 15
    top_n = max(1, min(200, top_n))

    t = raw.get("thresholds") or {}
    thresholds = Thresholds(
        fill_rate=float(t.get("fill_rate", 0.9)),
        outs_exposure=float(t.get("outs_exposure", 200.0)),
        coverage=float(t.get("coverage", 1.0)),
        woh_min=float(t.get("woh_min", 2.0)),
        woh_max=float(t.get("woh_max", 12.0)),
        billbacks_pct=float(t.get("billbacks_pct", 0.08)),
        min_ly_rev_floor=float(t.get("min_ly_rev_floor", 100.0)),
    )

    show_forecast_overlay = bool(raw.get("show_forecast_overlay", True))
    return DashboardFilters(
        selected_weeks=selected_weeks,
        selected_categories=selected_categories,
        selected_parts=selected_parts,
        sku_query=sku_query,
        top_n=top_n,
        thresholds=thresholds,
        show_forecast_overlay=show_forecast_overlay,
    )

