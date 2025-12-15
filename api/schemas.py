from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ThresholdsModel(BaseModel):
    fill_rate: float = 0.9
    outs_exposure: float = 200.0
    coverage: float = 1.0
    woh_min: float = 2.0
    woh_max: float = 12.0
    billbacks_pct: float = 0.08
    min_ly_rev_floor: float = 100.0


class DashboardFiltersModel(BaseModel):
    selected_weeks: List[int] = Field(default_factory=list)
    selected_categories: List[str] = Field(default_factory=list)
    selected_parts: List[str] = Field(default_factory=list)
    sku_query: str = ""
    top_n: int = 15
    thresholds: ThresholdsModel = Field(default_factory=ThresholdsModel)
    show_forecast_overlay: bool = True


class MetaPartsResponse(BaseModel):
    parts: List[str]


class MetaListResponse(BaseModel):
    values: List[str]

