from __future__ import annotations

from dataclasses import asdict
import logging
import math
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from fastapi.encoders import jsonable_encoder

from api.schemas import DashboardFiltersModel
from core.data import get_source_files, load_dashboard_data, prepare_context
from core.filters import DashboardFilters, Thresholds, normalize_filters
from core.metrics_actions import compute_action_center
from core.metrics_coverage import compute_coverage
from core.metrics_debug import compute_debug
from core.metrics_finance import compute_finance
from core.metrics_overview import compute_overview
from core.metrics_performance import compute_performance
from core.metrics_returns import compute_returns
from core.metrics_supply import compute_supply_health


app = FastAPI(title="AZ Dashboard API", version="0.1.0")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _filters_from_model(model: DashboardFiltersModel, *, available_weeks: list[int]) -> DashboardFilters:
    raw = model.model_dump()
    return normalize_filters(raw, available_weeks=available_weeks)

def _json(data: object) -> JSONResponse:
    """Return JSON with safe encoding for pandas/numpy objects."""

    def _safe_float(value: object) -> float | None:
        try:
            out = float(value)  # type: ignore[arg-type]
        except Exception:
            return None
        if math.isnan(out) or math.isinf(out):
            return None
        return out

    return JSONResponse(
        content=jsonable_encoder(
            data,
            custom_encoder={
                type(pd.NA): lambda _: None,
                np.integer: int,
                float: _safe_float,
                np.floating: _safe_float,
                np.bool_: bool,
                np.ndarray: lambda arr: arr.tolist(),
                pd.Timestamp: lambda ts: ts.isoformat(),
            },
        )
    )


@app.get("/meta/weeks")
def meta_weeks():
    try:
        data_ctx = load_dashboard_data()
        weeks = data_ctx.get("weeks", []) or []
        weeks = [int(w) for w in weeks if w is not None]
        return _json({"weeks": weeks})
    except Exception as exc:
        logger.exception("meta_weeks failed")
        return JSONResponse(status_code=500, content={"error": str(exc), "type": type(exc).__name__})


@app.get("/meta/categories")
def meta_categories():
    try:
        data_ctx = load_dashboard_data()
        dim_category: pd.DataFrame = data_ctx.get("dim_category", pd.DataFrame())
        if dim_category.empty or "major_category" not in dim_category.columns:
            return _json({"categories": []})
        cats = sorted([str(x) for x in dim_category["major_category"].dropna().unique().tolist()])
        return _json({"categories": cats})
    except Exception as exc:
        logger.exception("meta_categories failed")
        return JSONResponse(status_code=500, content={"error": str(exc), "type": type(exc).__name__})


@app.get("/meta/parts")
def meta_parts(sku_query: str = Query(default="")):
    try:
        data_ctx = load_dashboard_data()
        fact_sales: pd.DataFrame = data_ctx.get("fact_sales", pd.DataFrame())
        if fact_sales.empty or "part_number" not in fact_sales.columns:
            return _json({"parts": []})
        parts = fact_sales["part_number"].dropna().astype(str)
        q = (sku_query or "").strip().lower()
        if q:
            parts = parts[parts.str.lower().str.contains(q, na=False)]
        parts = sorted(parts.unique().tolist())[:500]
        return _json({"parts": parts})
    except Exception as exc:
        logger.exception("meta_parts failed")
        return JSONResponse(status_code=500, content={"error": str(exc), "type": type(exc).__name__})


@app.post("/overview")
def overview(filters: DashboardFiltersModel):
    try:
        data_ctx = load_dashboard_data()
        f = _filters_from_model(filters, available_weeks=data_ctx.get("weeks", []))
        ctx = prepare_context(f, data_ctx)
        return _json(compute_overview(f, ctx))
    except Exception as exc:
        logger.exception("overview failed")
        return JSONResponse(status_code=500, content={"error": str(exc), "type": type(exc).__name__})


@app.post("/performance")
def performance(
    filters: DashboardFiltersModel,
    metric: Literal["revenue", "units"] = Query(default="revenue"),
    view: Literal["product", "category"] = Query(default="product"),
):
    try:
        data_ctx = load_dashboard_data()
        f = _filters_from_model(filters, available_weeks=data_ctx.get("weeks", []))
        ctx = prepare_context(f, data_ctx)
        return _json(compute_performance(f, ctx, metric=metric, view=view))
    except Exception as exc:
        logger.exception("performance failed")
        return JSONResponse(status_code=500, content={"error": str(exc), "type": type(exc).__name__})


@app.post("/supply-health")
def supply_health(filters: DashboardFiltersModel):
    try:
        data_ctx = load_dashboard_data()
        f = _filters_from_model(filters, available_weeks=data_ctx.get("weeks", []))
        ctx = prepare_context(f, data_ctx)
        return _json(compute_supply_health(f, ctx))
    except Exception as exc:
        logger.exception("supply_health failed")
        return JSONResponse(status_code=500, content={"error": str(exc), "type": type(exc).__name__})


@app.post("/action-center")
def action_center(
    filters: DashboardFiltersModel,
    severity: str = Query(default="All"),
    q: str = Query(default=""),
    selected_sku: Optional[str] = Query(default=None),
):
    try:
        data_ctx = load_dashboard_data()
        f = _filters_from_model(filters, available_weeks=data_ctx.get("weeks", []))
        ctx = prepare_context(f, data_ctx)
        selected_sku = selected_sku or None
        return _json(compute_action_center(f, ctx, severity=severity, q=q, selected_sku=selected_sku))
    except Exception as exc:
        logger.exception("action_center failed")
        return JSONResponse(status_code=500, content={"error": str(exc), "type": type(exc).__name__})


@app.post("/coverage")
def coverage(filters: DashboardFiltersModel):
    try:
        data_ctx = load_dashboard_data()
        f = _filters_from_model(filters, available_weeks=data_ctx.get("weeks", []))
        ctx = prepare_context(f, data_ctx)
        return _json(compute_coverage(f, ctx))
    except Exception as exc:
        logger.exception("coverage failed")
        return JSONResponse(status_code=500, content={"error": str(exc), "type": type(exc).__name__})


@app.post("/returns")
def returns(filters: DashboardFiltersModel, volume_gate: int = Query(default=50)):
    try:
        data_ctx = load_dashboard_data()
        f = _filters_from_model(filters, available_weeks=data_ctx.get("weeks", []))
        ctx = prepare_context(f, data_ctx)
        return _json(compute_returns(f, ctx, volume_gate=volume_gate))
    except Exception as exc:
        logger.exception("returns failed")
        return JSONResponse(status_code=500, content={"error": str(exc), "type": type(exc).__name__})


@app.post("/finance")
def finance(filters: DashboardFiltersModel):
    try:
        data_ctx = load_dashboard_data()
        f = _filters_from_model(filters, available_weeks=data_ctx.get("weeks", []))
        ctx = prepare_context(f, data_ctx)
        return _json(compute_finance(f, ctx))
    except Exception as exc:
        logger.exception("finance failed")
        return JSONResponse(status_code=500, content={"error": str(exc), "type": type(exc).__name__})


@app.post("/debug")
def debug(filters: DashboardFiltersModel):
    try:
        data_ctx = load_dashboard_data()
        f = _filters_from_model(filters, available_weeks=data_ctx.get("weeks", []))
        ctx = prepare_context(f, data_ctx)
        return _json(compute_debug(f, ctx))
    except Exception as exc:
        logger.exception("debug failed")
        return JSONResponse(status_code=500, content={"error": str(exc), "type": type(exc).__name__})


@app.post("/export/{page}")
def export_page(page: str, filters: DashboardFiltersModel):
    data_ctx = load_dashboard_data()
    f = _filters_from_model(filters, available_weeks=data_ctx.get("weeks", []))
    ctx = prepare_context(f, data_ctx)

    export_df = None
    filename = f"{page}.csv"
    if page == "overview":
        export_df = ctx.get("exec_tyly_gt_filtered")
    elif page == "performance":
        export_df = ctx.get("filtered_sales")
    elif page in {"supply", "supply-health"}:
        export_df = ctx.get("filtered_cpfr")
        filename = "supply.csv"
    elif page == "returns":
        export_df = ctx.get("filtered_returns")
    elif page == "finance":
        export_df = ctx.get("filtered_billbacks")
    else:
        export_df = pd.DataFrame()

    if export_df is None or not hasattr(export_df, "to_csv"):
        export_df = pd.DataFrame()
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    return Response(content=csv_bytes, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})
