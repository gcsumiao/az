from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import altair as alt
import pandas as pd

from core.charts import to_vega_spec
from core.data import (
    compute_comparable_yoy,
    compute_margin,
    compute_wow_pct,
    get_snapshot_context,
    round_half_up,
)
from core.filters import DashboardFilters


def _metric_value(value: Optional[float]) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _as_int(value: Any) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        return None


def compute_overview(filters: DashboardFilters, ctx: Dict[str, Any]) -> Dict[str, Any]:
    gt_df: pd.DataFrame = ctx.get("exec_tyly_gt_filtered", pd.DataFrame())
    sales_df: pd.DataFrame = ctx.get("exec_sales_filtered", pd.DataFrame())
    filtered_sales: pd.DataFrame = ctx.get("filtered_sales", pd.DataFrame())
    filtered_cost: pd.DataFrame = ctx.get("filtered_cost", pd.DataFrame())
    alerts = ctx.get("alerts", []) or []

    latest_year, snapshot_week, prev_week = get_snapshot_context(gt_df)
    
    # Calculate Range Revenue (Grand Total) early for GM% denominator
    range_revenue = float(gt_df[gt_df["fiscal_year"] == latest_year]["fw_revenue_total"].sum()) if latest_year is not None and not gt_df.empty else 0.0

    desc_map = None
    if not sales_df.empty and {"part_number", "description"}.issubset(sales_df.columns):
        tmp = sales_df.dropna(subset=["part_number"]).copy()
        tmp["part_number"] = tmp["part_number"].astype(str)
        tmp["description"] = tmp["description"].astype("string")
        desc_map = (
            tmp.groupby("part_number")["description"]
            .apply(lambda s: (s.dropna().astype(str).mode().iloc[0] if not s.dropna().empty else None))
            .reset_index()
        )

    gm_summary: Dict[str, Any] = {
        "gross_margin": None,
        "gross_margin_pct": None,
        "top_product": None,
        "top_category": None,
    }
    margin = compute_margin(filtered_sales, filtered_cost, filters.selected_weeks)
    if not margin.empty:
        summary = margin.agg({"revenue": "sum", "gross_margin": "sum"})
        total_rev = float(summary.get("revenue", 0) or 0)
        total_gm = float(summary.get("gross_margin", 0) or 0)
        
        # User Logic: GM% = Total GM / FY Range Revenue (Grand Total)
        # Fallback to total_rev (SKU sum) if range_revenue (Grand Total) is missing or zero
        denominator = range_revenue if range_revenue > 0 else total_rev
        gm_pct = (total_gm / denominator) if denominator else None

        top_prod = None
        prod_group = (
            margin.groupby(["part_number", "description"])["gross_margin"]
            .sum()
            .reset_index()
            .sort_values("gross_margin", ascending=False)
        )
        if not prod_group.empty:
            r = prod_group.iloc[0]
            top_prod = {
                "part_number": str(r["part_number"]),
                "description": (str(r["description"]) if pd.notna(r["description"]) else None),
                "gross_margin": float(r["gross_margin"]) if pd.notna(r["gross_margin"]) else None,
            }

        top_cat = None
        if not filtered_sales.empty and "major_category" in filtered_sales.columns and "part_number" in filtered_sales.columns:
            part_cat = (
                filtered_sales.dropna(subset=["part_number"])
                .assign(part_number=lambda d: d["part_number"].astype(str))
                .groupby("part_number")["major_category"]
                .apply(lambda s: (s.dropna().astype(str).mode().iloc[0] if not s.dropna().empty else None))
                .reset_index()
            )
            margin_cat = margin.assign(part_number=lambda d: d["part_number"].astype(str)).merge(part_cat, on="part_number", how="left")
            cat_group = (
                margin_cat.dropna(subset=["major_category"])
                .groupby("major_category")["gross_margin"]
                .sum()
                .reset_index()
                .sort_values("gross_margin", ascending=False)
            )
            if not cat_group.empty:
                c = cat_group.iloc[0]
                top_cat = {
                    "major_category": str(c["major_category"]),
                    "gross_margin": float(c["gross_margin"]) if pd.notna(c["gross_margin"]) else None,
                }

        gm_summary = {
            "gross_margin": total_gm,
            "gross_margin_pct": gm_pct,
            "top_product": top_prod,
            "top_category": top_cat,
        }

    snap = (
        gt_df[(gt_df["fiscal_year"] == latest_year) & (gt_df["fiscal_week"] == snapshot_week)]
        if snapshot_week is not None and latest_year is not None
        else pd.DataFrame()
    )
    prev = (
        gt_df[(gt_df["fiscal_year"] == latest_year) & (gt_df["fiscal_week"] == prev_week)]
        if prev_week is not None and latest_year is not None
        else pd.DataFrame()
    )

    units = _metric_value(snap["fw_units_total"].iloc[0]) if not snap.empty and "fw_units_total" in snap.columns else None
    revenue = _metric_value(snap["fw_revenue_total"].iloc[0]) if not snap.empty and "fw_revenue_total" in snap.columns else None
    prev_units = _metric_value(prev["fw_units_total"].iloc[0]) if not prev.empty and "fw_units_total" in prev.columns else None
    prev_revenue = _metric_value(prev["fw_revenue_total"].iloc[0]) if not prev.empty and "fw_revenue_total" in prev.columns else None

    units_wow = (units - prev_units) / prev_units if units is not None and prev_units not in (None, 0) else None
    revenue_wow = (revenue - prev_revenue) / prev_revenue if revenue is not None and prev_revenue not in (None, 0) else None
    asp = revenue / units if revenue is not None and units not in (None, 0) else None
    
    range_units = float(gt_df[gt_df["fiscal_year"] == latest_year]["fw_units_total"].sum()) if latest_year is not None and not gt_df.empty else None

    heroes: Dict[str, Any] = {"product": {}, "category": {}}
    if snapshot_week is not None and not sales_df.empty:
        for dim, key in [("part_number", "product"), ("major_category", "category")]:
            yoy_df = compute_comparable_yoy(sales_df, snapshot_week, dim, filters.thresholds.min_ly_rev_floor)
            wow_df = compute_wow_pct(sales_df, snapshot_week, prev_week, dim)
            if dim == "part_number" and desc_map is not None and not desc_map.empty:
                yoy_df = yoy_df.merge(desc_map, on="part_number", how="left")
                if not wow_df.empty and "part_number" in wow_df.columns:
                    wow_df = wow_df.merge(desc_map, on="part_number", how="left")
            hero = yoy_df.sort_values("ty_rev", ascending=False).head(1)
            declining = yoy_df.sort_values("ty_rev", ascending=True).head(1)
            rising = wow_df.sort_values("wow_rev_pct", ascending=False).head(1) if not wow_df.empty else pd.DataFrame()
            heroes[key] = {
                "hero": hero.to_dict(orient="records")[0] if not hero.empty else None,
                "rising": rising.to_dict(orient="records")[0] if not rising.empty else None,
                "declining": declining.to_dict(orient="records")[0] if not declining.empty else None,
            }

    charts: Dict[str, Any] = {}
    if not gt_df.empty:
        trend = gt_df.sort_values(["fiscal_year", "fiscal_week"]).copy()
        trend["fw_revenue_total_disp"] = trend["fw_revenue_total"].apply(round_half_up)
        # Revenue Trend with Hover
        rev_hover = alt.selection_point(fields=["fiscal_year"], on="mouseover", empty="all")
        line_rev = (
            alt.Chart(trend)
            .mark_line(point={"filled": True, "size": 60})
            .encode(
                x=alt.X("fiscal_week:O", title="Fiscal Week", axis=alt.Axis(format="d", grid=False)),
                y=alt.Y("fw_revenue_total_disp:Q", axis=alt.Axis(format="$~s", gridDash=[4, 4], domain=False, ticks=False)),
                color="fiscal_year:N",
                opacity=alt.condition(rev_hover, alt.value(1), alt.value(0.2)),
                tooltip=["fiscal_year", "fiscal_week", alt.Tooltip("fw_revenue_total_disp:Q", format="$,.0f")],
            )
            .add_params(rev_hover)
            .properties(height=260)
        )

        # Units Trend with Hover
        units_hover = alt.selection_point(fields=["fiscal_year"], on="mouseover", empty="all")
        line_units = (
            alt.Chart(trend)
            .mark_line(point={"filled": True, "size": 60})
            .encode(
                x=alt.X("fiscal_week:O", title="Fiscal Week", axis=alt.Axis(format="d", grid=False)),
                y=alt.Y("fw_units_total:Q", axis=alt.Axis(format="~s", gridDash=[4, 4], domain=False, ticks=False)),
                color="fiscal_year:N",
                opacity=alt.condition(units_hover, alt.value(1), alt.value(0.2)),
                tooltip=["fiscal_year", "fiscal_week", alt.Tooltip("fw_units_total:Q", format=",")],
            )
            .add_params(units_hover)
            .properties(height=260)
        )
        charts = {"revenue_trend": to_vega_spec(line_rev), "units_trend": to_vega_spec(line_units)}

    return {
        "filters": asdict(filters),
        "snapshot": {"fiscal_year": _as_int(latest_year), "snapshot_week": _as_int(snapshot_week), "prev_week": _as_int(prev_week)},
        "kpis": {
            "units": units,
            "revenue": revenue,
            "asp": asp,
            "units_wow": units_wow,
            "revenue_wow": revenue_wow,
        },
        "gm": gm_summary,
        "range_totals": {"units": range_units, "revenue": range_revenue},
        "heroes": heroes,
        "charts": charts,
        "alerts": alerts,
    }
