from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Literal, Optional

import altair as alt
import pandas as pd

from core.charts import to_vega_spec
from core.filters import DashboardFilters

Metric = Literal["revenue", "units"]
View = Literal["product", "category"]


def compute_performance(
    filters: DashboardFilters,
    ctx: Dict[str, Any],
    *,
    metric: Metric = "revenue",
    view: View = "product",
    focus: Optional[list[str]] = None,
) -> Dict[str, Any]:
    df: pd.DataFrame = ctx.get("filtered_sales", pd.DataFrame()).copy()
    metric_col = "revenue" if metric == "revenue" else "units"
    metric_axis_format = "$~s" if metric_col == "revenue" else "~s"
    other_axis_format = "~s" if metric_col == "revenue" else "$~s"

    if df.empty:
        return {"filters": asdict(filters), "metric": metric, "view": view, "top": [], "charts": {}, "options": []}

    if view == "product":
        base = df.dropna(subset=["part_number"]).copy()
        base = base[~base["part_number"].astype(str).str.contains("TOTAL", case=False, na=False)]
        if "major_category" not in base.columns:
            base["major_category"] = pd.NA
        top = (
            base.groupby(["part_number", "description"])
            .agg(
                revenue=("revenue", "sum"),
                units=("units", "sum"),
                major_category=(
                    "major_category",
                    lambda s: (s.dropna().astype(str).mode().iloc[0] if not s.dropna().empty else None),
                ),
            )
            .reset_index()
            .sort_values(metric_col, ascending=False)
        )
        top.insert(0, "rank", range(1, len(top) + 1))
        top = top.head(filters.top_n)
        trend_data = (
            base.groupby(["part_number", "fiscal_week"])[metric_col]
            .sum()
            .reset_index()
            .sort_values(["part_number", "fiscal_week"])
        )
        options = [str(x) for x in top["part_number"].astype(str).tolist()]
        focus = focus or options[: min(10, len(options))]
        chart_df = trend_data[trend_data["part_number"].astype(str).isin(set(focus))] if focus else trend_data
        # Trend Chart with Hover Interaction
        hover = alt.selection_point(fields=["part_number"], on="mouseover", empty="all")
        trend_chart = (
            alt.Chart(chart_df)
            .mark_line(point={"filled": True, "size": 60})
            .encode(
                x=alt.X("fiscal_week:O", title="Fiscal Week", axis=alt.Axis(tickMinStep=1, format="d", grid=False)),
                y=alt.Y(f"{metric_col}:Q", axis=alt.Axis(format=metric_axis_format, gridDash=[4, 4], domain=False, ticks=False)),
                color="part_number:N",
                opacity=alt.condition(hover, alt.value(1), alt.value(0.2)),
                tooltip=[
                    alt.Tooltip("fiscal_week", title="Week"),
                    alt.Tooltip("part_number", title="SKU"),
                    alt.Tooltip(metric_col, title=metric.capitalize(), format=metric_axis_format)
                ]
            )
            .add_params(hover)
        )

        cat_group = (
            df.dropna(subset=["major_category"])
            .groupby("major_category")[metric_col]
            .sum()
            .reset_index()
            .sort_values(metric_col, ascending=False)
        )
        
        # Breakdown Chart with Hover Interaction
        cat_hover = alt.selection_point(fields=["major_category"], on="mouseover", empty="all")
        breakdown = (
            alt.Chart(cat_group)
            .mark_bar()
            .encode(
                x=alt.X("major_category:N", title="Major Category", axis=alt.Axis(grid=False)),
                y=alt.Y(f"{metric_col}:Q", axis=alt.Axis(format=metric_axis_format, gridDash=[4, 4], domain=False, ticks=False)),
                color="major_category:N",
                opacity=alt.condition(cat_hover, alt.value(1), alt.value(0.6)),
                tooltip=[
                    alt.Tooltip("major_category", title="Category"),
                    alt.Tooltip(metric_col, title=metric.capitalize(), format=metric_axis_format)
                ]
            )
            .add_params(cat_hover)
        )
        return {
            "filters": asdict(filters),
            "metric": metric,
            "view": view,
            "top": top.to_dict(orient="records"),
            "options": options,
            "charts": {"trend": to_vega_spec(trend_chart), "breakdown": to_vega_spec(breakdown)},
        }

    # category view
    base = df.dropna(subset=["major_category"]).copy()
    top = (
        base.groupby("major_category")
        .agg(revenue=("revenue", "sum"), units=("units", "sum"))
        .reset_index()
        .sort_values(metric_col, ascending=False)
    )
    top.insert(0, "rank", range(1, len(top) + 1))
    top = top.head(filters.top_n)

    trend_data = (
        base.groupby(["major_category", "fiscal_week"])[metric_col]
        .sum()
        .reset_index()
        .sort_values(["major_category", "fiscal_week"])
    )
    options = [str(x) for x in top["major_category"].astype(str).tolist()]
    focus = focus or options[: min(3, len(options))]
    chart_df = trend_data[trend_data["major_category"].astype(str).isin(set(focus))] if focus else trend_data
    
    # Category Trend Chart with Hover Interaction
    cat_trend_hover = alt.selection_point(fields=["major_category"], on="mouseover", empty="all")
    trend_chart = (
        alt.Chart(chart_df)
        .mark_line(point={"filled": True, "size": 60})
        .encode(
            x=alt.X("fiscal_week:O", title="Fiscal Week", axis=alt.Axis(tickMinStep=1, format="d", grid=False)),
            y=alt.Y(f"{metric_col}:Q", axis=alt.Axis(format=metric_axis_format, gridDash=[4, 4], domain=False, ticks=False)),
            color="major_category:N",
            opacity=alt.condition(cat_trend_hover, alt.value(1), alt.value(0.2)),
            tooltip=[
                alt.Tooltip("fiscal_week", title="Week"),
                alt.Tooltip("major_category", title="Category"),
                alt.Tooltip(metric_col, title=metric.capitalize(), format=metric_axis_format)
            ]
        )
        .add_params(cat_trend_hover)
    )
    prod_group = (
        df.dropna(subset=["part_number"])
        .groupby(["part_number", "description"])
        .agg(revenue=("revenue", "sum"), units=("units", "sum"))
        .reset_index()
        .sort_values(metric_col, ascending=False)
    ).head(filters.top_n)
    return {
        "filters": asdict(filters),
        "metric": metric,
        "view": view,
        "top": top.to_dict(orient="records"),
        "options": options,
        "product_detail": prod_group.to_dict(orient="records"),
        "charts": {"trend": to_vega_spec(trend_chart)},
    }
