from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import altair as alt
import pandas as pd

from core.charts import to_vega_spec
from core.data import compute_margin, normalize_fee_code
from core.filters import DashboardFilters


def compute_finance(filters: DashboardFilters, ctx: Dict[str, Any]) -> Dict[str, Any]:
    sales: pd.DataFrame = ctx.get("filtered_sales", pd.DataFrame()).copy()
    cost: pd.DataFrame = ctx.get("filtered_cost", pd.DataFrame()).copy()
    billbacks: pd.DataFrame = ctx.get("filtered_billbacks", pd.DataFrame()).copy()
    dim_billback_reason: pd.DataFrame = ctx.get("dim_billback_reason", pd.DataFrame()).copy()

    margin = compute_margin(sales, cost, filters.selected_weeks)
    margin_summary = {}
    top_margin = []
    if not margin.empty:
        summary = margin.agg({"revenue": "sum", "cogs": "sum", "gross_margin": "sum"})
        gm_pct = float(summary["gross_margin"] / summary["revenue"]) if float(summary["revenue"] or 0) else None
        margin_summary = {
            "revenue": float(summary["revenue"]),
            "cogs": float(summary["cogs"]),
            "gross_margin": float(summary["gross_margin"]),
            "gross_margin_pct": gm_pct,
            "top_gm_product": None,
            "top_gm_category": None,
        }
        
        # Top GM Product
        top_prod_df = margin.groupby("part_number")["gross_margin"].sum().reset_index()
        if not top_prod_df.empty: 
            best_prod = top_prod_df.sort_values("gross_margin", ascending=False).iloc[0]
            # Try to get product name if available in sales data context, otherwise use part_number
            # A simple lookup if name exists in margin df (it might not). 
            # Assuming part_number is sufficient for now, or the UI handles ID lookup.
            margin_summary["top_gm_product"] = {
                "name": str(best_prod["part_number"]), 
                "value": float(best_prod["gross_margin"])
            }

        # Top GM Category
        if "category" in margin.columns:
            top_cat_df = margin.groupby("category")["gross_margin"].sum().reset_index()
            if not top_cat_df.empty:
                best_cat = top_cat_df.sort_values("gross_margin", ascending=False).iloc[0]
                margin_summary["top_gm_category"] = {
                    "name": str(best_cat["category"]), 
                    "value": float(best_cat["gross_margin"])
                }
        
        top_margin = (
            margin.sort_values("gross_margin", ascending=False)
            .head(filters.top_n)
            .reset_index(drop=True)
        )
        top_margin.insert(0, "rank", top_margin.index + 1)

    billback_chart = None
    if not billbacks.empty:
        mapped = billbacks.copy()
        if not dim_billback_reason.empty and "all_codes_norm" in dim_billback_reason.columns:
            dim_lookup = (
                dim_billback_reason.explode("all_codes_norm")[["all_codes_norm", "bucket", "direction", "title"]]
                .rename(columns={"all_codes_norm": "code_norm"})
            )
            mapped["code_norm"] = mapped["type_code_norm"].apply(normalize_fee_code)
            mapped = mapped.merge(dim_lookup, on="code_norm", how="left")
        if "bucket" not in mapped.columns:
            mapped["bucket"] = "Unmapped"
        weekly = (
            mapped.assign(invoice_week_start=lambda d: d["invoice_date"] - pd.to_timedelta(d["invoice_date"].dt.weekday, unit="D"))
            .groupby(["invoice_week_start", "bucket"])["billback_amount"]
            .sum()
            .reset_index()
            .sort_values("invoice_week_start")
        )
        if not weekly.empty:
            bar = (
                alt.Chart(weekly)
                .mark_bar()
                .encode(
                    x=alt.X("invoice_week_start:T", title="Invoice Week"),
                    y=alt.Y("billback_amount:Q", stack="zero", axis=alt.Axis(format="$,.0f")),
                    color="bucket:N",
                    tooltip=["invoice_week_start", "bucket", alt.Tooltip("billback_amount:Q", format="$,.0f")],
                )
            )
            billback_chart = to_vega_spec(bar)

    return {
        "filters": asdict(filters),
        "margin_summary": margin_summary,
        "top_margin": top_margin.to_dict(orient="records") if hasattr(top_margin, "to_dict") else top_margin,
        "billbacks_chart": billback_chart,
    }

