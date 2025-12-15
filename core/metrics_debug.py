from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import pandas as pd

from core.filters import DashboardFilters


def compute_debug(filters: DashboardFilters, ctx: Dict[str, Any]) -> Dict[str, Any]:
    fact_sales: pd.DataFrame = ctx.get("exec_sales", pd.DataFrame()).copy()
    payload = {
        "filters": asdict(filters),
        "row_counts": {
            "sales_rows": int(len(fact_sales)),
            "forecast_rows": int(len(ctx.get("forecast_totals", pd.DataFrame()))),
            "orders_rows": int(len(ctx.get("order_totals", pd.DataFrame()))),
            "cpfr_rows": int(len(ctx.get("cpfr", pd.DataFrame()))),
            "outs_rows": int(len(ctx.get("outs", pd.DataFrame()))),
            "inventory_rows": int(len(ctx.get("inventory", pd.DataFrame()))),
            "returns_rows": int(len(ctx.get("returns", pd.DataFrame()))),
            "billbacks_rows": int(len(ctx.get("billbacks", pd.DataFrame()))),
        },
        "cleaning_checks": {
            "sales_rows_removed_invalid_sku": int(ctx.get("dq_removed_rows", 0) or 0),
            "bad_part_tokens_remaining": int(ctx.get("dq_bad_tokens_remaining", 0) or 0),
        },
        "category_dim_sample": [],
        "week_coverage": [],
        "null_sku_counts": {},
        "unmapped_top": [],
        "unmapped_details": [],
    }

    dim_category: pd.DataFrame = ctx.get("dim_category", pd.DataFrame()).copy()
    if not dim_category.empty:
        payload["category_dim_sample"] = dim_category.head(3).to_dict(orient="records")

    if not fact_sales.empty and "fiscal_year" in fact_sales.columns and "fiscal_week" in fact_sales.columns:
        week_cov = (
            fact_sales.groupby("fiscal_year")["fiscal_week"]
            .agg(["min", "max", "nunique"])
            .reset_index()
            .rename(columns={"nunique": "weeks_present"})
        )
        payload["week_coverage"] = week_cov.to_dict(orient="records")
        payload["null_sku_counts"] = {"null_part_number_rows": int(fact_sales["part_number"].isna().sum()) if "part_number" in fact_sales.columns else 0}
        if "major_category" in fact_sales.columns:
            unmapped = fact_sales[fact_sales["major_category"] == "Unmapped"]
            if not unmapped.empty:
                unmapped_top = (
                    unmapped["part_number"].value_counts().head(20).reset_index(name="count").rename(columns={"index": "part_number"})
                )
                payload["unmapped_top"] = unmapped_top.to_dict(orient="records")
                unmapped_details = unmapped[["part_number", "item_id", "description"]].drop_duplicates()
                payload["unmapped_details"] = unmapped_details.to_dict(orient="records")
    return payload

