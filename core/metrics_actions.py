from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import altair as alt
import pandas as pd

from core.charts import to_vega_spec
from core.filters import DashboardFilters


def compute_action_center(
    filters: DashboardFilters,
    ctx: Dict[str, Any],
    *,
    severity: str = "All",
    q: str = "",
    selected_sku: Optional[str] = None,
) -> Dict[str, Any]:
    filtered_outs: pd.DataFrame = ctx.get("filtered_outs", pd.DataFrame()).copy()
    filtered_redflags: pd.DataFrame = ctx.get("filtered_redflags", pd.DataFrame()).copy()
    filtered_sales: pd.DataFrame = ctx.get("filtered_sales", pd.DataFrame()).copy()

    action_rows = []
    if not filtered_outs.empty:
        outs_summary = (
            filtered_outs.groupby("part_number")
            .agg(store_oos_exposure=("store_oos_exposure", "sum"))
            .reset_index()
        )
        outs_summary["severity"] = outs_summary["store_oos_exposure"].apply(
            lambda v: "High" if float(v) >= filters.thresholds.outs_exposure else "Medium"
        )
        outs_summary["source"] = "OOS"
        action_rows.append(outs_summary)
    if not filtered_redflags.empty:
        rf_summary = (
            filtered_redflags.groupby("part_number")[["not_shipped_lfw", "not_shipped_l4w", "not_shipped_l52w"]]
            .max()
            .reset_index()
        )
        rf_summary["store_oos_exposure"] = rf_summary["not_shipped_lfw"].fillna(0)
        rf_summary["severity"] = "High"
        rf_summary["source"] = "Fill Rate Red Flags"
        rf_summary = rf_summary[["part_number", "store_oos_exposure", "severity", "source"]]
        action_rows.append(rf_summary)

    action_table = (
        pd.concat(action_rows, ignore_index=True)
        if action_rows
        else pd.DataFrame(columns=["part_number", "store_oos_exposure", "severity", "source"])
    )
    if not filtered_sales.empty and not action_table.empty:
        rev_map = (
            filtered_sales.groupby("part_number")["revenue"]
            .sum()
            .reset_index()
            .rename(columns={"revenue": "revenue_impact"})
        )
        action_table = action_table.merge(rev_map, on="part_number", how="left")

    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    action_table["severity_rank"] = action_table["severity"].map(severity_order).fillna(3)

    total_actions = int(action_table["part_number"].nunique()) if not action_table.empty else 0
    high_count = int(action_table[action_table["severity"].eq("High")]["part_number"].nunique()) if not action_table.empty else 0
    rev_risk = float(action_table["revenue_impact"].sum()) if "revenue_impact" in action_table.columns else 0.0

    table = action_table.copy()
    if severity != "All":
        table = table[table["severity"] == severity]
    query = (q or "").strip()
    if query:
        table = table[table["part_number"].astype(str).str.contains(query, case=False, na=False)]
    table = table.sort_values(["severity_rank", "revenue_impact"], ascending=[True, False])

    sku_options = sorted(action_table["part_number"].astype(str).unique()) if not action_table.empty else []
    if selected_sku is None and sku_options:
        if "revenue_impact" in action_table.columns:
            ranked = action_table.dropna(subset=["revenue_impact"]).sort_values("revenue_impact", ascending=False)
            if not ranked.empty:
                selected_sku = str(ranked["part_number"].iloc[0])
            else:
                selected_sku = sku_options[0]
        else:
            selected_sku = sku_options[0]

    drill = {"sku": selected_sku, "chart": None}
    if selected_sku and not filtered_sales.empty:
        sku_sales = filtered_sales[filtered_sales["part_number"].astype(str) == str(selected_sku)]
        if not sku_sales.empty:
            trend = sku_sales.groupby("fiscal_week")[["units", "revenue"]].sum().reset_index().sort_values("fiscal_week")
            long_df = trend.melt(id_vars="fiscal_week", value_vars=["units", "revenue"], var_name="metric", value_name="value")
            base = alt.Chart(long_df).encode(x=alt.X("fiscal_week:O", title="Fiscal Week", axis=alt.Axis(format="d")))

            units_line = (
                base.transform_filter(alt.datum.metric == "units")
                .mark_line(point=True, color="#2563eb")
                .encode(
                    y=alt.Y("value:Q", title="Units", axis=alt.Axis(format="~s")),
                    tooltip=["fiscal_week", alt.Tooltip("value:Q", title="Units", format=",")],
                )
            )
            revenue_line = (
                base.transform_filter(alt.datum.metric == "revenue")
                .mark_line(point=True, color="#16a34a")
                .encode(
                    y=alt.Y("value:Q", title="Revenue", axis=alt.Axis(format="$~s", orient="right")),
                    tooltip=["fiscal_week", alt.Tooltip("value:Q", title="Revenue", format="$,.0f")],
                )
            )
            chart = alt.layer(units_line, revenue_line).resolve_scale(y="independent")
            drill["chart"] = to_vega_spec(chart)

    return {
        "filters": asdict(filters),
        "kpis": {"skus_needing_action": total_actions, "high_severity": high_count, "revenue_at_risk": rev_risk},
        "severity": severity,
        "q": query,
        "table": table.drop(columns=["severity_rank"], errors="ignore").to_dict(orient="records"),
        "sku_options": sku_options,
        "drilldown": drill,
    }
