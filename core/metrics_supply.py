from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import altair as alt
import pandas as pd

from core.charts import to_vega_spec
from core.filters import DashboardFilters


def compute_supply_health(filters: DashboardFilters, ctx: Dict[str, Any]) -> Dict[str, Any]:
    cpfr: pd.DataFrame = ctx.get("filtered_cpfr", pd.DataFrame()).copy()
    redflags: pd.DataFrame = ctx.get("filtered_redflags", pd.DataFrame()).copy()
    outs: pd.DataFrame = ctx.get("filtered_outs", pd.DataFrame()).copy()

    service_kpis = {}
    charts = {}
    if not cpfr.empty:
        cpfr_base = cpfr.sort_values(["snapshot_year", "snapshot_week"]).drop_duplicates(
            subset=["fiscal_year", "fiscal_week"], keep="last"
        )
        current_fy = int(cpfr_base["fiscal_year"].max())
        current_fw = int(cpfr_base[cpfr_base["fiscal_year"] == current_fy]["fiscal_week"].max())
        cur = cpfr_base[(cpfr_base["fiscal_year"] == current_fy) & (cpfr_base["fiscal_week"] == current_fw)]
        cur_row = cur.iloc[0] if not cur.empty else None
        service_kpis = {
            "label": f"FY{current_fy} FW{current_fw}",
            "shipped_units": float(cur_row["shipped_units"]) if cur_row is not None and pd.notna(cur_row["shipped_units"]) else None,
            "fill_rate": float(cur_row["fill_rate"]) if cur_row is not None and pd.notna(cur_row["fill_rate"]) else None,
            "not_shipped_units": float(cur_row["not_shipped_units"]) if cur_row is not None and pd.notna(cur_row["not_shipped_units"]) else None,
        }

        years = sorted(cpfr["fiscal_year"].unique(), reverse=True)[:3]
        trend = cpfr[cpfr["fiscal_year"].isin(years)].sort_values(["fiscal_year", "fiscal_week"])
        shipped_chart = (
            alt.Chart(trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("fiscal_week:Q", title="Fiscal Week"),
                y=alt.Y("shipped_units:Q", title="Shipped Units", axis=alt.Axis(format="~s")),
                color=alt.Color("fiscal_year:N", title="Fiscal Year"),
                tooltip=["fiscal_year", "fiscal_week", alt.Tooltip("shipped_units:Q", format=",")],
            )
        )
        fill_chart = (
            alt.Chart(trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("fiscal_week:Q", title="Fiscal Week"),
                y=alt.Y("fill_rate:Q", title="Fill Rate", axis=alt.Axis(format=".0%")),
                color=alt.Color("fiscal_year:N", title="Fiscal Year"),
                tooltip=["fiscal_year", "fiscal_week", alt.Tooltip("fill_rate:Q", format=".2%")],
            )
        )
        charts = {"shipped_trend": to_vega_spec(shipped_chart), "fill_rate_trend": to_vega_spec(fill_chart)}

    exceptions = {"redflags": [], "outs": []}
    if not redflags.empty:
        rf = redflags.dropna(subset=["part_number"]).copy()
        for col in ["not_shipped_lfw", "not_shipped_l4w", "not_shipped_l52w"]:
            if col in rf.columns:
                rf[col] = pd.to_numeric(rf[col], errors="coerce")
        exceptions["redflags"] = rf.to_dict(orient="records")
    if not outs.empty:
        out_top = outs.sort_values("store_oos_exposure", ascending=False).head(filters.top_n)
        exceptions["outs"] = out_top.to_dict(orient="records")

    return {"filters": asdict(filters), "service_kpis": service_kpis, "charts": charts, "exceptions": exceptions}
