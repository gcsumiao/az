from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import altair as alt
import pandas as pd

from core.charts import to_vega_spec
from core.filters import DashboardFilters


def compute_returns(filters: DashboardFilters, ctx: Dict[str, Any], *, volume_gate: int = 50) -> Dict[str, Any]:
    df: pd.DataFrame = ctx.get("filtered_returns", pd.DataFrame()).copy()
    if df.empty:
        return {"filters": asdict(filters), "kpis": {}, "top": [], "charts": {}}

    volume_gate = max(0, int(volume_gate))
    latest_year = int(df["fiscal_year"].max())
    latest_week = int(df[df["fiscal_year"] == latest_year]["fiscal_week"].max())
    latest = df[(df["fiscal_year"] == latest_year) & (df["fiscal_week"] == latest_week)]
    latest = latest[latest["gross_units"] >= volume_gate]

    avg_damaged = float(latest["damaged_rate"].mean(skipna=True)) if not latest.empty else None
    avg_undamaged = float(latest["undamaged_rate"].mean(skipna=True)) if not latest.empty else None
    top_risk = None
    if not latest.empty:
        tr = latest.sort_values("damaged_rate", ascending=False).head(1)
        if not tr.empty:
            top_risk = {"part_number": str(tr.iloc[0]["part_number"]), "damaged_rate": float(tr.iloc[0]["damaged_rate"])}

    ret_top = df[df["gross_units"] >= volume_gate].sort_values("damaged_rate", ascending=False).head(filters.top_n).reset_index(drop=True)
    ret_top.insert(0, "rank", ret_top.index + 1)

    charts: Dict[str, Any] = {}
    gated = df[df["gross_units"] >= volume_gate].copy()
    if not gated.empty and {"fiscal_week", "damaged_rate", "undamaged_rate"}.issubset(gated.columns):
        trend = (
            gated.groupby("fiscal_week")[["damaged_rate", "undamaged_rate"]]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("fiscal_week")
        )
        long_df = trend.melt(id_vars="fiscal_week", value_vars=["damaged_rate", "undamaged_rate"], var_name="metric", value_name="rate")
        ret_hover = alt.selection_point(fields=["metric"], on="mouseover", empty="all")
        line = (
            alt.Chart(long_df)
            .mark_line(point={"filled": True})
            .encode(
                x=alt.X("fiscal_week:O", title="Fiscal Week", axis=alt.Axis(format="d", grid=False)),
                y=alt.Y("rate:Q", title="Return Rate", axis=alt.Axis(format=".1%", gridDash=[4, 4], domain=False, ticks=False)),
                color=alt.Color("metric:N", title="Metric"),
                opacity=alt.condition(ret_hover, alt.value(1), alt.value(0.2)),
                tooltip=[
                    alt.Tooltip("fiscal_week:Q", title="FW"),
                    alt.Tooltip("metric:N", title="Metric"),
                    alt.Tooltip("rate:Q", title="Rate", format=".2%"),
                ],
            )
            .add_params(ret_hover)
        )
        charts["return_rate_trend"] = to_vega_spec(line)

    return {
        "filters": asdict(filters),
        "snapshot": {"fiscal_year": latest_year, "fiscal_week": latest_week, "volume_gate": volume_gate},
        "kpis": {"avg_damaged_rate": avg_damaged, "avg_undamaged_rate": avg_undamaged, "top_risk": top_risk},
        "top": ret_top.to_dict(orient="records"),
        "charts": charts,
    }
