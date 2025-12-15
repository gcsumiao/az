from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import altair as alt
import pandas as pd

from core.charts import to_vega_spec
from core.filters import DashboardFilters


def compute_coverage(filters: DashboardFilters, ctx: Dict[str, Any]) -> Dict[str, Any]:
    # Use full history totals, not just filtered ones, to build the diagonal view correctly
    forecast: pd.DataFrame = ctx.get("forecast_totals", pd.DataFrame()).copy()
    orders: pd.DataFrame = ctx.get("order_totals", pd.DataFrame()).copy()
    forecast_actual: pd.DataFrame = ctx.get("forecast_actual_src", pd.DataFrame()).copy()

    payload: Dict[str, Any] = {
        "filters": asdict(filters),
        "snapshot": {},
        "kpis": {},
        "charts": {},
        "table": [],
        "dq_warning": None,
    }

    if forecast.empty and orders.empty and forecast_actual.empty:
        return payload

    # 1. Prepare Diagonal Data (Target == Snapshot)
    # This represents the "Final" Forecast and "Final" Order for each week.
    if not forecast.empty and {"snapshot_year", "snapshot_week", "target_year", "target_week"}.issubset(forecast.columns):
        forecast = forecast[
            (forecast["snapshot_year"] == forecast["target_year"]) & 
            (forecast["snapshot_week"] == forecast["target_week"])
        ]
    
    if not orders.empty and {"snapshot_year", "snapshot_week", "target_year", "target_week"}.issubset(orders.columns):
        orders = orders[
            (orders["snapshot_year"] == orders["target_year"]) & 
            (orders["snapshot_week"] == orders["target_week"])
        ]

    # Group by target week to handle any potential dupes (though diagonal implies unique per target)
    if not forecast.empty:
        forecast = (
            forecast.groupby(["target_year", "target_week"], dropna=False)["forecast_units"]
            .sum()
            .reset_index()
        )
        forecast["target_year"] = forecast["target_year"].astype("Int64")
        forecast["target_week"] = forecast["target_week"].astype("Int64")

    if not orders.empty:
        orders = (
            orders.groupby(["target_year", "target_week"], dropna=False)["order_units"]
            .sum()
            .reset_index()
        )
        orders["target_year"] = orders["target_year"].astype("Int64")
        orders["target_week"] = orders["target_week"].astype("Int64")

    # 2. Merge - CRITICAL: Do NOT fillna(0) for orders
    # We want order_units to be NaN if missing, so we can detect DQ issues
    if not forecast.empty:
        cov_table = forecast.merge(orders, on=["target_year", "target_week"], how="left")
    elif not orders.empty:
        cov_table = orders.rename(columns={"order_units": "order_units"}).copy()
        cov_table["forecast_units"] = pd.NA
    else:
        cov_table = pd.DataFrame()

    if not cov_table.empty:
        # Calculate Coverage & Gap
        # coverage = order / forecast (if forecast > 0)
        # Handle division by zero or missing orders
        def calc_coverage(row):
            f = row["forecast_units"]
            o = row["order_units"]
            if pd.isna(o): return pd.NA
            if pd.isna(f) or f == 0: return pd.NA
            return o / f

        cov_table["coverage"] = cov_table.apply(calc_coverage, axis=1)
        
        # Gap = Forecast - Order (only if both exist? User said "If order_units is NaN, display N/A")
        cov_table["gap_units"] = cov_table["forecast_units"] - cov_table["order_units"]

        cov_table = cov_table.sort_values(["target_year", "target_week"])

        # 3. Snapshot Context & DQ Warning
        # Identify the latest "diagonal" week we have data for
        if not cov_table.empty:
            latest_row = cov_table.iloc[-1]
            latest_year = int(latest_row["target_year"]) if pd.notna(latest_row["target_year"]) else None
            latest_week = int(latest_row["target_week"]) if pd.notna(latest_row["target_week"]) else None
            
            payload["snapshot"] = {"snapshot_year": latest_year, "snapshot_week": latest_week}
            
            # Check for DQ Warning on the latest snapshot
            # "If forecast_units_gt > 0 AND (order_units_gt is NaN)... Display a warning"
            f_latest = latest_row["forecast_units"]
            o_latest = latest_row["order_units"]
            
            if pd.notna(f_latest) and f_latest > 0 and pd.isna(o_latest):
                payload["dq_warning"] = (
                    f"Data Quality Warning: Order Projection (Grand Total) is missing for FW{latest_week}. "
                    "Coverage cannot be computed (showing N/A). Please verify the forecast sheet Order Projection block."
                )

            # KPIs
            cov_vals = cov_table["coverage"].dropna()
            payload["kpis"]["avg_coverage"] = float(cov_vals.mean()) if not cov_vals.empty else None
            
            # Latest coverage: Use the computed coverage from the latest row (might be NaN)
            if pd.notna(latest_row["coverage"]):
                payload["kpis"]["latest_coverage"] = float(latest_row["coverage"])
            else:
                payload["kpis"]["latest_coverage"] = None

            # 4. Feature: Bar Chart with Threshold Coloring
            # Rules: Green if coverage >= 1.0, Red if coverage < 1.0
            # Add a reference line at y=1
            
            base = alt.Chart(cov_table).encode(
                 x=alt.X("target_week:O", title="Target Fiscal Week", axis=alt.Axis(format="d", grid=False))
            )

            bars = base.mark_bar().encode(
                y=alt.Y("coverage:Q", title="Coverage", axis=alt.Axis(format=".0%", gridDash=[4, 4], domain=False, ticks=False)),
                color=alt.condition(
                    "datum.coverage < 1",
                    alt.value("#ef4444"),  # Red
                    alt.value("#10b981")   # Green
                ),
                tooltip=[
                    alt.Tooltip("target_year:Q", title="Target FY"),
                    alt.Tooltip("target_week:Q", title="Target FW"),
                    alt.Tooltip("forecast_units:Q", title="Forecast Units", format=","),
                    alt.Tooltip("order_units:Q", title="Order Units", format=","),
                    alt.Tooltip("coverage:Q", title="Coverage", format=".1%"),
                ]
            )
            
            rule = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(y='y')

            payload["charts"]["coverage"] = to_vega_spec(bars + rule)

            # Volume Chart (Forecast vs Orders) - Keep as line, but handle NaNs (breaks in line are fine/expected)
            vol_hover = alt.selection_point(fields=["series"], on="mouseover", empty="all")
            volume_chart = (
                alt.Chart(cov_table)
                .transform_fold(["forecast_units", "order_units"], as_=["series", "units"])
                .mark_line(point={"filled": True})
                .encode(
                    x=alt.X("target_week:O", title="Target Fiscal Week", axis=alt.Axis(format="d", grid=False)),
                    y=alt.Y("units:Q", title="Units", axis=alt.Axis(format="~s", gridDash=[4, 4], domain=False, ticks=False)),
                    color=alt.Color("series:N", title="Series"),
                    opacity=alt.condition(vol_hover, alt.value(1), alt.value(0.2)),
                    tooltip=[
                        alt.Tooltip("target_year:Q", title="Target FY"),
                        alt.Tooltip("target_week:Q", title="Target FW"),
                        alt.Tooltip("series:N", title="Series"),
                        alt.Tooltip("units:Q", title="Units", format=","),
                    ],
                )
                .add_params(vol_hover)
            )
            payload["charts"]["forecast_vs_orders"] = to_vega_spec(volume_chart)
            payload["table"] = cov_table.where(pd.notnull(cov_table), None).to_dict(orient="records")

    # Forecast error (Actuals vs Forecast) - existing logic seems OK, relies on forecast_actual
    if not forecast_actual.empty and {"fiscal_week", "units", "forecast_units"}.issubset(forecast_actual.columns):
        fa = forecast_actual.copy()
        fa["error_pct"] = (fa["forecast_units"] - fa["units"]) / fa["units"].replace({0: pd.NA})
        fa["abs_error_pct"] = fa["error_pct"].abs()
        payload["kpis"]["mape"] = float(fa["abs_error_pct"].dropna().mean()) if not fa["abs_error_pct"].dropna().empty else None
        
        error_chart = (
            alt.Chart(fa.sort_values("fiscal_week"))
            .mark_line(point=True)
            .encode(
                x=alt.X("fiscal_week:O", title="Fiscal Week", axis=alt.Axis(format="d")),
                y=alt.Y("abs_error_pct:Q", title="Forecast Error % (abs)", axis=alt.Axis(format=".0%")),
                tooltip=[
                    alt.Tooltip("fiscal_week:Q", title="FW"),
                    alt.Tooltip("forecast_units:Q", title="Forecast Units", format=","),
                    alt.Tooltip("units:Q", title="Actual Units", format=","),
                    alt.Tooltip("abs_error_pct:Q", title="Abs Error %", format=".1%"),
                ],
            )
        )
        payload["charts"]["forecast_error"] = to_vega_spec(error_chart)

    return payload

