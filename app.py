import altair as alt
import pandas as pd
import streamlit as st
from contextlib import contextmanager
from typing import Dict, List, Optional

from python import dashboard_core as dc
from python.dashboard_core import *  # noqa: F401,F403

alt.data_transformers.disable_max_rows()
# ---------- UI / layout helpers ----------
def inject_base_styles():
    if st.session_state.get("_base_css_injected"):
        return
    st.markdown(
        """
        <style>
        .app-top-bar {padding: 6px 0 4px;border-bottom: 1px solid #e5e7eb;margin-bottom: 10px;}
        .app-top-bar .breadcrumb {color: #6b7280;font-size: 0.9rem;margin-bottom: 2px;}
        .app-top-bar .page-title {font-size: 1.4rem;font-weight: 700;color: #111827;}
        .card {border: 1px solid #e5e7eb;border-radius: 12px;padding: 16px;background: #ffffff;
               box-shadow: 0 1px 2px rgba(0,0,0,0.04); margin-bottom: 12px;}
        .card-header {display: flex;justify-content: space-between;align-items: center;margin-bottom: 8px;}
        .card-title {font-weight: 600;font-size: 1.0rem;color: #111827;}
        .card-actions {font-size: 0.9rem;color: #2563eb;}
        .chip-row {display: flex;flex-wrap: wrap;gap: 6px;margin-top: 6px;}
        .chip {background: #f3f4f6;border: 1px solid #e5e7eb;border-radius: 14px;padding: 4px 10px;font-size: 0.85rem;color: #374151;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_base_css_injected"] = True


@contextmanager
def card(title: str, actions: Optional[str] = None):
    container = st.container()
    container.markdown(
        f"""
        <div class="card">
          <div class="card-header">
            <div class="card-title">{title}</div>
            <div class="card-actions">{actions or ""}</div>
          </div>
        """,
        unsafe_allow_html=True,
    )
    body = container.container()
    with body:
        yield body
    container.markdown("</div>", unsafe_allow_html=True)


def format_filter_summary(selected_weeks: List[int], selected_categories: List[str], selected_parts: List[str], sku_query: str) -> str:
    week_chip = (
        f"Weeks: {min(selected_weeks)}–{max(selected_weeks)}"
        if selected_weeks
        else "Weeks: All"
    )
    cat_chip = "Category: All" if not selected_categories or "All Categories" in selected_categories else f"Category: {', '.join(selected_categories)}"
    part_chip = (
        "SKU: search"
        if sku_query
        else ("SKU: selected" if selected_parts else "SKU: All")
    )
    return "".join([f"<span class='chip'>{txt}</span>" for txt in [week_chip, cat_chip, part_chip]])


def render_page_header(title: str, breadcrumb: str, filter_summary_html: str, export_df: Optional[pd.DataFrame] = None, export_name: str = "export.csv"):
    inject_base_styles()
    top = st.container()
    c1, c2, c3 = top.columns([6, 2, 2])
    with c1:
        st.markdown(
            f"<div class='app-top-bar'><div class='breadcrumb'>{breadcrumb}</div><div class='page-title'>{title}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        btn_cols = st.columns(2)
        if btn_cols[0].button("Refresh"):
            st.experimental_rerun()
        if export_df is not None and not export_df.empty:
            btn_cols[1].download_button(
                "Export CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name=export_name,
                mime="text/csv",
            )
    with c3:
        st.button("Help", help="Questions? Use the in-app actions or share feedback with the data team.")
    st.markdown(f"<div class='chip-row'>{filter_summary_html}</div>", unsafe_allow_html=True)


# ---------- UI setup ----------
st.set_page_config(page_title="Innova x AutoZone Decision Dashboard", layout="wide")
inject_base_styles()
st.title("Innova × AutoZone Decision Dashboard")
st.caption("Decision-first dashboard with streamlined navigation and filters.")

data_ctx = dc.load_dashboard_data()
files = data_ctx.get("files", [])
if not files:
    st.error("No files found. Place Innova-AZ FYxxFWxx.xlsx files next to app.py.")
    st.stop()

fact_sales = data_ctx.get("fact_sales", pd.DataFrame()).copy()
if fact_sales.empty:
    st.error("No sales data found. Check TY-LY sheets in the FYxxFWxx files.")
    st.stop()

# ----- Sidebar: navigation + filters -----
weeks = data_ctx.get("weeks") or sorted(fact_sales["fiscal_week"].dropna().unique())
with st.sidebar:
    st.markdown("### Navigate")
    nav_choice = st.radio("Navigate", ["Overview", "Performance", "Supply Health", "Action Center", "More…"], index=0)
    more_page = None
    if nav_choice == "More…":
        more_page = st.selectbox("More pages", ["Returns", "Finance / Profitability", "Data Quality / Debug"])

    st.markdown("---")
    st.markdown("### Quick filters")
    default_weeks = weeks[-4:] if len(weeks) >= 4 else weeks
    selected_weeks = st.multiselect("Fiscal Weeks", options=weeks, default=default_weeks)
    if not selected_weeks:
        selected_weeks = default_weeks or weeks
    category_options = ["All Categories"] + sorted(fact_sales["major_category"].dropna().unique())
    selected_categories = st.multiselect("Major Category", options=category_options, default=["All Categories"])
    sku_query = st.text_input("Part Number / SKU search (optional)", "")
    part_series = column_as_series(fact_sales, "part_number")
    part_options = sorted(part_series.dropna().unique())
    selected_parts = st.multiselect("Focus SKUs (optional)", options=part_options, default=[])

    st.markdown("---")
    with st.expander("Advanced settings", expanded=False):
        top_n = st.slider("Top N rows", min_value=5, max_value=50, value=15, step=5)
        show_forecast_overlay = st.checkbox("Overlay Actuals vs Forecast (where available)", value=True)
        st.subheader("Alert thresholds")
        fill_rate_threshold = st.slider("Fill rate minimum", 0.5, 1.0, 0.9, 0.05)
        outs_threshold = st.number_input("OOS exposure threshold", min_value=0.0, value=200.0, step=50.0)
        coverage_threshold = st.slider("Forecast coverage min", 0.2, 1.5, 1.0, 0.05)
        woh_min = st.slider("WOH min", 0.0, 8.0, 2.0, 0.5)
        woh_max = st.slider("WOH max", 4.0, 30.0, 12.0, 0.5)
        billbacks_pct = st.slider("Billbacks % of sales alert", 0.0, 0.25, 0.08, 0.01)
        st.subheader("Executive Summary Settings")
        min_ly_rev_floor = st.slider(
            "Comparable YoY LY revenue floor ($)",
            min_value=0.0,
            max_value=1000.0,
            value=EXEC_MIN_LY_REV_DEFAULT,
            step=50.0,
            help="Exclude parts/categories from YoY% rankings when |LY revenue| is below this floor.",
        )

thresholds = {
    "fill_rate": fill_rate_threshold,
    "outs_exposure": outs_threshold,
    "coverage": coverage_threshold,
    "woh_min": woh_min,
    "woh_max": woh_max,
    "billbacks_pct": billbacks_pct,
    "min_ly_rev_floor": min_ly_rev_floor,
}

filters = {
    "selected_weeks": selected_weeks,
    "selected_categories": selected_categories,
    "selected_parts": selected_parts,
    "sku_query": sku_query,
    "top_n": top_n,
    "thresholds": thresholds,
}

ctx = dc.prepare_context(filters, data_ctx)
filtered_sales = ctx["filtered_sales"]
filtered_units_yoy = ctx["filtered_units_yoy"]
filtered_store_counts = ctx["filtered_store_counts"]
filtered_forecast = ctx["filtered_forecast"]
filtered_orders = ctx["filtered_orders"]
forecast_actual_src = ctx["forecast_actual_src"]
filtered_cpfr = ctx["filtered_cpfr"]
cpfr_detail_filtered = ctx["cpfr_detail_filtered"]
filtered_redflags = ctx["filtered_redflags"]
filtered_outs = ctx["filtered_outs"]
filtered_outs_totals = ctx["filtered_outs_totals"]
filtered_inventory = ctx["filtered_inventory"]
filtered_returns = ctx["filtered_returns"]
filtered_billbacks = ctx["filtered_billbacks"]
filtered_cost = ctx["filtered_cost"]
exec_sales = ctx["exec_sales"]
exec_sales_filtered = ctx["exec_sales_filtered"]
exec_tyly_gt = ctx["exec_tyly_gt"]
exec_tyly_gt_filtered = ctx["exec_tyly_gt_filtered"]
forecast_totals = ctx["forecast_totals"]
order_totals = ctx["order_totals"]
cpfr = ctx["cpfr"]
outs = ctx["outs"]
outs_totals = ctx["outs_totals"]
inventory = ctx["inventory"]
returns = ctx["returns"]
billbacks = ctx["billbacks"]
dim_category = ctx["dim_category"]
dim_billback_reason = ctx["dim_billback_reason"]
dq_removed_rows = ctx.get("dq_removed_rows", 0)
dq_bad_tokens_remaining = ctx.get("dq_bad_tokens_remaining", 0)
alerts = ctx["alerts"]


def render_kpi_tiles_es(gt_df: pd.DataFrame):
    if gt_df.empty:
        st.info("No Grand Total rows found in TY-LY sheets.")
        return

    latest_year, snapshot_week, prev_week = get_snapshot_context(gt_df)
    if snapshot_week is None:
        st.info("No fiscal week found in Grand Total rows.")
        return

    snap = gt_df[(gt_df["fiscal_year"] == latest_year) & (gt_df["fiscal_week"] == snapshot_week)]
    prev = (
        gt_df[(gt_df["fiscal_year"] == latest_year) & (gt_df["fiscal_week"] == prev_week)]
        if prev_week is not None
        else pd.DataFrame()
    )

    units = float(snap["fw_units_total"].iloc[0]) if not snap.empty and pd.notna(snap["fw_units_total"].iloc[0]) else None
    revenue = float(snap["fw_revenue_total"].iloc[0]) if not snap.empty and pd.notna(snap["fw_revenue_total"].iloc[0]) else None
    units_yoy = (
        float(snap["fw_unit_diff_pct"].iloc[0])
        if ("fw_unit_diff_pct" in snap.columns and not snap.empty and pd.notna(snap["fw_unit_diff_pct"].iloc[0]))
        else None
    )
    revenue_yoy = (
        float(snap["fw_revenue_diff_pct"].iloc[0])
        if ("fw_revenue_diff_pct" in snap.columns and not snap.empty and pd.notna(snap["fw_revenue_diff_pct"].iloc[0]))
        else None
    )
    prev_units = float(prev["fw_units_total"].iloc[0]) if not prev.empty and pd.notna(prev["fw_units_total"].iloc[0]) else None
    prev_revenue = float(prev["fw_revenue_total"].iloc[0]) if not prev.empty and pd.notna(prev["fw_revenue_total"].iloc[0]) else None

    units_wow = (units - prev_units) / prev_units if prev_units and prev_units != 0 else None
    revenue_wow = (revenue - prev_revenue) / prev_revenue if prev_revenue and prev_revenue != 0 else None
    asp = revenue / units if units else None

    cols = st.columns(5)
    cols[0].metric(
        f"Units (FW{snapshot_week})",
        f"{units:,.0f}" if units is not None else "N/A",
        delta=f"{units_wow:.2%} WoW" if units_wow is not None else "N/A" if prev_week is None else None,
        help="Snapshot = max imported fiscal week Grand Total FW Units. WoW uses previous imported week.",
    )
    cols[1].metric(
        f"Revenue (FW{snapshot_week})",
        format_currency_0(revenue),
        delta=f"{revenue_wow:.2%} WoW" if revenue_wow is not None else "N/A" if prev_week is None else None,
        help="Snapshot = max imported fiscal week Grand Total FW Net Rtl. WoW uses previous imported week.",
    )
    cols[2].metric("ASP", f"${asp:,.2f}" if asp else "N/A", help="ASP = Revenue / Units from Grand Total.")
    cols[3].metric(
        "Units YoY%",
        f"{units_yoy:.2%}" if units_yoy is not None else "N/A",
        help="Grand Total TY-LY column FW Unit Diff% (ratio).",
    )
    cols[4].metric(
        "Revenue YoY%",
        f"{revenue_yoy:.2%}" if revenue_yoy is not None else "N/A",
        help="Grand Total TY-LY column FW Rtl Diff% (ratio).",
    )

    latest_year_rows = gt_df[gt_df["fiscal_year"] == latest_year]
    range_units = latest_year_rows["fw_units_total"].sum()
    range_revenue = latest_year_rows["fw_revenue_total"].sum()
    st.markdown(
        f"<div style='font-size:16px;font-weight:600;color:#0f766e;'>Range Totals (FY{latest_year}): Units {range_units:,.0f} | Revenue {format_currency_0(range_revenue)}</div>",
        unsafe_allow_html=True,
    )


def render_alerts(alerts_list: List[Dict[str, str]]):
    if not alerts_list:
        st.success("No alerts triggered for the selected filters.")
        return
    severity_order = {"high": 0, "medium": 1, "low": 2}
    sorted_alerts = sorted(alerts_list, key=lambda a: severity_order.get(a.get("severity", "medium"), 3))
    top_alerts = sorted_alerts[:5]
    for alert in top_alerts:
        st.warning(f"**{alert['alert_type']}** — {alert['message']}  \nAction: {alert['action']}")
    if len(sorted_alerts) > 5:
        with st.expander("View all alerts"):
            st.dataframe(pd.DataFrame(sorted_alerts), hide_index=True, use_container_width=True)


def render_hero_tiles_es(df: pd.DataFrame, dimension: str, label: str, snapshot_week: int, prev_week: Optional[int]):
    if df.empty:
        return
    yoy_df = compute_comparable_yoy(df, snapshot_week, dimension, thresholds["min_ly_rev_floor"])
    hero = yoy_df.sort_values("ty_rev", ascending=False).head(1)
    declining = yoy_df.sort_values("ty_rev", ascending=True).head(1)
    wow_df = compute_wow_pct(df, snapshot_week, prev_week, dimension)
    rising = wow_df.sort_values("wow_rev_pct", ascending=False).head(1) if not wow_df.empty else pd.DataFrame()

    cols = st.columns(3)
    if not hero.empty:
        r = hero.iloc[0]
        yoy_delta = f"{r['yoy_rev_pct_comp']:.2%} Rev YoY" if pd.notna(r.get("yoy_rev_pct_comp")) else "N/A Rev YoY"
        cols[0].metric(
            f"Hero {label}",
            str(r[dimension]),
            delta=yoy_delta,
            help=f"Revenue YoY (comparable) shown for the highest revenue {label.lower()} in snapshot week.",
        )
    else:
        cols[0].metric(f"Hero {label}", "N/A", help="No comparable YoY% after filters.")
    if not rising.empty and pd.notna(rising.iloc[0]["wow_rev_pct"]):
        r = rising.iloc[0]
        cols[1].metric(
            f"Rising {label}",
            str(r[dimension]),
            delta=f"{r['wow_rev_pct']:.2%} Rev WoW",
            help="Revenue WoW Growth: (FW snapshot revenue - FW prev revenue) / FW prev revenue.",
        )
    else:
        cols[1].metric(f"Rising {label}", "N/A", help="Requires previous week revenue > 0.")
    if not declining.empty:
        r = declining.iloc[0]
        yoy_delta = f"{r['yoy_rev_pct_comp']:.2%} Rev YoY" if pd.notna(r.get("yoy_rev_pct_comp")) else "N/A Rev YoY"
        cols[2].metric(
            f"Declining {label}",
            str(r[dimension]),
            delta=yoy_delta,
            help=f"Revenue YoY (comparable) shown for the lowest revenue {label.lower()} in snapshot week.",
        )
    else:
        cols[2].metric(f"Declining {label}", "N/A", help="No comparable YoY% after filters.")


def render_hero_tiles_es_ytd(df: pd.DataFrame, dimension: str, label: str, snapshot_week: int, prev_week: Optional[int]):
    if df.empty:
        return
    yoy_df = compute_ytd_yoy(df, snapshot_week, dimension, thresholds["min_ly_rev_floor"])
    hero = yoy_df.sort_values("fytd_rev", ascending=False).head(1)
    declining = yoy_df.sort_values("fytd_rev", ascending=True).head(1)
    wow_df = compute_ytd_wow(df, snapshot_week, prev_week, dimension)
    rising = wow_df.sort_values("fytd_wow_pct", ascending=False).head(1) if not wow_df.empty else pd.DataFrame()

    cols = st.columns(3)
    if not hero.empty:
        r = hero.iloc[0]
        yoy_delta = f"{r['fytd_yoy_pct']:.2%} Rev YoY" if pd.notna(r.get("fytd_yoy_pct")) else "N/A Rev YoY"
        cols[0].metric(
            f"Hero {label} (FYTD)",
            str(r[dimension]),
            delta=yoy_delta,
            help=f"FYTD Revenue YoY Growth: TY-LY FYTD Rtl$ Diff% with LY floor {thresholds['min_ly_rev_floor']}.",
        )
    else:
        cols[0].metric(f"Hero {label} (FYTD)", "N/A", help="No FYTD comparable YoY% after filters.")

    if not rising.empty and pd.notna(rising.iloc[0]["fytd_wow_pct"]):
        r = rising.iloc[0]
        cols[1].metric(
            f"Rising {label} (FYTD)",
            str(r[dimension]),
            delta=f"{r['fytd_wow_pct']:.2%} Rev WoW",
            help="FYTD WoW% = (FYTD revenue snapshot - FYTD revenue prev) / FYTD revenue prev.",
        )
    else:
        cols[1].metric(f"Rising {label} (FYTD)", "N/A", help="Requires previous week FYTD revenue > 0.")

    if not declining.empty:
        r = declining.iloc[0]
        yoy_delta = f"{r['fytd_yoy_pct']:.2%} Rev YoY" if pd.notna(r.get("fytd_yoy_pct")) else "N/A Rev YoY"
        cols[2].metric(
            f"Declining {label} (FYTD)",
            str(r[dimension]),
            delta=yoy_delta,
            help=f"FYTD Revenue YoY Decrease: TY-LY FYTD Rtl$ Diff% with LY floor {thresholds['min_ly_rev_floor']}.",
        )
    else:
        cols[2].metric(f"Declining {label} (FYTD)", "N/A", help="No FYTD comparable YoY% after filters.")



# ----- Page renderers -----

def render_overview_page():
    filter_summary_html = format_filter_summary(selected_weeks, selected_categories, selected_parts, sku_query)
    render_page_header("Overview", "Home / Overview", filter_summary_html, export_df=exec_tyly_gt_filtered, export_name="overview.csv")
    with card("KPI Tiles"):
        render_kpi_tiles_es(exec_tyly_gt_filtered)

    snapshot_year, snapshot_week, prev_week = get_snapshot_context(exec_tyly_gt_filtered)
    hero_cols = st.columns(2)
    with hero_cols[0]:
        with card("Hero / Rising / Declining Products"):
            if snapshot_week is None:
                st.info("No snapshot week found in Grand Total rows.")
            else:
                render_hero_tiles_es(exec_sales_filtered, dimension="part_number", label="Product", snapshot_week=snapshot_week, prev_week=prev_week)
    with hero_cols[1]:
        with card("Hero / Rising / Declining Categories"):
            if snapshot_week is None:
                st.info("No snapshot week found in Grand Total rows.")
            else:
                render_hero_tiles_es(exec_sales_filtered, dimension="major_category", label="Category", snapshot_week=snapshot_week, prev_week=prev_week)

    if exec_tyly_gt_filtered.empty:
        st.info("Not enough data for trend.")
    else:
        trend = exec_tyly_gt_filtered.sort_values(["fiscal_year", "fiscal_week"])
        trend = trend.assign(fw_revenue_total_disp=lambda d: d["fw_revenue_total"].apply(round_half_up))
        line_units = (
            alt.Chart(trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("fiscal_week:O", title="Fiscal Week", axis=alt.Axis(format="d")),
                y="fw_units_total:Q",
                color="fiscal_year:N",
                tooltip=["fiscal_year", "fiscal_week", alt.Tooltip("fw_units_total:Q", format=",")],
            )
            .properties(height=260)
        )
        line_rev = (
            alt.Chart(trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("fiscal_week:O", title="Fiscal Week", axis=alt.Axis(format="d")),
                y=alt.Y("fw_revenue_total_disp:Q", axis=alt.Axis(format="$,.0f")),
                color="fiscal_year:N",
                tooltip=["fiscal_year", "fiscal_week", alt.Tooltip("fw_revenue_total_disp:Q", format="$,.0f")],
            )
            .properties(height=260)
        )
        trend_cols = st.columns(2)
        with trend_cols[0]:
            with card("Revenue Trend (weekly)"):
                st.altair_chart(line_rev, use_container_width=True)
        with trend_cols[1]:
            with card("Units Trend (weekly)"):
                st.altair_chart(line_units, use_container_width=True)

    with card("Alerts"):
        render_alerts(alerts)


def render_performance_page():
    filter_summary_html = format_filter_summary(selected_weeks, selected_categories, selected_parts, sku_query)
    render_page_header("Performance", "Home / Performance", filter_summary_html, export_df=filtered_sales, export_name="performance.csv")
    with card("Controls"):
        perf_cols = st.columns(2)
        with perf_cols[0]:
            metric_choice = st.selectbox("Metric", ["Revenue", "Units"], key="perf_metric")
        with perf_cols[1]:
            view_choice = st.selectbox("View", ["Product", "Category"], key="perf_view")
    metric_col = "revenue" if metric_choice == "Revenue" else "units"

    if view_choice == "Product":
        prod_df = filtered_sales.copy()
        prod_df = prod_df.dropna(subset=["part_number"])
        prod_df = prod_df[~prod_df["part_number"].astype(str).str.contains("TOTAL", case=False, na=False)]
        prod_group = (
            prod_df.groupby(["part_number", "description"])
            .agg(revenue=("revenue", "sum"), units=("units", "sum"))
            .reset_index()
            .sort_values(metric_col, ascending=False)
        )
        prod_group.insert(0, "rank", range(1, len(prod_group) + 1))
        prod_group = prod_group.head(top_n)
        trend_data = (
            prod_df.groupby(["part_number", "fiscal_week"])[metric_col]
            .sum()
            .reset_index()
            .sort_values(["part_number", "fiscal_week"])
        )
        left, right = st.columns(2)
        with left:
            with card("Top performers (Product)"):
                display = prod_group.copy()
                if metric_col == "revenue":
                    display = format_currency_columns(display, ["revenue"])
                st.dataframe(display, use_container_width=True, hide_index=True)
        with right:
            with card(f"{metric_choice} trend by product"):
                options = list(prod_group["part_number"].astype(str))
                default_sel = options[: min(3, len(options))]
                focus_parts = st.multiselect("Select SKUs", options=options, default=default_sel, key="trend_parts")
                chart_df = trend_data[trend_data["part_number"].astype(str).isin(focus_parts)] if focus_parts else trend_data
                if chart_df.empty:
                    st.info("No data for selected SKUs.")
                else:
                    chart = (
                        alt.Chart(chart_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("fiscal_week:O", title="Fiscal Week", axis=alt.Axis(tickMinStep=1, format="d")),
                            y=alt.Y(f"{metric_col}:Q", axis=alt.Axis(format="$,.0f" if metric_col == "revenue" else ",")),
                            color="part_number:N",
                        )
                    )
                    st.altair_chart(chart, use_container_width=True)
        with card("Category breakdown"):
            cat_group = (
                filtered_sales.dropna(subset=["major_category"])
                .groupby("major_category")[metric_col]
                .sum()
                .reset_index()
                .sort_values(metric_col, ascending=False)
            )
            bar = (
                alt.Chart(cat_group)
                .mark_bar()
                .encode(
                    x=alt.X("major_category:N", title="Major Category"),
                    y=alt.Y(f"{metric_col}:Q", axis=alt.Axis(format="$,.0f" if metric_col == "revenue" else ",")),
                    color="major_category:N",
                )
            )
            st.altair_chart(bar, use_container_width=True)
            st.caption("Breakdown uses filtered sales with current quick filters applied.")
    else:
        cat_df = filtered_sales.copy()
        cat_df = cat_df.dropna(subset=["major_category"])
        cat_group = (
            cat_df.groupby("major_category")
            .agg(revenue=("revenue", "sum"), units=("units", "sum"))
            .reset_index()
            .sort_values(metric_col, ascending=False)
        )
        cat_group.insert(0, "rank", range(1, len(cat_group) + 1))
        cat_group = cat_group.head(top_n)
        trend_data = (
            cat_df.groupby(["major_category", "fiscal_week"])[metric_col]
            .sum()
            .reset_index()
            .sort_values(["major_category", "fiscal_week"])
        )
        left, right = st.columns(2)
        with left:
            with card("Top performers (Category)"):
                display = cat_group.copy()
                if metric_col == "revenue":
                    display = format_currency_columns(display, ["revenue"])
                st.dataframe(display, use_container_width=True, hide_index=True)
        with right:
            with card(f"{metric_choice} trend by category"):
                options = list(cat_group["major_category"].astype(str))
                default_sel = options[: min(3, len(options))]
                focus_cats = st.multiselect("Select categories", options=options, default=default_sel, key="trend_cats")
                chart_df = trend_data[trend_data["major_category"].astype(str).isin(focus_cats)] if focus_cats else trend_data
                if chart_df.empty:
                    st.info("No data for selected categories.")
                else:
                    chart = (
                        alt.Chart(chart_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("fiscal_week:O", title="Fiscal Week", axis=alt.Axis(tickMinStep=1, format="d")),
                            y=alt.Y(f"{metric_col}:Q", axis=alt.Axis(format="$,.0f" if metric_col == "revenue" else ",")),
                            color="major_category:N",
                        )
                    )
                    st.altair_chart(chart, use_container_width=True)
        with card("Product detail (top N)"):
            prod_df = filtered_sales.dropna(subset=["part_number"])
            prod_group = (
                prod_df.groupby(["part_number", "description"])
                .agg(revenue=("revenue", "sum"), units=("units", "sum"))
                .reset_index()
                .sort_values(metric_col, ascending=False)
            ).head(top_n)
            display = prod_group.copy()
            if metric_col == "revenue":
                display = format_currency_columns(display, ["revenue"])
            st.dataframe(display, use_container_width=True, hide_index=True)


def render_supply_health_page():
    filter_summary_html = format_filter_summary(selected_weeks, selected_categories, selected_parts, sku_query)
    render_page_header("Supply Health", "Home / Supply Health", filter_summary_html, export_df=filtered_cpfr, export_name="supply.csv")
    with card("Service KPIs"):
        if filtered_cpfr.empty:
            st.info("No CPFR data available.")
        else:
            cpfr_base = filtered_cpfr.sort_values(["snapshot_year", "snapshot_week"])
            cpfr_base = cpfr_base.drop_duplicates(subset=["fiscal_year", "fiscal_week"], keep="last")
            current_fy = int(cpfr_base["fiscal_year"].max())
            current_fw = int(cpfr_base[cpfr_base["fiscal_year"] == current_fy]["fiscal_week"].max())
            cur = cpfr_base[(cpfr_base["fiscal_year"] == current_fy) & (cpfr_base["fiscal_week"] == current_fw)]
            cur_row = cur.iloc[0] if not cur.empty else None
            cols = st.columns(3)
            label = f"FY{current_fy} FW{current_fw}"
            cols[0].metric(
                f"Shipped Units ({label})",
                f"{cur_row['shipped_units']:,.0f}" if cur_row is not None and pd.notna(cur_row["shipped_units"]) else "N/A",
            )
            cols[1].metric(
                f"Fill Rate ({label})",
                f"{cur_row['fill_rate']:.1%}" if cur_row is not None and pd.notna(cur_row["fill_rate"]) else "N/A",
                help="Fill Rate from CPFR Grand Total (no recompute).",
            )
            cols[2].metric(
                f"Not Shipped ({label})",
                f"{cur_row['not_shipped_units']:,.0f}" if cur_row is not None and pd.notna(cur_row["not_shipped_units"]) else "N/A",
                help="Not Shipped from CPFR Grand Total (no recompute).",
            )

    if not filtered_cpfr.empty:
        years = sorted(filtered_cpfr["fiscal_year"].unique(), reverse=True)[:3]
        trend = filtered_cpfr[filtered_cpfr["fiscal_year"].isin(years)].sort_values(["fiscal_year", "fiscal_week"])
        trend_cols = st.columns(2)
        with trend_cols[0]:
            with card("Coverage / supply trend"):
                shipped_chart = (
                    alt.Chart(trend)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("fiscal_week:Q", title="Fiscal Week"),
                        y=alt.Y("shipped_units:Q", title="Shipped Units"),
                        color=alt.Color("fiscal_year:N", title="Fiscal Year"),
                        tooltip=["fiscal_year", "fiscal_week", alt.Tooltip("shipped_units:Q", format=",")],
                    )
                )
                st.altair_chart(shipped_chart, use_container_width=True)
        with trend_cols[1]:
            with card("Fill rate / forecast error"):
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
                st.altair_chart(fill_chart, use_container_width=True)

    with card("Supply exceptions"):
        if filtered_redflags.empty and filtered_outs.empty:
            st.success("No supply exceptions found for the selected filters.")
        else:
            if not filtered_redflags.empty:
                cols_rf = st.columns(3)
                rf_cur = filtered_redflags.copy()
                rf_cur = rf_cur.dropna(subset=["part_number"])
                def top_card(metric: str, title: str, col):
                    sub = rf_cur.dropna(subset=[metric])
                    if sub.empty:
                        col.metric(title, "N/A")
                        return
                    row = sub.loc[sub[metric].idxmax()]
                    col.metric(title, str(row["part_number"]), delta=f"{int(row[metric]):,} Not Shipped")
                top_card("not_shipped_lfw", "Top Red Flag — LFW", cols_rf[0])
                top_card("not_shipped_l4w", "Top Red Flag — L4FW", cols_rf[1])
                top_card("not_shipped_l52w", "Top Red Flag — L52FW", cols_rf[2])
            if not filtered_outs.empty:
                outs_top = filtered_outs.sort_values("store_oos_exposure", ascending=False).head(top_n)
                st.dataframe(outs_top, use_container_width=True, hide_index=True)


def render_action_center_page():
    filter_summary_html = format_filter_summary(selected_weeks, selected_categories, selected_parts, sku_query)
    render_page_header("Action Center", "Home / Action Center", filter_summary_html, export_df=None, export_name="actions.csv")
    action_rows = []
    if not filtered_outs.empty:
        outs_summary = (
            filtered_outs.groupby("part_number")
            .agg(store_oos_exposure=("store_oos_exposure", "sum"))
            .reset_index()
        )
        outs_summary["severity"] = outs_summary["store_oos_exposure"].apply(lambda v: "High" if v >= outs_threshold else "Medium")
        outs_summary["source"] = "OOS"
        action_rows.append(outs_summary)
    if not filtered_redflags.empty:
        rf_summary = filtered_redflags.groupby("part_number")[["not_shipped_lfw", "not_shipped_l4w", "not_shipped_l52w"]].max().reset_index()
        rf_summary["store_oos_exposure"] = rf_summary["not_shipped_lfw"].fillna(0)
        rf_summary["severity"] = "High"
        rf_summary["source"] = "Fill Rate Red Flags"
        rf_summary = rf_summary[["part_number", "store_oos_exposure", "severity", "source"]]
        action_rows.append(rf_summary)
    action_table = pd.concat(action_rows, ignore_index=True) if action_rows else pd.DataFrame(columns=["part_number", "store_oos_exposure", "severity", "source"])
    if not filtered_sales.empty and not action_table.empty:
        rev_map = filtered_sales.groupby("part_number")["revenue"].sum().reset_index().rename(columns={"revenue": "revenue_impact"})
        action_table = action_table.merge(rev_map, on="part_number", how="left")
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    action_table["severity_rank"] = action_table["severity"].map(severity_order).fillna(3)

    with card("Action KPIs"):
        total_actions = action_table["part_number"].nunique()
        high_count = int(action_table[action_table["severity"].eq("High")]["part_number"].nunique()) if not action_table.empty else 0
        rev_risk = action_table["revenue_impact"].sum() if "revenue_impact" in action_table.columns else 0
        cols = st.columns(3)
        cols[0].metric("SKUs needing action", f"{total_actions}")
        cols[1].metric("High severity", f"{high_count}")
        cols[2].metric("Est. revenue at risk", format_currency_0(rev_risk))

    with card("Action list"):
        if action_table.empty:
            st.info("No action-triggering SKUs for the selected filters.")
        else:
            severity_filter = st.selectbox("Severity", ["All", "High", "Medium", "Low"], key="severity_filter")
            keyword = st.text_input("Search keyword", value=sku_query)
            table = action_table.copy()
            if severity_filter != "All":
                table = table[table["severity"] == severity_filter]
            if keyword:
                table = table[table["part_number"].astype(str).str.contains(keyword, case=False, na=False)]
            table = table.sort_values(["severity_rank", "revenue_impact"], ascending=[True, False])
            st.dataframe(table.drop(columns=["severity_rank"], errors="ignore"), use_container_width=True, hide_index=True)
            st.caption("Sorted by severity then revenue impact.")

    with card("SKU drill-down"):
        if action_table.empty:
            st.info("No SKU selected.")
        else:
            options = sorted(action_table["part_number"].astype(str).unique())
            chosen_sku = st.selectbox("Select SKU", options=options)
            sku_sales = filtered_sales[filtered_sales["part_number"].astype(str) == chosen_sku]
            if sku_sales.empty:
                st.info("No sales history for selected SKU.")
            else:
                trend = sku_sales.groupby("fiscal_week")[["units", "revenue"]].sum().reset_index().sort_values("fiscal_week")
                chart = (
                    alt.Chart(trend.melt(id_vars="fiscal_week", value_vars=["units", "revenue"], var_name="metric", value_name="value"))
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("fiscal_week:O", title="Fiscal Week"),
                        y=alt.Y("value:Q", axis=alt.Axis(format=",")),
                        color="metric:N",
                    )
                )
                st.altair_chart(chart, use_container_width=True)
                st.caption("Units and revenue trend for selected SKU using filtered weeks.")


def render_returns_page():
    filter_summary_html = format_filter_summary(selected_weeks, selected_categories, selected_parts, sku_query)
    render_page_header("Returns", "Home / More / Returns", filter_summary_html, export_df=filtered_returns, export_name="returns.csv")
    with card("Returns KPIs"):
        if filtered_returns.empty:
            st.info("No returns data available.")
        else:
            volume_gate = st.number_input("Volume gate (gross units min)", min_value=0, value=50, step=10)
            latest_year = filtered_returns["fiscal_year"].max()
            latest_week = filtered_returns[filtered_returns["fiscal_year"] == latest_year]["fiscal_week"].max()
            latest = filtered_returns[
                (filtered_returns["fiscal_year"] == latest_year) & (filtered_returns["fiscal_week"] == latest_week)
            ]
            latest = latest[latest["gross_units"] >= volume_gate]
            avg_damaged = latest["damaged_rate"].mean(skipna=True)
            avg_undamaged = latest["undamaged_rate"].mean(skipna=True)
            cols = st.columns(3)
            cols[0].metric(
                "Avg Damaged Rate",
                f"{avg_damaged:.2%}" if pd.notna(avg_damaged) else "NA",
                help="Mean damaged_rate for snapshot week (filtered by volume gate).",
            )
            cols[1].metric(
                "Avg Undamaged Rate",
                f"{avg_undamaged:.2%}" if pd.notna(avg_undamaged) else "NA",
                help="Mean undamaged_rate for snapshot week (filtered by volume gate).",
            )
            if not latest.empty:
                top_risk = latest.sort_values("damaged_rate", ascending=False).head(1)
                cols[2].metric("Top Risk SKU", str(top_risk.iloc[0]["part_number"]), delta=f"{top_risk.iloc[0]['damaged_rate']:.2%} Damaged")
    with card("Top Risk SKUs"):
        if filtered_returns.empty:
            st.info("No returns data available.")
        else:
            volume_gate = st.number_input("Volume gate for list", min_value=0, value=50, step=10, key="volume_gate_list")
            ret_top = (
                filtered_returns[filtered_returns["gross_units"] >= volume_gate]
                .sort_values("damaged_rate", ascending=False)
                .head(top_n)
            )
            ret_top = ret_top.reset_index(drop=True)
            ret_top.insert(0, "rank", ret_top.index + 1)
            st.dataframe(ret_top, use_container_width=True, hide_index=True)


def render_finance_page():
    filter_summary_html = format_filter_summary(selected_weeks, selected_categories, selected_parts, sku_query)
    render_page_header("Finance / Profitability", "Home / More / Finance", filter_summary_html, export_df=filtered_billbacks, export_name="finance.csv")
    with card("Gross Margin by SKU"):
        margin = compute_margin(filtered_sales, filtered_cost, selected_weeks)
        if margin.empty:
            st.info("Missing cost data for margin calculation.")
        else:
            summary = margin.agg({"revenue": "sum", "cogs": "sum", "gross_margin": "sum"})
            gm_pct = summary["gross_margin"] / summary["revenue"] if summary["revenue"] else None
            cols = st.columns(3)
            cols[0].metric("Revenue", f"${summary['revenue']:,.0f}", help="Selected weeks total revenue.")
            cols[1].metric("Gross Margin", f"${summary['gross_margin']:,.0f}", help="GM = Revenue - COGS using unit_cost from Cost file.")
            cols[2].metric("GM%", f"{gm_pct:.1%}" if gm_pct else "NA", help="GM% = GM / Revenue.")
            top_margin = margin.sort_values("gross_margin", ascending=False).head(top_n)
            top_margin = top_margin.reset_index(drop=True)
            top_margin.insert(0, "rank", top_margin.index + 1)
            display_df = top_margin[
                ["rank", "part_number", "item_id", "description", "revenue", "cogs", "gross_margin", "gross_margin_pct"]
            ]
            display_df = format_currency_columns(display_df, ["revenue", "cogs", "gross_margin"])
            display_df = format_percent_columns(display_df, ["gross_margin_pct"], decimals=1)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    with card("Billbacks Trend"):
        if filtered_billbacks.empty:
            st.info("Billbacks data not available.")
        else:
            mapped = filtered_billbacks.copy()
            if not dim_billback_reason.empty:
                dim_lookup = (
                    dim_billback_reason.explode("all_codes_norm")[["all_codes_norm", "bucket", "direction", "title"]]
                    .rename(columns={"all_codes_norm": "code_norm"})
                )
                mapped["code_norm"] = mapped["type_code_norm"].apply(normalize_fee_code)
                mapped = mapped.merge(dim_lookup, on="code_norm", how="left")
            weekly = (
                mapped.assign(invoice_week_start=lambda d: d["invoice_date"] - pd.to_timedelta(d["invoice_date"].dt.weekday, unit="D"))
                .groupby(["invoice_week_start", "bucket"])["billback_amount"]
                .sum()
                .reset_index()
                .sort_values("invoice_week_start")
            )
            if weekly.empty:
                st.info("No billback totals to plot.")
            else:
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
                st.altair_chart(bar, use_container_width=True)


def render_debug_page():
    filter_summary_html = format_filter_summary(selected_weeks, selected_categories, selected_parts, sku_query)
    render_page_header("Data Quality / Debug", "Home / More / Debug", filter_summary_html)
    with card("Data Quality"):
        st.markdown("**Row counts**")
        st.write(
            {
                "sales_rows": len(fact_sales),
                "forecast_rows": len(forecast_totals),
                "orders_rows": len(order_totals),
                "cpfr_rows": len(cpfr),
                "outs_rows": len(outs),
                "inventory_rows": len(inventory),
                "returns_rows": len(returns),
                "billbacks_rows": len(billbacks),
            }
        )
        st.markdown("**Cleaning checks**")
        st.write(
            {
                "sales_rows_removed_invalid_sku": int(dq_removed_rows),
                "bad_part_tokens_remaining": dq_bad_tokens_remaining,
            }
        )
        if not dim_category.empty:
            st.markdown(f"**Category dim** (rows={len(dim_category)})")
            st.dataframe(dim_category.head(3), hide_index=True)
        if not fact_sales.empty:
            st.markdown("**Week coverage**")
            week_cov = (
                fact_sales.groupby("fiscal_year")["fiscal_week"]
                .agg(["min", "max", "nunique"])
                .reset_index()
                .rename(columns={"nunique": "weeks_present"})
            )
            st.dataframe(week_cov, hide_index=True)
            st.markdown("**Null SKU counts (sales)**")
            null_skus = fact_sales["part_number"].isna().sum()
            st.write({"null_part_number_rows": int(null_skus)})
            if "major_category" in fact_sales.columns:
                mapped_pct = (fact_sales["major_category"] != "Unmapped").mean() * 100
                st.write({"category_mapped_pct": f"{mapped_pct:.1f}%"})
                unmapped_top = (
                    fact_sales[fact_sales["major_category"] == "Unmapped"]["part_number"]
                    .value_counts()
                    .head(20)
                    .reset_index(name="count")
                    .rename(columns={"index": "part_number"})
                )
                st.markdown("**Top unmapped SKUs (sales)**")
                st.dataframe(unmapped_top, hide_index=True)
                st.markdown("**Unmapped category details**")
                unmapped_details = fact_sales[fact_sales["major_category"] == "Unmapped"][
                    ["part_number", "item_id", "description"]
                ]
                st.dataframe(unmapped_details.drop_duplicates(), hide_index=True)
    st.caption("Add new FYxxFWxx files to extend the time series; dashboards refresh automatically.")


filter_summary_html = format_filter_summary(selected_weeks, selected_categories, selected_parts, sku_query)
current_page = nav_choice if nav_choice != "More…" else (more_page or "Overview")

if current_page == "Overview":
    render_overview_page()
elif current_page == "Performance":
    render_performance_page()
elif current_page == "Supply Health":
    render_supply_health_page()
elif current_page == "Action Center":
    render_action_center_page()
elif current_page == "Returns":
    render_returns_page()
elif current_page == "Finance / Profitability":
    render_finance_page()
else:
    render_debug_page()
