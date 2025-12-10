import re
from pathlib import Path
from typing import List, Tuple

import altair as alt
import pandas as pd
import streamlit as st


# ---------- Paths & constants ----------
DATA_DIR = Path(__file__).parent
FILE_GLOB = "Innova-AZ FY26FW*.xlsx"

SALES_COLUMNS = {
    "POV Number": "POVNumber",
    "Part": "PartNumber",
    "Item": "ItemNumber",
    "Description": "Description",
    "Store Count": "StoreCount",
    "FW Units": "FW_Units",
    "FW Net Rtl": "FW_NetRtl",
    "LY FW Units": "FW_Units_LY",
    "LYFW Net Rtl": "FW_NetRtl_LY",
    "L4FW Net Units": "L4_Units",
    "L4FW Net Rtl": "L4_NetRtl",
    "L52FW Net Units": "L52_Units",
    "L52FW Net Rtl$": "L52_NetRtl",
}


# ---------- Helpers ----------
def parse_fiscal_from_name(filename: str) -> Tuple[int, int]:
    """Return fiscal year (e.g., 2026) and fiscal week from file name."""
    match = re.search(r"FY(\d{2})FW(\d+)", filename)
    if not match:
        raise ValueError(f"Cannot parse fiscal info from {filename}")
    year = 2000 + int(match.group(1))
    week = int(match.group(2))
    return year, week


def numericize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric, keeping original frame reference."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_sales_fact() -> pd.DataFrame:
    frames = []
    for path in sorted(DATA_DIR.glob(FILE_GLOB)):
        fiscal_year, fiscal_week = parse_fiscal_from_name(path.name)
        sales = pd.read_excel(path, sheet_name="TY-LY", header=3)
        sales = sales.rename(columns=SALES_COLUMNS)
        sales = sales[list(SALES_COLUMNS.values())]
        sales["FiscalYear"] = fiscal_year
        sales["FiscalWeek"] = fiscal_week
        sales = numericize(
            sales,
            [
                "StoreCount",
                "FW_Units",
                "FW_NetRtl",
                "FW_Units_LY",
                "FW_NetRtl_LY",
                "L4_Units",
                "L4_NetRtl",
                "L52_Units",
                "L52_NetRtl",
            ],
        )
        # Drop rows without a Part/Item number to avoid totals/blanks.
        sales = sales.dropna(subset=["PartNumber", "ItemNumber"], how="all")
        frames.append(sales)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_units_yoy() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob(FILE_GLOB))
    if not files:
        return pd.DataFrame()
    latest = files[-1]
    df = pd.read_excel(latest, sheet_name="Units YOY", header=15)
    df = df.rename(columns={"FW": "FiscalWeek"})
    df = df.dropna(subset=["FiscalWeek"])
    long = df.melt(id_vars=["FiscalWeek"], var_name="FiscalYear", value_name="TotalUnits")
    long["FiscalYear"] = pd.to_numeric(long["FiscalYear"], errors="coerce").astype("Int64")
    long["FiscalWeek"] = pd.to_numeric(long["FiscalWeek"], errors="coerce").astype("Int64")
    long["TotalUnits"] = pd.to_numeric(long["TotalUnits"], errors="coerce")
    long = long.dropna(subset=["FiscalYear", "FiscalWeek", "TotalUnits"])
    long["FiscalYear"] = long["FiscalYear"].astype(int)
    long["FiscalWeek"] = long["FiscalWeek"].astype(int)
    return long


@st.cache_data(show_spinner=False)
def load_store_counts() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob(FILE_GLOB))
    if not files:
        return pd.DataFrame()
    latest = files[-1]
    df = pd.read_excel(latest, sheet_name="Historical Store Counts", header=8)
    week_cols = [c for c in df.columns if isinstance(c, str) and re.match(r"FY\d{2}FW\d{2}", c)]
    if not week_cols:
        return pd.DataFrame()
    long = df.melt(
        id_vars=["Part Number", "Item", "STORECOUNT"],
        value_vars=week_cols,
        var_name="FiscalLabel",
        value_name="StoreCount",
    )
    fiscal = long["FiscalLabel"].str.extract(r"FY(\d{2})FW(\d{2})")
    long["FiscalYear"] = fiscal[0].astype(float)
    long["FiscalWeek"] = fiscal[1].astype(float)
    long["FiscalYear"] = long["FiscalYear"].apply(lambda x: 2000 + int(x) if pd.notna(x) else None)
    long["FiscalWeek"] = long["FiscalWeek"].apply(lambda x: int(x) if pd.notna(x) else None)
    long["StoreCount"] = pd.to_numeric(long["StoreCount"], errors="coerce")
    long = long.dropna(subset=["FiscalYear", "FiscalWeek", "StoreCount"])
    return long[
        [
            "FiscalYear",
            "FiscalWeek",
            "StoreCount",
            "Part Number",
            "Item",
            "STORECOUNT",
        ]
    ]


@st.cache_data(show_spinner=False)
def load_cost_dim() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob(FILE_GLOB))
    if not files:
        return pd.DataFrame()
    latest = files[-1]
    df = pd.read_excel(latest, sheet_name="Cost File", header=13)
    base_cols = [c for c in df.columns if isinstance(c, (int, float, str)) and str(c).isdigit()]
    df["Cost_Base"] = df[base_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    df = df.rename(columns={"Part Nbr": "PartNumber", "Item Nbr": "ItemNumber"})
    df = df[["PartNumber", "ItemNumber", "Description", "Cost_Base"]]
    df = df.dropna(subset=["PartNumber", "ItemNumber"])
    return df


@st.cache_data(show_spinner=False)
def load_billbacks() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob(FILE_GLOB))
    if not files:
        return pd.DataFrame()
    latest = files[-1]
    df_raw = pd.read_excel(latest, sheet_name="Accntg-Billbacks", header=None)
    if df_raw.empty:
        return pd.DataFrame()
    header_idx = df_raw.index[df_raw.iloc[:, 0] == "Type Code"]
    if len(header_idx) == 0:
        return pd.DataFrame()
    header_row = header_idx[0]
    cols = df_raw.iloc[header_row].tolist()
    df = df_raw.iloc[header_row + 1 :].copy()
    df.columns = cols
    df = df.rename(
        columns={
            "Type Code": "TypeCode",
            "Invoice Nbr": "InvoiceNumber",
            "Invoice Date": "InvoiceDateRaw",
            "Total": "BillbackAmount",
        }
    )
    if "InvoiceDateRaw" not in df.columns:
        return pd.DataFrame()
    df["InvoiceDate"] = (
        df["InvoiceDateRaw"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(8)
        .pipe(pd.to_datetime, format="%m%d%Y", errors="coerce")
    )
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["BillbackAmount"] = pd.to_numeric(df["BillbackAmount"], errors="coerce")
    df = df.dropna(subset=["BillbackAmount"])
    return df[["TypeCode", "InvoiceNumber", "InvoiceDate", "YearMonth", "BillbackAmount"]]


def build_product_dim(fact_sales: pd.DataFrame) -> pd.DataFrame:
    if fact_sales.empty:
        return pd.DataFrame(columns=["ProductKey", "PartNumber", "ItemNumber", "Description"])
    products = (
        fact_sales[["PartNumber", "ItemNumber", "Description"]]
        .drop_duplicates()
        .assign(ProductKey=lambda df: df["PartNumber"].astype(str) + "-" + df["ItemNumber"].astype(str))
    )
    return products


def agg_units_per_store(fact_sales: pd.DataFrame, store_counts: pd.DataFrame) -> pd.DataFrame:
    if fact_sales.empty or store_counts.empty:
        return pd.DataFrame()
    if "STORECOUNT" in store_counts.columns and store_counts["STORECOUNT"].notna().any():
        weekly_store = (
            store_counts.groupby(["FiscalYear", "FiscalWeek"])["STORECOUNT"]
            .max()
            .reset_index()
            .rename(columns={"STORECOUNT": "WeeklyStoreCount"})
        )
    else:
        weekly_store = (
            store_counts.groupby(["FiscalYear", "FiscalWeek"])["StoreCount"]
            .max()
            .reset_index()
            .rename(columns={"StoreCount": "WeeklyStoreCount"})
        )
    fact = (
        fact_sales.groupby(["FiscalYear", "FiscalWeek", "PartNumber", "ItemNumber", "Description"])
        [["FW_Units", "FW_NetRtl"]]
        .sum()
        .reset_index()
    )
    merged = fact.merge(weekly_store, on=["FiscalYear", "FiscalWeek"], how="left")
    merged["UnitsPerStore"] = merged["FW_Units"] / merged["WeeklyStoreCount"]
    return merged


def compute_margin(fact_sales: pd.DataFrame, cost_dim: pd.DataFrame, fiscal_week: int) -> pd.DataFrame:
    if fact_sales.empty or cost_dim.empty:
        return pd.DataFrame()
    wk = fact_sales[fact_sales["FiscalWeek"] == fiscal_week]
    merged = wk.merge(
        cost_dim,
        on=["PartNumber", "ItemNumber"],
        how="left",
        suffixes=("", "_cost"),
    )
    if "Description_cost" in merged.columns:
        merged["Description"] = merged["Description"].fillna(merged["Description_cost"])
    merged["UnitCost"] = merged["Cost_Base"]
    merged["COGS"] = merged["FW_Units"] * merged["UnitCost"]
    merged["GrossMargin"] = merged["FW_NetRtl"] - merged["COGS"]
    merged["GrossMarginPct"] = merged["GrossMargin"] / merged["FW_NetRtl"]
    keep_cols = [
        "PartNumber",
        "ItemNumber",
        "Description",
        "FW_NetRtl",
        "FW_Units",
        "UnitCost",
        "COGS",
        "GrossMargin",
        "GrossMarginPct",
    ]
    for c in keep_cols:
        if c not in merged.columns:
            merged[c] = None
    return merged[keep_cols]


def render_kpi_tiles(fact_sales: pd.DataFrame, fiscal_week: int):
    wk = fact_sales[fact_sales["FiscalWeek"] == fiscal_week]
    units = wk["FW_Units"].sum()
    units_ly = wk["FW_Units_LY"].sum()
    sales = wk["FW_NetRtl"].sum()
    sales_ly = wk["FW_NetRtl_LY"].sum()
    units_yoy = (units - units_ly) / units_ly if units_ly else None
    sales_yoy = (sales - sales_ly) / sales_ly if sales_ly else None
    cols = st.columns(4)
    cols[0].metric(f"Total Units FW{fiscal_week}", f"{units:,.0f}")
    cols[1].metric(f"Total Net Revenue FW{fiscal_week}", f"${sales:,.0f}")
    cols[2].metric("Units YoY%", f"{units_yoy:.1%}" if units_yoy is not None else "NA")
    cols[3].metric("Net Revenue YoY%", f"{sales_yoy:.1%}" if sales_yoy is not None else "NA")


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Innova x AZ Sales Dashboard", layout="wide")
st.title("Innova x AZ Sales Dashboard")
st.caption("Actual-only view built from weekly Innova vendor reports (FW12 & FW13).")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 14px 10px;
        color: #111827;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    div[data-testid="stMetric"] * {
        color: inherit !important;
    }
    @media (prefers-color-scheme: dark) {
        div[data-testid="stMetric"] {
            background-color: #1f2937;
            border: 1px solid #374151;
            color: #f9fafb;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.45);
        }
        div[data-testid="stMetric"] * {
            color: #f9fafb !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

fact_sales = load_sales_fact()
units_yoy = load_units_yoy()
store_counts = load_store_counts()
cost_dim = load_cost_dim()
billbacks = load_billbacks()
products = build_product_dim(fact_sales)

if fact_sales.empty:
    st.error("No sales data found. Make sure the Innova-AZ FY26FWxx files are in the app directory.")
    st.stop()

available_weeks = sorted(fact_sales["FiscalWeek"].unique())
default_week = max(available_weeks)

with st.sidebar:
    st.header("Filters")
    fiscal_week = st.selectbox("Fiscal Week", available_weeks, index=available_weeks.index(default_week))
    top_n = st.slider("Top N products by L52 Net Revenue", min_value=5, max_value=50, value=20, step=5)
    part_options = ["All Parts"] + sorted(fact_sales["PartNumber"].dropna().unique())
    selected_parts = st.multiselect(
        "Filter by Part Number",
        options=part_options,
        default=["All Parts"],
    )
    st.markdown("Data source: TY-LY, Units YOY, Historical Store Counts, Cost File, Accntg-Billbacks")


tab_products, tab_overview, tab_store, tab_profit = st.tabs(
    ["Product Performance", "Sales Executive Overview", "Store Counts", "Profitability & Billbacks"]
)


# ----- Product Performance -----
with tab_products:
    st.subheader("KPI Snapshot (Current Week)")
    render_kpi_tiles(fact_sales, fiscal_week)

    # Top contributors callouts
    st.markdown("**Top Contributors**")
    wk_all = fact_sales[fact_sales["FiscalWeek"] == fiscal_week].copy()
    if selected_parts and "All Parts" not in selected_parts:
        wk_all = wk_all[wk_all["PartNumber"].isin(selected_parts)]
    top_rev = wk_all.sort_values("FW_NetRtl", ascending=False).head(1)
    top_units_growth = (
        wk_all.assign(Units_YoYPct=lambda df: (df["FW_Units"] - df["FW_Units_LY"]) / df["FW_Units_LY"])
        .replace([pd.NA, pd.NaT, pd.NA], None)
    )
    top_units_growth = top_units_growth[top_units_growth["FW_Units_LY"] > 0]
    pos_growth = top_units_growth.sort_values("Units_YoYPct", ascending=False).head(1)
    neg_growth = top_units_growth.sort_values("Units_YoYPct", ascending=True).head(1)
    tcols = st.columns(3)
    if not top_rev.empty:
        row = top_rev.iloc[0]
        tcols[0].metric(
            "Highest Net Revenue Product",
            f"{row['PartNumber']}",
            delta=f"${row['FW_NetRtl']:,.0f}",
            help=f"{row.get('Description','')}",
        )
    if not pos_growth.empty:
        row = pos_growth.iloc[0]
        tcols[1].metric(
            "Strongest Units YoY Growth",
            f"{row['PartNumber']}",
            delta=f"{row['Units_YoYPct']:.1%}",
            help=f"{row.get('Description','')}",
        )
    if not neg_growth.empty:
        row = neg_growth.iloc[0]
        tcols[2].metric(
            "Largest Units YoY Decline",
            f"{row['PartNumber']}",
            delta=f"{row['Units_YoYPct']:.1%}",
            help=f"{row.get('Description','')}",
        )

    st.subheader("Product Ranking (Current Week)")
    wk = fact_sales[fact_sales["FiscalWeek"] == fiscal_week].copy()
    if selected_parts and "All Parts" not in selected_parts:
        wk = wk[wk["PartNumber"].isin(selected_parts)]
    wk["Units_YoYPct"] = (wk["FW_Units"] - wk["FW_Units_LY"]) / wk["FW_Units_LY"]
    wk["Sales_YoYPct"] = (wk["FW_NetRtl"] - wk["FW_NetRtl_LY"]) / wk["FW_NetRtl_LY"]
    wk = wk.sort_values(by="L52_NetRtl", ascending=False).head(top_n).reset_index(drop=True)
    wk["Rank"] = wk.index + 1
    cols_to_show = [
        "Rank",
        "PartNumber",
        "ItemNumber",
        "Description",
        "FW_Units",
        "FW_NetRtl",
        "L52_Units",
        "L52_NetRtl",
        "Units_YoYPct",
        "Sales_YoYPct",
    ]
    display_df = wk[cols_to_show].rename(
        columns={
            "FW_Units": "FW Units",
            "FW_NetRtl": "FW Net Revenue",
            "L52_Units": "L52 Units",
            "L52_NetRtl": "L52 Net Revenue",
            "Units_YoYPct": "Units YoY%",
            "Sales_YoYPct": "Net Revenue YoY%",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Units Trend by Product (FW12 → FW13)")
    trend = fact_sales[fact_sales["PartNumber"].isin(wk["PartNumber"].unique())]
    trend_units = (
        trend.groupby(["FiscalWeek", "PartNumber"])[["FW_Units"]]
        .sum()
        .reset_index()
    )

    line_units = (
        alt.Chart(trend_units)
        .mark_line(point=True)
        .encode(
            x=alt.X("FiscalWeek:O", title="Fiscal Week"),
            y=alt.Y("FW_Units:Q", title="Units"),
            color=alt.Color("PartNumber:N"),
        )
    )
    st.altair_chart(line_units, use_container_width=True)
    if not trend_units.empty:
        latest = trend_units[trend_units["FiscalWeek"] == trend_units["FiscalWeek"].max()]
        msg = f"Units trend shows FW{int(trend_units['FiscalWeek'].min())} → FW{int(trend_units['FiscalWeek'].max())}; top products remain consistent week-over-week."
        st.caption(msg)

    st.subheader("Net Revenue Trend by Product (FW12 → FW13)")
    trend_sales = (
        trend.groupby(["FiscalWeek", "PartNumber"])[["FW_NetRtl"]]
        .sum()
        .reset_index()
    )
    line_sales = (
        alt.Chart(trend_sales)
        .mark_line(point=True)
        .encode(
            x=alt.X("FiscalWeek:O", title="Fiscal Week"),
            y=alt.Y("FW_NetRtl:Q", title="Net Revenue", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("PartNumber:N"),
            tooltip=[
                alt.Tooltip("PartNumber:N", title="Part"),
                alt.Tooltip("FiscalWeek:O", title="Fiscal Week"),
                alt.Tooltip("FW_NetRtl:Q", title="Net Revenue", format="$,.0f"),
            ],
        )
    )
    st.altair_chart(line_sales, use_container_width=True)
    if not trend_sales.empty:
        msg = "Net revenue lines show short-term week move for the selected top products."
        st.caption(msg)


# ----- Sales Executive Overview -----
with tab_overview:
    st.subheader("Units YOY Trend (All Years)")
    if units_yoy.empty:
        st.info("Units YOY sheet is empty or missing.")
    else:
        chart_df = units_yoy.copy()
        chart = (
            alt.Chart(chart_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("FiscalWeek:O", title="Fiscal Week"),
                y=alt.Y("TotalUnits:Q", title="Units"),
                color=alt.Color("FiscalYear:N"),
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
        # One-line read
        latest_year = chart_df["FiscalYear"].max()
        latest_week = chart_df["FiscalWeek"].max()
        latest_units = (
            chart_df[(chart_df["FiscalYear"] == latest_year) & (chart_df["FiscalWeek"] == latest_week)]["TotalUnits"]
            .sum()
        )
        st.caption(f"Latest data point: FY{latest_year} FW{latest_week} at {latest_units:,.0f} units.")

    st.subheader("FW12 vs FW13 Comparison (All Products)")
    ww = (
        fact_sales.groupby("FiscalWeek")[["FW_Units", "FW_NetRtl"]]
        .sum()
        .reset_index()
    )
    bar_units = (
        alt.Chart(ww)
        .mark_bar()
        .encode(
            x=alt.X("FiscalWeek:O", title="Fiscal Week"),
            y=alt.Y("FW_Units:Q", title="Units"),
            tooltip=[alt.Tooltip("FiscalWeek:O", title="Fiscal Week"), alt.Tooltip("FW_Units:Q", title="Units", format=",")],
        )
        .properties(title="Units")
    )
    bar_revenue = (
        alt.Chart(ww)
        .mark_bar()
        .encode(
            x=alt.X("FiscalWeek:O", title="Fiscal Week"),
            y=alt.Y("FW_NetRtl:Q", title="Net Revenue", axis=alt.Axis(format="$,.0f")),
            tooltip=[
                alt.Tooltip("FiscalWeek:O", title="Fiscal Week"),
                alt.Tooltip("FW_NetRtl:Q", title="Net Revenue", format="$,.0f"),
            ],
        )
        .properties(title="Net Revenue")
    )
    st.altair_chart(bar_units, use_container_width=True)
    st.altair_chart(bar_revenue, use_container_width=True)
    if len(ww) >= 2:
        w12 = ww.loc[ww["FiscalWeek"] == 12, "FW_NetRtl"].sum()
        w13 = ww.loc[ww["FiscalWeek"] == 13, "FW_NetRtl"].sum()
        delta = (w13 - w12) / w12 if w12 else None
        st.caption(
            f"FW13 net revenue: ${w13:,.0f} vs FW12: ${w12:,.0f}"
            + (f" ({delta:+.1%} change)" if delta is not None else "")
        )


# ----- Store Counts -----
with tab_store:
    # Top 3 products by units/store (names only)
    ups_top = agg_units_per_store(fact_sales, store_counts)
    if ups_top.empty:
        st.info("Store counts unavailable to derive top products.")
    else:
        filtered_top = ups_top[ups_top["FiscalWeek"].isin([12, 13])]
        top_products = (
            filtered_top.groupby("PartNumber")["UnitsPerStore"]
            .mean()
            .sort_values(ascending=False)
            .head(3)
        )
        tcols = st.columns(3)
        for idx, name in enumerate(top_products.index):
            tcols[idx].metric(f"Top Units/Store #{idx+1}", name)

    st.subheader("Store Count Trend")
    if store_counts.empty:
        st.info("Historical Store Counts sheet is empty or missing.")
    else:
        store_weekly = (
            store_counts.groupby(["FiscalYear", "FiscalWeek"])["StoreCount"]
            .sum()
            .reset_index()
        )
        chart = (
            alt.Chart(store_weekly)
            .mark_line(point=True)
            .encode(
                x=alt.X("FiscalWeek:O", title="Fiscal Week"),
                y=alt.Y("StoreCount:Q", title="Store Count"),
                color=alt.Color("FiscalYear:N"),
            )
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Units per Store (FW12 vs FW13)")
    ups = agg_units_per_store(fact_sales, store_counts)
    if ups.empty:
        st.info("Cannot compute Units per Store without store counts.")
    else:
        filtered = ups[ups["FiscalWeek"].isin([12, 13])]
        if selected_parts and "All Parts" not in selected_parts:
            filtered = filtered[filtered["PartNumber"].isin(selected_parts)]
        top_products = (
            filtered.groupby("PartNumber")["FW_NetRtl"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )
        filtered = filtered[filtered["PartNumber"].isin(top_products)]
        long = filtered[["PartNumber", "FiscalWeek", "UnitsPerStore"]]
        chart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                x=alt.X("PartNumber:N", title="Part"),
                y=alt.Y("UnitsPerStore:Q", title="Units per Store"),
                color=alt.Color("FiscalWeek:O", title="Fiscal Week"),
                tooltip=[
                    alt.Tooltip("PartNumber:N", title="Part"),
                    alt.Tooltip("FiscalWeek:O", title="Fiscal Week"),
                    alt.Tooltip("UnitsPerStore:Q", title="Units per Store", format=",.4f"),
                ],
            )
        )
        st.altair_chart(chart, use_container_width=True)
        if not filtered.empty:
            w12 = filtered[filtered["FiscalWeek"] == 12]["UnitsPerStore"].mean()
            w13 = filtered[filtered["FiscalWeek"] == 13]["UnitsPerStore"].mean()
            delta = (w13 - w12) / w12 if w12 else None
            st.caption(
                f"Average units/store moved from {w12:.4f} (FW12) to {w13:.4f} (FW13)"
                + (f" ({delta:+.1%})" if delta is not None else "")
            )
        # Key metrics summary
        top_avg = (
            filtered.groupby("PartNumber")["UnitsPerStore"]
            .mean()
            .sort_values(ascending=False)
            .head(1)
        )
        overall_avg = filtered["UnitsPerStore"].mean()
        cols = st.columns(3)
        if not filtered.empty:
            cols[0].metric("Overall Avg Units/Store", f"{overall_avg:.3f}")
        if not top_avg.empty:
            top_names = top_avg.head(3)
            tcols = cols[1].columns(len(top_names))
            for idx, (name, _) in enumerate(top_names.items()):
                tcols[idx].metric(f"Top Units/Store #{idx+1}", name)
        if not filtered.empty:
            w13_avg = filtered[filtered["FiscalWeek"] == 13]["UnitsPerStore"].mean()
            cols[2].metric("Latest Week Units/Store", f"{w13_avg:.3f}", help="FW13 average")


# ----- Profitability & Billbacks -----
with tab_profit:
    st.subheader(f"Margin by Product (FW{fiscal_week})")
    margin = compute_margin(fact_sales, cost_dim, fiscal_week)
    if margin.empty:
        st.info("Cost data not available to compute margins.")
    else:
        total_net_rev = margin["FW_NetRtl"].sum()
        total_cogs = margin["COGS"].sum()
        total_gm = margin["GrossMargin"].sum()
        gm_pct = total_gm / total_net_rev if total_net_rev else None

        cols = st.columns(4)
        cols[0].metric("Net Revenue", f"${total_net_rev:,.0f}")
        cols[1].metric("COGS", f"${total_cogs:,.0f}")
        cols[2].metric("Gross Margin", f"${total_gm:,.0f}")
        cols[3].metric("Gross Margin %", f"{gm_pct:.1%}" if gm_pct else "NA")

        top_gm = margin.sort_values(by="GrossMargin", ascending=False).head(1)
        top_gm_pct = margin[margin["GrossMarginPct"].notna()].sort_values(
            by="GrossMarginPct", ascending=False
        ).head(1)
        st.markdown("**Top Contributors**")
        tcols = st.columns(2)
        if not top_gm.empty:
            row = top_gm.iloc[0]
            tcols[0].metric(
                "Top Gross Margin Product",
                f"{row['PartNumber']}",
                help=f"{row.get('Description','')}",
                delta=f"${row['GrossMargin']:,.0f}",
            )
        if not top_gm_pct.empty:
            row = top_gm_pct.iloc[0]
            tcols[1].metric(
                "Best GM% Product",
                f"{row['PartNumber']}",
                help=f"{row.get('Description','')}",
                delta=f"{row['GrossMarginPct']:.1%}",
            )

        display_cols = [
            "Rank",
            "PartNumber",
            "ItemNumber",
            "Description",
            "FW_NetRtl",
            "COGS",
            "GrossMargin",
            "GrossMarginPct",
        ]
        margin = margin.sort_values(by="GrossMargin", ascending=False).reset_index(drop=True)
        margin["Rank"] = margin.index + 1
        st.dataframe(
            margin[display_cols].rename(
                columns={
                    "FW_NetRtl": "Net Revenue",
                    "COGS": "COGS",
                    "GrossMargin": "Gross Margin",
                    "GrossMarginPct": "Gross Margin %",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Billbacks Summary")
    if billbacks.empty:
        st.info("Billbacks data not available.")
    else:
        agg = (
            billbacks.groupby("YearMonth")["BillbackAmount"]
            .sum()
            .reset_index()
            .rename(columns={"BillbackAmount": "TotalBillbacks"})
        )
        bar = (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                x=alt.X("YearMonth:T", title="Month"),
                y=alt.Y("TotalBillbacks:Q", title="Total Billbacks", axis=alt.Axis(format="$,.0f")),
                tooltip=[
                    alt.Tooltip("YearMonth:T", title="Month"),
                    alt.Tooltip("TotalBillbacks:Q", title="Total Billbacks", format="$,.0f"),
                ],
            )
        )
        st.altair_chart(bar, use_container_width=True)

st.caption("Add new FY26FWxx files to extend the time series; the dashboard will automatically ingest them.")
