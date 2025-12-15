import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st


# ---------- Paths & constants ----------
DATA_DIR = Path(__file__).parent
FILE_GLOB = "Innova-AZ FY26FW*.xlsx"
CATEGORY_MAPPING_PATH = DATA_DIR / "category_mapping.csv"
BILLBACK_REASON_PATH = DATA_DIR / "billback_reason_mapping.csv"
CATEGORY_MAPPING_XLSX = DATA_DIR / "Product_Full_List.xlsx"
BILLBACK_REASON_XLSX = DATA_DIR / "Fee_Code_List.xlsx"
SUMMARY_SHEET_NAME = "Summary Rtl L52FW"
EXEC_MIN_LY_REV_DEFAULT = 100.0

SALES_COLUMNS = {
    "POV Number": "pov_number",
    "POV": "pov_number",
    "Part": "part_number",
    "Part Number": "part_number",
    "Item": "item_id",
    "Item ID": "item_id",
    "Item Number": "item_id",
    "Description": "description",
    "Store Count": "store_count",
    "FW Units": "units",
    "FW Net Rtl": "revenue",
    "LY FW Units": "ly_units",
    "LYFW Net Rtl": "ly_revenue",
    "L4FW Net Units": "l4w_units",
    "L4FW Net Rtl": "l4w_revenue",
    "L52FW Net Units": "l52w_units",
    "L52FW Net Rtl$": "l52w_revenue",
    "FYTD Net Rtl": "fytd_revenue",
    "FYTD Rtl$ Diff %": "fytd_revenue_diff_pct",
    "FYTD Rtl$ Diff%": "fytd_revenue_diff_pct",
}

alt.data_transformers.disable_max_rows()


# ---------- Helpers ----------
def get_source_files() -> List[Path]:
    return sorted(DATA_DIR.glob(FILE_GLOB))


def parse_fiscal_from_name(filename: str) -> Tuple[Optional[int], Optional[int]]:
    match = re.search(r"FY(\d{2})FW(\d+)", filename)
    if not match:
        return None, None
    return 2000 + int(match.group(1)), int(match.group(2))


def parse_week_code(value) -> Tuple[Optional[int], Optional[int]]:
    """Parse a 6-digit fiscal code like 202612 -> (2026, 12)."""
    if pd.isna(value):
        return None, None
    s = re.sub(r"\D", "", str(value))
    if len(s) < 6:
        return None, None
    fiscal_year = int(s[:4])
    fiscal_week = int(s[4:6])
    return fiscal_year, fiscal_week


def numericize(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def coerce_str_safe(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Convert columns to string dtype without turning NaN into 'nan'."""
    for col in cols:
        if col in df.columns:
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = series.astype("string").str.strip()
            series = series.replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
            df[col] = series
    return df


# Backward compatibility alias
def coerce_str(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    return coerce_str_safe(df, cols)


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate-named columns keeping the first occurrence."""
    return df.loc[:, ~df.columns.duplicated()]


def column_as_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column as Series even if duplicated name created a DataFrame."""
    if col not in df.columns:
        return pd.Series(dtype=object)
    val = df[col]
    if isinstance(val, pd.DataFrame):
        return val.iloc[:, 0]
    return val


def ensure_series_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns and isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
    df = drop_duplicate_columns(df)
    return df


def format_currency_columns(df: pd.DataFrame, cols: Iterable[str], decimals: int = 0) -> pd.DataFrame:
    formatted = df.copy()
    fmt = f"${{0:,.{decimals}f}}"
    for c in cols:
        if c in formatted.columns:
            formatted[c] = formatted[c].apply(lambda x: fmt.format(x) if pd.notna(x) else x)
    return formatted


def round_half_up(val: Optional[float]) -> Optional[int]:
    if val is None or pd.isna(val):
        return None
    return int(Decimal(str(val)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def format_currency_0(val: Optional[float]) -> str:
    if val is None or pd.isna(val):
        return "N/A"
    rounded = round_half_up(val)
    return f"${rounded:,}" if rounded is not None else "N/A"


def format_percent_columns(df: pd.DataFrame, cols: Iterable[str], decimals: int = 1) -> pd.DataFrame:
    formatted = df.copy()
    fmt = f"{{0:.{decimals}%}}"
    for c in cols:
        if c in formatted.columns:
            formatted[c] = formatted[c].apply(lambda x: fmt.format(x) if pd.notna(x) else x)
    return formatted


def clean_sku_value(val: object):
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()
    if s.endswith(".0") and s.replace(".", "").isdigit():
        s = s[:-2]
    return s


def normalize_fee_code(val: object) -> str:
    if pd.isna(val):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(val).upper())


def normalize_sku_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    na_tokens = {"<NA>", "NA", "N/A", "NONE", "NULL", "", "NAN"}
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(clean_sku_value)
            df[c] = df[c].astype("string")
            df[c] = df[c].str.upper()
            df[c] = df[c].replace(na_tokens, pd.NA)
            df[c] = df[c].str.strip()
    return df


def drop_invalid_products(df: pd.DataFrame, drop_blank_part: bool = True) -> pd.DataFrame:
    if df.empty:
        return df
    na_tokens = {"<NA>", "NA", "N/A", "NONE", "NULL", ""}
    for c in ["part_number", "item_id"]:
        if c in df.columns:
            df[c] = df[c].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
            df[c] = df[c].replace(na_tokens, pd.NA)
    if drop_blank_part and "part_number" in df.columns:
        df = df.dropna(subset=["part_number"])
    elif not drop_blank_part and "part_number" in df.columns and "item_id" in df.columns:
        # Keep blank parts but drop rows missing both identifiers.
        df = df.dropna(subset=["part_number", "item_id"], how="all")
    return df


def find_header_row(df: pd.DataFrame, keywords: Iterable[str], search_rows: int = 25) -> Optional[int]:
    lowered = [k.lower() for k in keywords]
    for idx in range(min(search_rows, len(df))):
        row = df.iloc[idx].astype(str).str.lower().tolist()
        if any(k in " ".join(row) for k in lowered):
            return idx
    return None


def ensure_week_cols(df: pd.DataFrame, week_col: str = "fiscal_week") -> pd.DataFrame:
    if week_col in df.columns:
        df[week_col] = pd.to_numeric(df[week_col], errors="coerce").astype("Int64")
    return df


def file_signature(files: List[Path]) -> Tuple[Tuple[str, float], ...]:
    return tuple((f.name, f.stat().st_mtime) for f in files)


# ---------- Loaders ----------
@st.cache_data(show_spinner=False)
def load_sales_actual_tyly(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    frames = []
    for path in files:
        fiscal_year, fiscal_week = parse_fiscal_from_name(path.name)
        if fiscal_year is None:
            continue
        raw = pd.read_excel(path, sheet_name="TY-LY", header=None)
        header_row = find_header_row(raw, ["Part", "Item", "FW Units"], search_rows=8) or 3
        df = pd.read_excel(path, sheet_name="TY-LY", header=header_row)
        df = df.rename(columns=SALES_COLUMNS)
        df = drop_duplicate_columns(df)
        df = df[[c for c in SALES_COLUMNS.values() if c in df.columns]]
        df = ensure_series_columns(df, ["part_number", "item_id", "description"])
        df = coerce_str_safe(df, ["part_number", "item_id", "description"])
        df = normalize_sku_columns(df, ["part_number", "item_id"])
        df["fiscal_year"] = fiscal_year
        df["fiscal_week"] = fiscal_week
        df = numericize(
            df,
            [
                "store_count",
                "units",
                "revenue",
                "ly_units",
                "ly_revenue",
                "l4w_units",
                "l4w_revenue",
                "l52w_units",
                "l52w_revenue",
                "fytd_revenue",
                "fytd_revenue_diff_pct",
            ],
        )
        df = drop_invalid_products(df, drop_blank_part=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_units_yoy(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    if not files:
        return pd.DataFrame()
    latest = files[-1]
    df = pd.read_excel(latest, sheet_name="Units YOY", header=15)
    df = df.rename(columns={"FW": "fiscal_week"})
    df = df.dropna(subset=["fiscal_week"])
    long = df.melt(id_vars=["fiscal_week"], var_name="fiscal_year", value_name="total_units")
    long["fiscal_year"] = pd.to_numeric(long["fiscal_year"], errors="coerce").astype("Int64")
    long["fiscal_week"] = pd.to_numeric(long["fiscal_week"], errors="coerce").astype("Int64")
    long["total_units"] = pd.to_numeric(long["total_units"], errors="coerce")
    long = long.dropna(subset=["fiscal_year", "fiscal_week", "total_units"])
    long["fiscal_year"] = long["fiscal_year"].astype(int)
    long["fiscal_week"] = long["fiscal_week"].astype(int)
    return long


@st.cache_data(show_spinner=False)
def load_store_counts(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
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
        var_name="fiscal_label",
        value_name="store_count",
    )
    fiscal = long["fiscal_label"].str.extract(r"FY(\d{2})FW(\d{2})")
    long["fiscal_year"] = fiscal[0].astype(float)
    long["fiscal_week"] = fiscal[1].astype(float)
    long["fiscal_year"] = long["fiscal_year"].apply(lambda x: 2000 + int(x) if pd.notna(x) else None)
    long["fiscal_week"] = long["fiscal_week"].apply(lambda x: int(x) if pd.notna(x) else None)
    long["store_count"] = pd.to_numeric(long["store_count"], errors="coerce")
    long = long.dropna(subset=["fiscal_year", "fiscal_week", "store_count"])
    long = coerce_str_safe(long, ["Part Number", "Item"])
    long = normalize_sku_columns(long, ["Part Number", "Item"])
    return long.rename(columns={"Part Number": "part_number", "Item": "item_id"})


@st.cache_data(show_spinner=False)
def load_forecast(files_sig: Tuple[Tuple[str, float], ...]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = [DATA_DIR / name for name, _ in files_sig]
    fc_rows: List[Dict[str, float]] = []
    op_rows: List[Dict[str, float]] = []

    for path in files:
        fiscal_year_file, fiscal_week_file = parse_fiscal_from_name(path.name)
        raw = pd.read_excel(path, sheet_name="forecast", header=None)
        if raw.empty:
            continue
        block_row = raw.iloc[10].ffill()
        week_row = raw.iloc[13]
        df = pd.read_excel(path, sheet_name="forecast", header=13)
        item_col = "Item ID"
        if item_col not in df.columns:
            continue
        gt_mask = df[item_col].astype(str).str.strip().str.lower().eq("grand total")
        if not gt_mask.any():
            continue
        gt_idx = gt_mask.idxmax()

        for idx, col_name in enumerate(df.columns):
            week_code = week_row.iloc[idx]
            fiscal_year, fiscal_week = parse_week_code(week_code)
            if fiscal_year is None or fiscal_week is None or col_name in ["Item ID", "Part Number", "Item"]:
                continue

            block_label = str(block_row.iloc[idx]).lower() if not pd.isna(block_row.iloc[idx]) else ""
            val = pd.to_numeric(df.loc[gt_idx, col_name], errors="coerce")
            if pd.isna(val):
                continue

            if "order projection" in block_label:
                op_rows.append(
                    {
                        "snapshot_year": fiscal_year_file,
                        "snapshot_week": fiscal_week_file,
                        "target_year": fiscal_year,
                        "target_week": fiscal_week,
                        "order_units": val,
                    }
                )
            elif "sales forecast" in block_label or "forecast" in block_label:
                fc_rows.append(
                    {
                        "snapshot_year": fiscal_year_file,
                        "snapshot_week": fiscal_week_file,
                        "target_year": fiscal_year,
                        "target_week": fiscal_week,
                        "forecast_units": val,
                    }
                )

    forecast_df = pd.DataFrame(fc_rows)
    order_df = pd.DataFrame(op_rows)
    return forecast_df, order_df


@st.cache_data(show_spinner=False)
def load_cpfr(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    records: List[Dict[str, float]] = []
    for path in files:
        snapshot_year, snapshot_week = parse_fiscal_from_name(path.name)
        raw = pd.read_excel(path, sheet_name="CPFR", header=None)
        if raw.empty or len(raw) < 13:
            continue
        week_row = raw.iloc[8]  # row 9 in Excel
        fill_row = raw.iloc[9]  # row 10
        shipped_row = raw.iloc[10]  # row 11
        requested_row = raw.iloc[11]  # row 12
        not_shipped_row = raw.iloc[12]  # row 13
        for col_idx, week_code in enumerate(week_row):
            fiscal_year, fiscal_week = parse_week_code(week_code)
            if fiscal_year is None or fiscal_week is None:
                continue
            fill_val = pd.to_numeric(fill_row.iloc[col_idx], errors="coerce")
            shipped_val = pd.to_numeric(shipped_row.iloc[col_idx], errors="coerce")
            requested_val = pd.to_numeric(requested_row.iloc[col_idx], errors="coerce")
            not_shipped_val = pd.to_numeric(not_shipped_row.iloc[col_idx], errors="coerce")
            records.append(
                {
                    "snapshot_year": snapshot_year,
                    "snapshot_week": snapshot_week,
                    "fiscal_year": fiscal_year,
                    "fiscal_week": fiscal_week,
                    "fill_rate": fill_val,
                    "shipped_units": shipped_val,
                    "requested_units": requested_val,
                    "not_shipped_units": not_shipped_val,
                }
            )
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def load_cpfr_detail(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    frames = []
    for path in files:
        snapshot_year, snapshot_week = parse_fiscal_from_name(path.name)
        raw = pd.read_excel(path, sheet_name="CPFR", header=None)
        if raw.empty:
            continue
        header_row = find_header_row(raw, ["POV"], search_rows=25)
        if header_row is None:
            continue
        header = raw.iloc[header_row]
        weeks = header.iloc[1:]
        data = raw.iloc[header_row + 1 :]
        part_col = data.columns[0]
        requested_block = find_header_row(raw, ["Sum of REQUESTED"], search_rows=header_row + 10)
        shipped_block = find_header_row(raw, ["Sum of ADJSHIP"], search_rows=header_row + 10)
        for col_idx, week_code in enumerate(weeks, start=1):
            fiscal_year, fiscal_week = parse_week_code(week_code)
            if fiscal_year is None:
                continue
            subset = data[[part_col, col_idx]].copy()
            subset = subset.rename(columns={part_col: "part_number", col_idx: "shipped_units"})
            subset["fiscal_year"] = fiscal_year
            subset["fiscal_week"] = fiscal_week
            subset["snapshot_year"] = snapshot_year
            subset["snapshot_week"] = snapshot_week
            subset = numericize(subset, ["shipped_units"])
            if requested_block is not None and (requested_block + 1) < len(raw):
                req_val = pd.to_numeric(raw.iloc[requested_block + 1, col_idx], errors="coerce")
                subset["requested_units"] = req_val
            if shipped_block is not None and (shipped_block + 1) < len(raw):
                adj_val = pd.to_numeric(raw.iloc[shipped_block + 1, col_idx], errors="coerce")
                subset["shipped_units"] = adj_val if pd.notna(adj_val) else subset["shipped_units"]
            subset["not_shipped_units"] = subset.get("requested_units", pd.Series(dtype=float)) - subset.get(
                "shipped_units", pd.Series(dtype=float)
            )
            frames.append(subset)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = coerce_str_safe(df, ["part_number"])
    df = normalize_sku_columns(df, ["part_number"])
    df = drop_invalid_products(df)
    return df

@st.cache_data(show_spinner=False)
def load_redflags(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    frames: List[pd.DataFrame] = []

    for path in files:
        try:
            raw = pd.read_excel(path, sheet_name="Fill Rate Red Flags", header=None)
        except Exception:
            continue

        # FY26FW13 expectation: rows 10–46 exist; columns A–F exist (A,D,E,F used)
        if raw.shape[0] < 10 or raw.shape[1] < 6:
            st.warning(
                f"Fill Rate Red Flags sheet is missing expected rows/columns in {path.name}. "
                f"Expected >=10 rows and >=6 columns (A–F). Got shape={raw.shape}."
            )
            continue

        end_row = min(46, raw.shape[0])  # Excel row 46 (1-based), pandas end is exclusive
        header_row_idx = 8  # Excel row 9 (1-based)
        hdr = raw.iloc[header_row_idx].astype(str).str.upper().fillna("")

        def _col_idx(pattern: str, fallback: int) -> int:
            hits = [i for i, v in enumerate(hdr) if re.search(pattern, v)]
            return hits[0] if hits else fallback

        # Prefer header match; fallback to fixed A,D,E,F positions for FY26FW13
        col_part = _col_idx(r"\bPART\s*NUMBER\b", 0)
        col_lfw = _col_idx(r"\bLFW\b.*NOT\s*SHIP", 3)
        col_l4w = _col_idx(r"\bL4FW\b.*NOT\s*SHIP", 4)
        col_l52w = _col_idx(r"\bL52FW\b.*NOT\s*SHIP", 5)  # FY26FW13 uses column F

        needed = [col_part, col_lfw, col_l4w, col_l52w]
        if max(needed) >= raw.shape[1]:
            st.warning(
                f"Fill Rate Red Flags sheet columns out of range in {path.name}. "
                f"Needed indices {needed} but sheet has {raw.shape[1]} columns."
            )
            continue

        df = raw.iloc[9:end_row, needed].copy()  # Excel row 10 -> pandas index 9
        df.columns = ["part_number", "not_shipped_lfw", "not_shipped_l4w", "not_shipped_l52w"]

        # Clean + types
        df["part_number"] = df["part_number"].ffill()
        df = coerce_str_safe(df, ["part_number"])
        df = normalize_sku_columns(df, ["part_number"])
        df["not_shipped_lfw"] = pd.to_numeric(df["not_shipped_lfw"], errors="coerce")
        df["not_shipped_l4w"] = pd.to_numeric(df["not_shipped_l4w"], errors="coerce")
        df["not_shipped_l52w"] = pd.to_numeric(df["not_shipped_l52w"], errors="coerce")

        # Drop TOTAL/subtotals and empty rows
        df = df[~df["part_number"].astype(str).str.contains("TOTAL", case=False, na=False)]
        df = df.dropna(subset=["part_number"], how="all")
        df = df[df[["not_shipped_lfw", "not_shipped_l4w", "not_shipped_l52w"]].notna().any(axis=1)]

        snapshot_year, snapshot_week = parse_fiscal_from_name(path.name)
        df["snapshot_year"] = snapshot_year
        df["snapshot_week"] = snapshot_week

        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_outs(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    frames = []
    for path in files:
        fiscal_year, fiscal_week = parse_fiscal_from_name(path.name)
        raw = pd.read_excel(path, sheet_name="Outs", header=None)
        header_row = find_header_row(raw, ["STORES OUT OF STOCK", "ITEM NUMBER"], search_rows=20) or 11
        df = pd.read_excel(path, sheet_name="Outs", header=header_row)
        df = df.rename(
            columns={
                "ITEM NUMBER": "item_id",
                "PART NUMBER": "part_number",
                "ITEM DESCRIPTION": "description",
                "STORE COUNT": "store_count",
                "STORES OUT OF STOCK": "store_oos_exposure",
            }
        )
        df = coerce_str_safe(df, ["item_id", "part_number", "description"])
        df = normalize_sku_columns(df, ["item_id", "part_number"])
        df = numericize(df, ["store_count", "store_oos_exposure"])
        df = df.dropna(subset=["item_id", "part_number"], how="all")
        df["fiscal_year"] = fiscal_year
        df["fiscal_week"] = fiscal_week
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined = combined[~combined["part_number"].str.contains("TOTAL", case=False, na=False)]
    combined = combined[~combined["item_id"].str.contains("TOTAL", case=False, na=False)]
    combined = drop_invalid_products(combined)
    return combined


@st.cache_data(show_spinner=False)
def load_outs_totals(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    records: List[Dict[str, float]] = []
    for path in files:
        fiscal_year, fiscal_week = parse_fiscal_from_name(path.name)
        try:
            raw = pd.read_excel(path, sheet_name="Outs", header=None)
        except Exception:
            continue
        if raw.shape[0] <= 91 or raw.shape[1] <= 9:
            continue
        val = pd.to_numeric(raw.iloc[91, 9], errors="coerce")
        records.append({"fiscal_year": fiscal_year, "fiscal_week": fiscal_week, "store_oos_total": val})
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def load_inventory(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    frames = []
    for path in files:
        fiscal_year, fiscal_week = parse_fiscal_from_name(path.name)
        raw = pd.read_excel(path, sheet_name="Inventory", header=None)
        header_row = find_header_row(raw, ["MINOR DEPT", "AVG WKS"], search_rows=25) or 14
        df = pd.read_excel(path, sheet_name="Inventory", header=header_row)
        df = df.rename(columns={"MINOR DEPT": "minor_dept"})
        # Weeks-on-hand column has a verbose name; pick the one containing WKS ON HAND
        woh_col = next((c for c in df.columns if isinstance(c, str) and "WKS ON HAND" in c.upper()), None)
        if woh_col:
            df = df.rename(columns={woh_col: "weeks_on_hand"})
        total_inv_col = next((c for c in df.columns if isinstance(c, str) and "TOTAL INVENTORY" in c.upper()), None)
        if total_inv_col:
            df = df.rename(columns={total_inv_col: "total_inventory"})
        df = numericize(df, ["weeks_on_hand", "total_inventory"])
        df = df.dropna(subset=["minor_dept", "weeks_on_hand"], how="all")
        df["fiscal_year"] = fiscal_year
        df["fiscal_week"] = fiscal_week
        df["is_grand_total"] = df["minor_dept"].astype(str).str.contains("grand total", case=False, na=False)
        keep_cols = ["minor_dept", "weeks_on_hand", "fiscal_week"]
        if "total_inventory" in df.columns:
            keep_cols.append("total_inventory")
        keep_cols.extend(["fiscal_year", "is_grand_total"])
        frames.append(df[keep_cols])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_returns(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    frames = []
    for path in files:
        raw = pd.read_excel(path, sheet_name="Returns", header=None)
        header_row = find_header_row(raw, ["Part", "Item", "UNDAMAGED"], search_rows=15) or 8
        df = pd.read_excel(path, sheet_name="Returns", header=header_row)
        df = df.rename(
            columns={
                "Part": "part_number",
                "Item": "item_id",
                "%DAMAGED": "damaged_rate",
                "%UNDAMAGED": "undamaged_rate",
                "L52FW GROSS UNITS": "gross_units",
                "L52FW DMG UNITS": "damaged_units",
                "L52FW UNDMG UNITS": "undamaged_units",
                "L52FW UNDAMG UNITS": "undamaged_units",
                "L52FW UNDAMAGED UNITS": "undamaged_units",
            }
        )
        df = coerce_str_safe(df, ["part_number", "item_id"])
        df = normalize_sku_columns(df, ["part_number", "item_id"])
        df = numericize(df, ["damaged_rate", "undamaged_rate", "gross_units", "damaged_units", "undamaged_units"])
        for col in ["damaged_rate", "undamaged_rate"]:
            if col in df.columns and df[col].max(skipna=True) is not None and df[col].max(skipna=True) > 1:
                df[col] = df[col] / 100
        df = df.dropna(subset=["part_number", "item_id"], how="all")
        fiscal_year, fiscal_week = parse_fiscal_from_name(path.name)
        df["fiscal_year"] = fiscal_year
        df["fiscal_week"] = fiscal_week
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined = drop_invalid_products(combined)
    return combined


@st.cache_data(show_spinner=False)
def load_tyly_grand_total(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    records = []
    for path in files:
        fiscal_year, fiscal_week = parse_fiscal_from_name(path.name)
        raw = pd.read_excel(path, sheet_name="TY-LY", header=None)
        header_row = find_header_row(raw, ["FW Units", "FW Net Rtl"], search_rows=10)
        if header_row is None:
            continue
        df = pd.read_excel(path, sheet_name="TY-LY", header=header_row)
        df.columns = [str(c).strip() for c in df.columns]
        gt_mask = df.iloc[:, 0].astype(str).str.contains("Grand Total", case=False, na=False)
        if not gt_mask.any():
            continue
        gt = df[gt_mask].iloc[0]
        def find_col(options: Iterable[str]) -> Optional[str]:
            opts = [o.lower() for o in options]
            for c in df.columns:
                c_low = str(c).lower()
                if any(o in c_low for o in opts):
                    return c
            return None

        units_col = find_col(["fw units"])
        rev_col = find_col(["fw net rtl"])
        unit_diff_col = find_col(["fw unit diff"])
        rev_diff_col = find_col(["fw rtl diff"])
        ly_rev_col = find_col(["ly fw net rtl", "lyfw net rtl"])
        ly_units_col = find_col(["ly fw units"])

        if units_col is None or rev_col is None:
            continue
        records.append(
            {
                "fiscal_year": fiscal_year,
                "fiscal_week": fiscal_week,
                "fw_units_total": pd.to_numeric(gt[units_col], errors="coerce"),
                "fw_revenue_total": pd.to_numeric(gt[rev_col], errors="coerce"),
                "fw_unit_diff_pct": pd.to_numeric(gt[unit_diff_col], errors="coerce") if unit_diff_col else None,
                "fw_revenue_diff_pct": pd.to_numeric(gt[rev_diff_col], errors="coerce") if rev_diff_col else None,
                "ly_fw_revenue_total": pd.to_numeric(gt[ly_rev_col], errors="coerce") if ly_rev_col else None,
                "ly_fw_units_total": pd.to_numeric(gt[ly_units_col], errors="coerce") if ly_units_col else None,
            }
        )
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def load_summary_rtl_l52fw(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    frames = []
    for path in files:
        fiscal_year, fiscal_week = parse_fiscal_from_name(path.name)
        try:
            df = pd.read_excel(path, sheet_name=SUMMARY_SHEET_NAME, header=0)
        except Exception:
            continue
        if df.empty or df.shape[1] < 2:
            continue
        # Normalize headers
        df.columns = [str(c).strip() for c in df.columns]
        # Find grand total row
        gt_mask = df.iloc[:, 0].astype(str).str.contains("Grand Total", case=False, na=False)
        if not gt_mask.any():
            continue
        gt_row = df[gt_mask].iloc[0]
        # Attempt to find FW Units and FW Net Rtl columns
        units_col = next((c for c in df.columns if "FW Units" in c), None)
        rev_col = next((c for c in df.columns if "FW Net Rtl" in c), None)
        if units_col is None or rev_col is None:
            continue
        frames.append(
            {
                "fiscal_year": fiscal_year,
                "fiscal_week": fiscal_week,
                "fw_units_total": pd.to_numeric(gt_row[units_col], errors="coerce"),
                "fw_revenue_total": pd.to_numeric(gt_row[rev_col], errors="coerce"),
            }
        )
    return pd.DataFrame(frames)


@st.cache_data(show_spinner=False)
def load_cost_dim(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    if not files:
        return pd.DataFrame()
    latest = files[-1]
    df = pd.read_excel(latest, sheet_name="Cost File", header=13)
    base_cols = [c for c in df.columns if isinstance(c, (int, float, str)) and str(c).isdigit()]
    df["unit_cost"] = df[base_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    df = df.rename(columns={"Part Nbr": "part_number", "Item Nbr": "item_id", "Description": "description"})
    df = df[["part_number", "item_id", "description", "unit_cost"]]
    df = df.dropna(subset=["part_number", "item_id"])
    df = numericize(df, ["unit_cost"])
    df = coerce_str_safe(df, ["part_number", "item_id", "description"])
    df = normalize_sku_columns(df, ["part_number", "item_id"])
    df = drop_invalid_products(df)
    return df


@st.cache_data(show_spinner=False)
def load_billbacks(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    """
    Load billbacks from ANY workbook that contains the 'Accntg-Billbacks' sheet.
    This data is calendar invoice-date based and should remain independent from the FW selector.
    """
    files = [DATA_DIR / name for name, _ in files_sig]
    if not files:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []

    for path in files:
        try:
            df_raw = pd.read_excel(path, sheet_name="Accntg-Billbacks", header=None)
        except Exception:
            continue

        if df_raw.empty or df_raw.shape[1] < 4:
            continue

        # Find header row robustly: a row containing "type code" AND "invoice date" AND "total"
        raw_str = df_raw.astype("string").fillna("").apply(lambda s: s.str.strip())

        def row_has_headers(row: pd.Series) -> bool:
            joined = " | ".join([str(x).lower() for x in row.tolist()])
            return ("type code" in joined) and ("invoice date" in joined) and ("total" in joined)

        header_candidates = [i for i in range(len(raw_str)) if row_has_headers(raw_str.iloc[i])]
        if not header_candidates:
            continue

        header_row = int(header_candidates[0])
        cols = df_raw.iloc[header_row].tolist()
        df = df_raw.iloc[header_row + 1 :].copy()
        df.columns = cols

        df = df.rename(
            columns={
                "Type Code": "type_code",
                "Invoice Nbr": "invoice_number",
                "Invoice Date": "invoice_date_raw",
                "Total": "billback_amount",
            }
        )

        if "invoice_date_raw" not in df.columns or "billback_amount" not in df.columns:
            continue

        def parse_invoice_date(val):
            if isinstance(val, pd.Timestamp):
                return val
            if isinstance(val, (int, float)) and not pd.isna(val):
                # Excel serials tend to be small (e.g., 45449); MMDDYYYY numbers are large (e.g., 6062024)
                if float(val) <= 60000:
                    return pd.to_datetime(val, unit="D", origin="1899-12-30", errors="coerce")
                # Treat as MMDDYYYY even if float
                try:
                    digits = str(int(val)).zfill(8)
                    return pd.to_datetime(digits, format="%m%d%Y", errors="coerce")
                except Exception:
                    return pd.NaT
            parsed = pd.to_datetime(val, errors="coerce")
            if pd.notna(parsed):
                return parsed
            digits = re.sub(r"\D", "", str(val))
            if len(digits) == 8:
                return pd.to_datetime(digits, format="%m%d%Y", errors="coerce")
            return pd.NaT

        df["invoice_date"] = df["invoice_date_raw"].apply(parse_invoice_date)
        df["billback_amount"] = pd.to_numeric(df["billback_amount"], errors="coerce")

        # Keep only valid rows
        df = df.dropna(subset=["billback_amount", "invoice_date"])
        if df.empty:
            continue

        df = coerce_str_safe(df, ["type_code", "invoice_number"])
        df["type_code_norm"] = df["type_code"].apply(normalize_fee_code)
        df["fiscal_year"] = df["invoice_date"].dt.year.astype(int)
        df["fiscal_week"] = df["invoice_date"].dt.isocalendar().week.astype(int)

        keep = ["type_code", "type_code_norm", "invoice_number", "invoice_date", "fiscal_year", "fiscal_week", "billback_amount"]
        frames.append(df[keep])

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # De-dupe (some FW files repeat the same 2024 billbacks)
    dedupe_cols = [c for c in ["invoice_number", "type_code_norm", "invoice_date", "billback_amount"] if c in out.columns]
    if dedupe_cols:
        out = out.drop_duplicates(subset=dedupe_cols, keep="first")

    return out


@st.cache_data(show_spinner=False)
def load_category_dim() -> pd.DataFrame:
    """Load category mapping with optional minor_dept bridge fallback."""
    minor_dept_bridge = pd.DataFrame()
    if CATEGORY_MAPPING_XLSX.exists():
        xls = pd.ExcelFile(CATEGORY_MAPPING_XLSX, engine="openpyxl")
        df = pd.read_excel(xls, dtype=str)
        for sheet_name in ["minor_dept_bridge", "minor_dept_map", "minor_dept_lookup"]:
            if sheet_name in xls.sheet_names:
                minor_dept_bridge = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
                break
    elif CATEGORY_MAPPING_PATH.exists():
        df = pd.read_csv(CATEGORY_MAPPING_PATH, dtype=str, encoding="utf-8-sig")
    else:
        return pd.DataFrame()
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {
        "major category": "major_category",
        "major_category": "major_category",
        "category": "major_category",
        "majorcat": "major_category",
        "major cat": "major_category",
        "part_number": "part_number",
        "part number": "part_number",
        "part": "part_number",
        "partnbr": "part_number",
        "part nbr": "part_number",
        "item_id": "item_id",
        "item id": "item_id",
        "item": "item_id",
        "item number": "item_id",
        "itemnbr": "item_id",
        "item nbr": "item_id",
        "minor dept": "minor_dept",
        "minor_dept": "minor_dept",
    }
    df = df.rename(columns=rename_map)

    # Optional bridge if only minor_dept is provided (Path 2)
    if "major_category" not in df.columns and "minor_dept" in df.columns and not minor_dept_bridge.empty:
        bridge = minor_dept_bridge.rename(columns=rename_map)
        if {"minor_dept", "major_category"}.issubset(bridge.columns):
            df = df.merge(bridge[["minor_dept", "major_category"]].drop_duplicates(), on="minor_dept", how="left")

    if "major_category" not in df.columns and "minor_dept" in df.columns:
        df["major_category"] = df["minor_dept"]

    required = {"major_category", "part_number", "item_id"}
    if not required.issubset(df.columns):
        st.error(f"Category mapping missing columns {required - set(df.columns)}; found: {list(df.columns)}")
        return pd.DataFrame()
    df = coerce_str_safe(df, ["major_category", "part_number", "item_id"])
    df = normalize_sku_columns(df, ["part_number", "item_id"])
    df = drop_invalid_products(df)
    return df


@st.cache_data(show_spinner=False)
def load_billback_reason_dim() -> pd.DataFrame:
    if BILLBACK_REASON_XLSX.exists():
        df = pd.read_excel(BILLBACK_REASON_XLSX, dtype=str, engine="openpyxl")
    elif BILLBACK_REASON_PATH.exists():
        df = pd.read_csv(BILLBACK_REASON_PATH, dtype=str, encoding="utf-8-sig")
    else:
        return pd.DataFrame()
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {
        "code_primary": "code_primary",
        "type code": "code_primary",
        "type_code": "code_primary",
        "code": "code_primary",
        "fee code": "code_primary",
        "code_aliases": "code_aliases",
        "aliases": "code_aliases",
        "title": "title",
        "reason": "title",
        "description": "description",
        "direction": "direction",
        "bucket": "bucket",
        "reason bucket": "bucket",
        "category": "bucket",
        "p9": "code_primary",
    }
    df = df.rename(columns=rename_map)
    if "bucket" in df.columns:
        df["bucket"] = df["bucket"]
    else:
        title_series = df["title"] if "title" in df.columns else pd.Series(["Unbucketed"] * len(df), index=df.index)
        df["bucket"] = title_series
    df["bucket"] = df["bucket"].fillna("Unbucketed")
    if "direction" not in df.columns:
        df["direction"] = "unknown"
    df["direction"] = df["direction"].fillna("unknown")
    required = {"code_primary", "title"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Billback reason mapping missing columns {missing}; found: {list(df.columns)}")
        return pd.DataFrame()
    if "code_aliases" not in df.columns:
        df["code_aliases"] = ""
    df["code_aliases"] = df["code_aliases"].fillna("")
    df["all_codes"] = df.apply(
        lambda r: [str(r["code_primary"]).strip().upper()]
        + [c.strip().upper() for c in str(r["code_aliases"]).split("|") if c.strip()],
        axis=1,
    )
    df["code_primary"] = df["code_primary"].str.upper()
    df["bucket"] = df["bucket"].fillna(df["title"]).str.strip()
    df["direction"] = df["direction"].fillna("unknown").str.strip()
    df["code_primary_norm"] = df["code_primary"].apply(normalize_fee_code)
    df["all_codes_norm"] = df["all_codes"].apply(lambda codes: [normalize_fee_code(c) for c in codes])
    return df


# ---------- Derived helpers ----------
def attach_categories(fact_sales: pd.DataFrame, dim_category: pd.DataFrame) -> pd.DataFrame:
    if fact_sales.empty or dim_category.empty:
        fact_sales["major_category"] = "Unmapped"
        return fact_sales
    merged = fact_sales.merge(dim_category, on=["part_number", "item_id"], how="left")
    if merged["major_category"].isna().any():
        # Fallback mapping on part_number only when item_id join fails.
        fallback = dim_category.drop_duplicates(subset=["part_number"])[["part_number", "major_category"]]
        merged = merged.merge(
            fallback.rename(columns={"major_category": "major_category_fallback"}),
            on="part_number",
            how="left",
        )
        merged["major_category"] = merged["major_category"].fillna(merged["major_category_fallback"])
        merged = merged.drop(columns=["major_category_fallback"])
    merged["major_category"] = merged["major_category"].fillna("Unmapped")
    return merged


def filter_valid_sku_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove blank/NA part numbers and subtotal/total rows from SKU-level facts.
    Must be pandas-version-safe (no Series.eq(..., na=...)).
    """
    if df is None or df.empty:
        return df

    cleaned = df.copy()
    na_tokens = {"<NA>", "NA", "N/A", "NONE", "NULL", "", "NAN"}

    # --- part_number cleanup (preserve real NA as NA; do not stringify into '<NA>') ---
    if "part_number" in cleaned.columns:
        pn = column_as_series(cleaned, "part_number").astype("string")
        pn = pn.str.strip()
        pn = pn.replace(na_tokens, pd.NA)
        cleaned["part_number"] = pn
    else:
        cleaned["part_number"] = pd.NA

    mask_na = cleaned["part_number"].isna()

    # pandas-safe blank check: fillna("") then compare
    mask_blank = cleaned["part_number"].fillna("").str.strip().eq("")

    # --- remove totals/subtotals via POV Number ---
    mask_total = pd.Series(False, index=cleaned.index)
    if "pov_number" in cleaned.columns:
        pov = column_as_series(cleaned, "pov_number").astype("string").str.strip()
        pov = pov.replace(na_tokens, pd.NA)

        pov_norm = pov.fillna("").str.strip()
        mask_total = (
            pov_norm.str.fullmatch(r"grand total", case=False)
            | pov_norm.str.contains(r"\btotal\b", case=False, regex=True)
        )

    drop_mask = mask_na | mask_blank | mask_total
    return cleaned.loc[~drop_mask].copy()


def build_product_dim(fact_sales: pd.DataFrame) -> pd.DataFrame:
    if fact_sales.empty:
        return pd.DataFrame(columns=["product_key", "part_number", "item_id", "description"])
    base = fact_sales[["part_number", "item_id", "description"]].copy()
    base = ensure_series_columns(base, ["part_number", "item_id", "description"])
    products = base.drop_duplicates()
    products["product_key"] = column_as_series(products, "part_number").astype(str) + "-" + column_as_series(
        products, "item_id"
    ).astype(str)
    return products


def compute_margin(fact_sales: pd.DataFrame, cost_dim: pd.DataFrame, fiscal_weeks: List[int]) -> pd.DataFrame:
    if fact_sales.empty or cost_dim.empty:
        return pd.DataFrame()
    wk = fact_sales[fact_sales["fiscal_week"].isin(fiscal_weeks)]
    merged = wk.merge(cost_dim, on=["part_number", "item_id"], how="left", suffixes=("", "_cost"))
    merged["cogs"] = merged["units"] * merged["unit_cost"]
    merged["gross_margin"] = merged["revenue"] - merged["cogs"]
    merged["gross_margin_pct"] = merged["gross_margin"] / merged["revenue"].replace({0: pd.NA})
    merged = drop_invalid_products(merged)
    merged = merged[~merged["part_number"].str.contains("TOTAL", case=False, na=False)]
    return merged


def compute_alerts(
    fact_sales: pd.DataFrame,
    cpfr: pd.DataFrame,
    outs: pd.DataFrame,
    forecast: pd.DataFrame,
    orders: pd.DataFrame,
    inventory: pd.DataFrame,
    billbacks: pd.DataFrame,
    thresholds: Dict[str, float],
) -> List[Dict[str, str]]:
    alerts: List[Dict[str, str]] = []
    latest_week = fact_sales["fiscal_week"].max() if not fact_sales.empty else None

    if latest_week and not cpfr.empty:
        latest_cpfr = cpfr[cpfr["fiscal_week"] == latest_week]
        if "fill_rate" in latest_cpfr.columns and latest_cpfr["fill_rate"].notna().any():
            fill_rate = latest_cpfr["fill_rate"].astype(float).mean()
            if fill_rate < thresholds["fill_rate"]:
                alerts.append(
                    {
                        "alert_type": "Fill Rate",
                        "severity": "high",
                        "message": f"Fill rate {fill_rate:.1%} below target {thresholds['fill_rate']:.0%} for FW{latest_week}.",
                        "action": "Work with DC to expedite shipments for top SKUs.",
                    }
                )

    if latest_week and not outs.empty:
        top_outs = outs.nlargest(1, "store_oos_exposure") if "store_oos_exposure" in outs.columns else pd.DataFrame()
        if not top_outs.empty:
            val = float(top_outs["store_oos_exposure"].iloc[0])
            if val > thresholds["outs_exposure"]:
                alerts.append(
                    {
                        "alert_type": "Out-of-stock",
                        "severity": "high",
                        "message": f"Highest store OOS exposure {val:,.0f} exceeds threshold.",
                        "action": "Rebalance inventory or expedite replenishment.",
                    }
                )

    if latest_week and not forecast.empty and not orders.empty:
        fc = forecast[forecast["fiscal_week"] == latest_week]["forecast_units"].sum()
        op = orders[orders["fiscal_week"] == latest_week]["order_units"].sum()
        if fc:
            coverage = op / fc
            if coverage < thresholds["coverage"]:
                alerts.append(
                    {
                        "alert_type": "Forecast Coverage",
                        "severity": "medium",
                        "message": f"Order coverage {coverage:.2f} below 1.0 for FW{latest_week}.",
                        "action": "Increase POs or adjust forecast to align.",
                    }
                )

    if not inventory.empty and "weeks_on_hand" in inventory.columns:
        low_woh = inventory[inventory["weeks_on_hand"] < thresholds["woh_min"]]
        high_woh = inventory[inventory["weeks_on_hand"] > thresholds["woh_max"]]
        if not low_woh.empty:
            alerts.append(
                {
                    "alert_type": "Low WOH",
                    "severity": "medium",
                    "message": f"{len(low_woh)} depts below {thresholds['woh_min']} WOH.",
                    "action": "Prioritize replenishment for low WOH areas.",
                }
            )
        if not high_woh.empty:
            alerts.append(
                {
                    "alert_type": "High WOH",
                    "severity": "low",
                    "message": f"{len(high_woh)} depts above {thresholds['woh_max']} WOH.",
                    "action": "Slow orders or promo to burn down inventory.",
                }
            )

    if latest_week and not billbacks.empty and not fact_sales.empty:
        bb_week = billbacks[billbacks["fiscal_week"] == latest_week]["billback_amount"].sum()
        sales_week = fact_sales[fact_sales["fiscal_week"] == latest_week]["revenue"].sum()
        if sales_week:
            bb_pct = bb_week / sales_week
            if bb_pct > thresholds["billbacks_pct"]:
                alerts.append(
                    {
                        "alert_type": "Billbacks Spike",
                        "severity": "medium",
                        "message": f"Billbacks {bb_pct:.1%} of sales in FW{latest_week}.",
                        "action": "Audit type codes and recover fees where possible.",
                    }
                )

    return alerts


# ---------- UI setup ----------
st.set_page_config(page_title="Innova x AutoZone Decision Dashboard", layout="wide")
st.title("Innova × AutoZone Decision Dashboard")
st.caption("Expanded multi-subtab view across sales, supply, inventory, forecast, and leakage.")

files = get_source_files()
if not files:
    st.error("No files found. Place Innova-AZ FYxxFWxx.xlsx files next to app.py.")
    st.stop()

sig = file_signature(files)
raw_fact_sales = load_sales_actual_tyly(sig)
units_yoy = load_units_yoy(sig)
store_counts = load_store_counts(sig)
forecast_totals, order_totals = load_forecast(sig)
# Backward-compatible aliases for forecast/orders (so existing code using fiscal_week won't crash)
for _df in (forecast_totals, order_totals):
    if _df is not None and not _df.empty:
        if "snapshot_week" in _df.columns and "fiscal_week" not in _df.columns:
            _df["fiscal_week"] = _df["snapshot_week"]
        if "snapshot_year" in _df.columns and "fiscal_year" not in _df.columns:
            _df["fiscal_year"] = _df["snapshot_year"]
cpfr = load_cpfr(sig)
cpfr_detail = load_cpfr_detail(sig)
redflags = load_redflags(sig)
outs = load_outs(sig)
outs_totals = load_outs_totals(sig)
inventory = load_inventory(sig)
returns = load_returns(sig)
cost_dim = load_cost_dim(sig)
billbacks = load_billbacks(sig)
dim_category = load_category_dim()
dim_billback_reason = load_billback_reason_dim()
tyly_grand_total = load_tyly_grand_total(sig)

fact_sales = raw_fact_sales.copy()
if fact_sales.empty:
    st.error("No sales data found. Check TY-LY sheets in the FYxxFWxx files.")
    st.stop()

# Category enrichment
fact_sales = attach_categories(fact_sales, dim_category)
fact_sales = filter_valid_sku_rows(fact_sales)
product_dim = build_product_dim(fact_sales)
dq_removed_rows = len(raw_fact_sales) - len(fact_sales)
pn_series = column_as_series(fact_sales, "part_number").astype("string")
dq_bad_token_mask = pn_series.isin(["<NA>", "N/A", "NA", "NONE", "NULL"])
dq_bad_tokens_remaining = int(dq_bad_token_mask.sum())
exec_sales = fact_sales.copy()
exec_tyly_gt = tyly_grand_total.copy()

# ----- Sidebar filters -----
weeks = sorted(fact_sales["fiscal_week"].unique())
with st.sidebar:
    st.header("Global Filters")
    selected_weeks = st.multiselect("Fiscal Weeks", options=weeks, default=weeks)
    if not selected_weeks:
        selected_weeks = weeks
    category_options = ["All Categories"] + sorted(fact_sales["major_category"].dropna().unique())
    selected_categories = st.multiselect("Major Category", options=category_options, default=["All Categories"])
    part_series = column_as_series(fact_sales, "part_number")
    part_options = ["All Parts"] + sorted(part_series.dropna().unique())
    selected_parts = st.multiselect("Part Number / SKU", options=part_options, default=["All Parts"])
    top_n = st.slider("Top N rows", min_value=5, max_value=50, value=15, step=5)
    show_forecast_overlay = st.checkbox("Overlay Actuals vs Forecast (where available)", value=True)
    st.markdown("---")
    st.subheader("Alert Thresholds")
    fill_rate_threshold = st.slider("Fill rate minimum", 0.5, 1.0, 0.9, 0.05)
    outs_threshold = st.number_input("OOS exposure threshold", min_value=0.0, value=200.0, step=50.0)
    coverage_threshold = st.slider("Forecast coverage min", 0.2, 1.5, 1.0, 0.05)
    woh_min = st.slider("WOH min", 0.0, 8.0, 2.0, 0.5)
    woh_max = st.slider("WOH max", 4.0, 30.0, 12.0, 0.5)
    billbacks_pct = st.slider("Billbacks % of sales alert", 0.0, 0.25, 0.08, 0.01)
    st.markdown("---")
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


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    filtered = df.copy()
    if "fiscal_week" in filtered.columns:
        filtered = filtered[filtered["fiscal_week"].isin(selected_weeks)]
    if "major_category" in filtered.columns and selected_categories and "All Categories" not in selected_categories:
        filtered = filtered[filtered["major_category"].isin(selected_categories)]
    if "part_number" in filtered.columns and selected_parts and "All Parts" not in selected_parts:
        part_series = column_as_series(filtered, "part_number")
        filtered = filtered[part_series.isin(selected_parts)]
    return filtered


def apply_week_filter_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter ONLY by weeks (snapshot or fiscal); do NOT filter by category/part."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "snapshot_week" in out.columns:
        out = out[out["snapshot_week"].isin(selected_weeks)]
    elif "fiscal_week" in out.columns:
        out = out[out["fiscal_week"].isin(selected_weeks)]
    return out


filtered_sales = apply_filters(fact_sales)
filtered_units_yoy = apply_filters(units_yoy)
filtered_store_counts = apply_filters(store_counts)
# Forecast/Orders use week-only filter to remain "full-book" totals (not affected by category/part filters)
filtered_forecast = apply_week_filter_only(forecast_totals)
filtered_orders = apply_week_filter_only(order_totals)
# For Forecast tab, actuals should also be week-only so the three series are comparable
forecast_actual_src = apply_week_filter_only(exec_tyly_gt)
filtered_cpfr = apply_filters(cpfr)
cpfr_detail_filtered = apply_week_filter_only(cpfr_detail)
filtered_redflags = apply_filters(redflags)
filtered_outs = apply_filters(outs)
filtered_outs_totals = apply_week_filter_only(outs_totals)
filtered_inventory = apply_filters(inventory)
filtered_returns = apply_filters(returns)
# Billbacks use invoice_date (calendar-based). Do NOT filter by FY fiscal week selector.
filtered_billbacks = billbacks.copy()
filtered_cost = cost_dim  # cost not week-based

# Executive Summary data constrained only by selected weeks (ignores category/part filters)
exec_sales_filtered = exec_sales[exec_sales["fiscal_week"].isin(selected_weeks)] if selected_weeks else exec_sales
exec_tyly_gt_filtered = exec_tyly_gt[exec_tyly_gt["fiscal_week"].isin(selected_weeks)] if selected_weeks else exec_tyly_gt

alerts = compute_alerts(
    filtered_sales,
    filtered_cpfr,
    filtered_outs,
    filtered_forecast,
    filtered_orders,
    filtered_inventory,
    filtered_billbacks,
    thresholds,
)


def get_snapshot_context(gt_df: pd.DataFrame) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if gt_df.empty:
        return None, None, None
    latest_year = gt_df["fiscal_year"].max()
    latest_year_df = gt_df[gt_df["fiscal_year"] == latest_year]
    snapshot_week = latest_year_df["fiscal_week"].max()
    prev_week = snapshot_week - 1 if (snapshot_week - 1) in latest_year_df["fiscal_week"].values else None
    return latest_year, snapshot_week, prev_week


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
    st.subheader("Alerts")
    if not alerts_list:
        st.success("No alerts triggered for the selected filters.")
        return
    for alert in alerts_list:
        st.warning(f"{alert['alert_type']}: {alert['message']}  \nAction: {alert['action']}")


def compute_comparable_yoy(
    df: pd.DataFrame, week: int, dimension: str, min_ly_rev_floor: float
) -> pd.DataFrame:
    wk = df[df["fiscal_week"] == week].copy()
    wk = wk.dropna(subset=[dimension])
    wk = wk[wk[dimension].astype(str).str.strip().ne("")]
    grouped = (
        wk.groupby([dimension])
        .agg(ty_rev=("revenue", "sum"), ly_rev=("ly_revenue", "sum"), ty_units=("units", "sum"), ly_units=("ly_units", "sum"))
        .reset_index()
    )
    grouped["yoy_rev_pct_comp"] = (grouped["ty_rev"] - grouped["ly_rev"]) / grouped["ly_rev"]
    bad_mask = (grouped["ly_rev"] <= 0) & (grouped["ty_rev"] > 0)
    zero_mask = grouped["ly_rev"] == 0
    small_mask = grouped["ly_rev"].abs() < min_ly_rev_floor
    grouped.loc[bad_mask | zero_mask | small_mask, "yoy_rev_pct_comp"] = pd.NA
    return grouped


def compute_wow_pct(df: pd.DataFrame, snap_week: int, prev_week: Optional[int], dimension: str) -> pd.DataFrame:
    if prev_week is None:
        return pd.DataFrame()
    snap = df[df["fiscal_week"] == snap_week].copy()
    prev = df[df["fiscal_week"] == prev_week].copy()
    snap = snap.dropna(subset=[dimension])
    prev = prev.dropna(subset=[dimension])
    snap[dimension] = snap[dimension].astype(str).str.strip()
    prev[dimension] = prev[dimension].astype(str).str.strip()
    snap = snap[snap[dimension].ne("")]
    prev = prev[prev[dimension].ne("")]
    snap_group = snap.groupby([dimension]).agg(rev_snap=("revenue", "sum")).reset_index()
    prev_group = prev.groupby([dimension]).agg(rev_prev=("revenue", "sum")).reset_index()
    merged = snap_group.merge(prev_group, on=dimension, how="outer").fillna(0)
    merged = merged[merged["rev_prev"] != 0]
    merged["wow_rev_pct"] = (merged["rev_snap"] - merged["rev_prev"]) / merged["rev_prev"]
    return merged


def compute_ytd_yoy(df: pd.DataFrame, week: int, dimension: str, min_ly_rev_floor: float) -> pd.DataFrame:
    wk = df[df["fiscal_week"] == week].copy()
    wk = wk.dropna(subset=[dimension])
    wk = wk[wk[dimension].astype(str).str.strip().ne("")]
    grouped = (
        wk.groupby([dimension])
        .agg(fytd_rev=("fytd_revenue", "sum"), fytd_yoy_pct=("fytd_revenue_diff_pct", "max"))
        .reset_index()
    )
    # Estimate LY from TY and pct to gate comparability
    grouped["ly_est"] = grouped.apply(
        lambda r: r["fytd_rev"] / (1 + r["fytd_yoy_pct"]) if pd.notna(r["fytd_yoy_pct"]) and (1 + r["fytd_yoy_pct"]) != 0 else pd.NA,
        axis=1,
    )
    bad_mask = grouped["ly_est"].le(0) | grouped["ly_est"].abs().lt(min_ly_rev_floor)
    grouped.loc[bad_mask, "fytd_yoy_pct"] = pd.NA
    return grouped


def compute_ytd_wow(df: pd.DataFrame, snap_week: int, prev_week: Optional[int], dimension: str) -> pd.DataFrame:
    if prev_week is None:
        return pd.DataFrame()
    snap = df[df["fiscal_week"] == snap_week].copy()
    prev = df[df["fiscal_week"] == prev_week].copy()
    snap = snap.dropna(subset=[dimension])
    prev = prev.dropna(subset=[dimension])
    snap[dimension] = snap[dimension].astype(str).str.strip()
    prev[dimension] = prev[dimension].astype(str).str.strip()
    snap = snap[snap[dimension].ne("")]
    prev = prev[prev[dimension].ne("")]
    snap_group = snap.groupby([dimension]).agg(fytd_rev_snap=("fytd_revenue", "sum")).reset_index()
    prev_group = prev.groupby([dimension]).agg(fytd_rev_prev=("fytd_revenue", "sum")).reset_index()
    merged = snap_group.merge(prev_group, on=dimension, how="outer").fillna(0)
    merged = merged[merged["fytd_rev_prev"] != 0]
    merged["fytd_wow_pct"] = (merged["fytd_rev_snap"] - merged["fytd_rev_prev"]) / merged["fytd_rev_prev"]
    return merged


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

tab_exec, tab_perf, tab_forecast, tab_supply, tab_inventory, tab_returns_tab, tab_finance = st.tabs(
    [
        "Executive Summary",
        "Category & Product Performance",
        "Forecast & Orders",
        "Supply Health",
        "Inventory & Outs",
        "Returns Analysis",
        "Profitability & Billbacks",
    ]
)


# ----- Executive Summary -----
with tab_exec:
    st.subheader("KPI Tiles")
    render_kpi_tiles_es(exec_tyly_gt_filtered)
    st.subheader("Hero / Rising / Declining (Current Week)")
    snapshot_year, snapshot_week, prev_week = get_snapshot_context(exec_tyly_gt_filtered)
    if snapshot_week is None:
        st.info("No snapshot week found in Grand Total rows.")
    else:
        render_hero_tiles_es(exec_sales_filtered, dimension="part_number", label="Product", snapshot_week=snapshot_week, prev_week=prev_week)
        render_hero_tiles_es(exec_sales_filtered, dimension="major_category", label="Category", snapshot_week=snapshot_week, prev_week=prev_week)

    st.subheader("Hero / Rising / Declining — Fiscal YTD")
    if snapshot_week is None:
        st.info("No snapshot week found for FYTD view.")
    elif "fytd_revenue" not in exec_sales_filtered.columns:
        st.info("FYTD columns not available in TY-LY data.")
    else:
        render_hero_tiles_es_ytd(exec_sales_filtered, dimension="part_number", label="Product", snapshot_week=snapshot_week, prev_week=prev_week)
        render_hero_tiles_es_ytd(exec_sales_filtered, dimension="major_category", label="Category", snapshot_week=snapshot_week, prev_week=prev_week)

    st.subheader("Revenue & Units Trend")
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
        st.altair_chart(line_units, use_container_width=True)
        st.altair_chart(line_rev, use_container_width=True)

    st.subheader("Top Movers (WoW)")
    if snapshot_week is None or prev_week is None:
        st.info("Need snapshot and previous week to compute movers.")
    else:
        latest = exec_sales_filtered[exec_sales_filtered["fiscal_week"] == snapshot_week]
        prev = exec_sales_filtered[exec_sales_filtered["fiscal_week"] == prev_week]
        latest = latest.dropna(subset=["part_number"]).copy()
        prev = prev.dropna(subset=["part_number"]).copy()
        latest = latest[~latest["part_number"].astype(str).str.contains("TOTAL", case=False, na=False)]
        prev = prev[~prev["part_number"].astype(str).str.contains("TOTAL", case=False, na=False)]
        latest_units = latest.groupby("part_number")[["units", "revenue"]].sum().reset_index()
        prev_units = prev.groupby("part_number")[["units", "revenue"]].sum().reset_index()
        merged = latest_units.merge(prev_units, on="part_number", how="outer", suffixes=("", "_prev"))
        latest_parts = set(latest_units["part_number"].astype(str))
        prev_parts = set(prev_units["part_number"].astype(str))
        merged["part_number"] = merged["part_number"].astype(str)
        merged["snapshot_present"] = merged["part_number"].isin(latest_parts)
        merged["prev_present"] = merged["part_number"].isin(prev_parts)
        for col in ["units", "revenue", "units_prev", "revenue_prev"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        merged["units_delta"] = merged["units"] - merged["units_prev"]
        merged["revenue_delta"] = merged["revenue"] - merged["revenue_prev"]
        cols_keep_units = ["part_number", "units_delta", "snapshot_present", "prev_present"]
        cols_keep_rev = ["part_number", "revenue_delta", "snapshot_present", "prev_present"]
        top_units_up = merged.sort_values("units_delta", ascending=False).head(5)[cols_keep_units]
        top_units_down = merged.sort_values("units_delta", ascending=True).head(5)[cols_keep_units]
        top_rev_up = merged.sort_values("revenue_delta", ascending=False).head(5)[cols_keep_rev]
        top_rev_down = merged.sort_values("revenue_delta", ascending=True).head(5)[cols_keep_rev]
        for df_rev in [top_rev_up, top_rev_down]:
            df_rev["revenue_delta"] = df_rev["revenue_delta"].apply(lambda v: format_currency_0(v))
        cols = st.columns(2)
        cols[0].markdown("**Top + Movers by Units Δ**")
        cols[0].dataframe(
            top_units_up.rename(
                columns={"units_delta": "Units Δ", "snapshot_present": "FW Snapshot", "prev_present": "FW Prev"}
            ).drop(columns=["FW Snapshot", "FW Prev"], errors="ignore"),
            hide_index=True,
            use_container_width=True,
        )
        cols[1].markdown("**Top - Movers by Units Δ**")
        cols[1].dataframe(
            top_units_down.rename(
                columns={"units_delta": "Units Δ", "snapshot_present": "FW Snapshot", "prev_present": "FW Prev"}
            ).drop(columns=["FW Snapshot", "FW Prev"], errors="ignore"),
            hide_index=True,
            use_container_width=True,
        )
        cols2 = st.columns(2)
        cols2[0].markdown("**Top + Movers by Revenue Δ**")
        cols2[0].dataframe(
            top_rev_up.rename(
                columns={"revenue_delta": "Revenue Δ", "snapshot_present": "FW Snapshot", "prev_present": "FW Prev"}
            ).drop(columns=["FW Snapshot", "FW Prev"], errors="ignore"),
            hide_index=True,
            use_container_width=True,
        )
        cols2[1].markdown("**Top - Movers by Revenue Δ**")
        cols2[1].dataframe(
            top_rev_down.rename(
                columns={"revenue_delta": "Revenue Δ", "snapshot_present": "FW Snapshot", "prev_present": "FW Prev"}
            ).drop(columns=["FW Snapshot", "FW Prev"], errors="ignore"),
            hide_index=True,
            use_container_width=True,
        )

    render_alerts(alerts)


# ----- Category & Product Performance -----
with tab_perf:
    st.subheader("Product Performance")
    prod_df = filtered_sales.copy()
    prod_df = prod_df.dropna(subset=["part_number"])
    prod_df = prod_df[~prod_df["part_number"].astype(str).str.contains("TOTAL", case=False, na=False)]
    prod_group = (
        prod_df.groupby(["part_number", "description"])
        .agg(revenue=("revenue", "sum"), units=("units", "sum"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    prod_group.insert(0, "rank", range(1, len(prod_group) + 1))
    prod_group = prod_group.head(top_n)
    st.dataframe(
        format_currency_columns(prod_group, ["revenue"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Units Trend by Product (TY-LY)")
    prod_trend = (
        filtered_sales.dropna(subset=["part_number"])
        .groupby(["part_number", "fiscal_week"])["units"]
        .sum()
        .reset_index()
        .sort_values(["part_number", "fiscal_week"])
    )
    if not prod_trend.empty:
        prod_trend = prod_trend[prod_trend["part_number"].notna() & (prod_trend["part_number"].astype(str) != "<NA>")]
        chart_prod = (
            alt.Chart(prod_trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("fiscal_week:O", title="Fiscal Week", axis=alt.Axis(tickMinStep=1, format="d")),
                y="units:Q",
                color="part_number:N",
            )
        )
        st.altair_chart(chart_prod, use_container_width=True)
    st.subheader("Revenue Trend by Product (TY-LY)")
    rev_prod = (
        filtered_sales.dropna(subset=["part_number"])
        .groupby(["part_number", "fiscal_week"])["revenue"]
        .sum()
        .reset_index()
        .sort_values(["part_number", "fiscal_week"])
    )
    if not rev_prod.empty:
        rev_prod = rev_prod[rev_prod["part_number"].notna() & (rev_prod["part_number"].astype(str) != "<NA>")]
        chart_rev_prod = (
            alt.Chart(rev_prod)
            .mark_line(point=True)
            .encode(
                x=alt.X("fiscal_week:O", title="Fiscal Week", axis=alt.Axis(tickMinStep=1, format="d")),
                y=alt.Y("revenue:Q", axis=alt.Axis(format="$,.0f")),
                color="part_number:N",
            )
        )
        st.altair_chart(chart_rev_prod, use_container_width=True)

    st.subheader("Category Ranking")
    cat_df = filtered_sales.copy()
    cat_df = cat_df.dropna(subset=["major_category"])
    cat_group = (
        cat_df.groupby("major_category")
        .agg(revenue=("revenue", "sum"), units=("units", "sum"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    cat_group.insert(0, "rank", range(1, len(cat_group) + 1))
    cat_group = cat_group.head(top_n)
    st.dataframe(
        format_currency_columns(cat_group, ["revenue"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Units Trend by Category (TY-LY)")
    cat_units_trend = (
        filtered_sales.dropna(subset=["major_category"])
        .groupby(["major_category", "fiscal_week"])["units"]
        .sum()
        .reset_index()
        .sort_values(["major_category", "fiscal_week"])
    )
    if not cat_units_trend.empty:
        chart_cat_units = (
            alt.Chart(cat_units_trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("fiscal_week:O", title="Fiscal Week", axis=alt.Axis(tickMinStep=1, format="d")),
                y="units:Q",
                color="major_category:N",
            )
        )
        st.altair_chart(chart_cat_units, use_container_width=True)

    st.subheader("Revenue Trend by Category (TY-LY)")
    cat_rev_trend = (
        filtered_sales.dropna(subset=["major_category"])
        .groupby(["major_category", "fiscal_week"])["revenue"]
        .sum()
        .reset_index()
        .sort_values(["major_category", "fiscal_week"])
    )
    if not cat_rev_trend.empty:
        chart_cat_rev = (
            alt.Chart(cat_rev_trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("fiscal_week:O", title="Fiscal Week", axis=alt.Axis(tickMinStep=1, format="d")),
                y=alt.Y("revenue:Q", axis=alt.Axis(format="$,.0f")),
                color="major_category:N",
            )
        )
        st.altair_chart(chart_cat_rev, use_container_width=True)


# ----- Forecast & Orders -----
with tab_forecast:
    st.subheader("Forecast vs Order Coverage")
    # Use week-only filtered data for all three series to ensure comparability
    # (forecast/orders/actuals all use "full-book" totals, not affected by category/part filters)
    if filtered_forecast.empty and filtered_orders.empty and forecast_actual_src.empty:
        st.info("No forecast/order data found.")
    else:
        fc = filtered_forecast.copy()
        op = filtered_orders.copy()
        actual = exec_tyly_gt_filtered.rename(
            columns={"fw_units_total": "actual_units_gt", "fiscal_year": "target_year", "fiscal_week": "target_week"}
        )

        # Snapshot-week series: use rows where target matches snapshot (target_week == snapshot_week, target_year == snapshot_year)
        fc_snap_series = fc[(fc["target_year"] == fc["snapshot_year"]) & (fc["target_week"] == fc["snapshot_week"])]
        op_snap_series = op[(op["target_year"] == op["snapshot_year"]) & (op["target_week"] == op["snapshot_week"])]

        snap_merge = (
            fc_snap_series.rename(columns={"forecast_units": "forecast_units_gt"})
            .merge(
                op_snap_series.rename(columns={"order_units": "order_units_gt"}),
                on=["snapshot_year", "snapshot_week", "target_year", "target_week"],
                how="outer",
            )
            .merge(
                actual.rename(columns={"target_week": "snapshot_week", "target_year": "snapshot_year"}),
                on=["snapshot_year", "snapshot_week"],
                how="left",
            )
        )
        snap_merge["has_forecast"] = snap_merge["forecast_units_gt"].notna()
        snap_merge["order_missing"] = snap_merge["order_units_gt"].isna()
        snap_merge["coverage"] = snap_merge.apply(
            lambda r: r["order_units_gt"] / r["forecast_units_gt"] if r["has_forecast"] and r["forecast_units_gt"] != 0 else pd.NA,
            axis=1,
        )
        snap_merge["forecast_error_pct"] = snap_merge.apply(
            lambda r: (r["actual_units_gt"] - r["forecast_units_gt"]) / r["forecast_units_gt"]
            if r["has_forecast"] and r["forecast_units_gt"] != 0 and pd.notna(r["actual_units_gt"])
            else pd.NA,
            axis=1,
        )
        snap_merge["gap_units"] = snap_merge.apply(
            lambda r: r["forecast_units_gt"] - r["order_units_gt"] if r["has_forecast"] else pd.NA, axis=1
        )

        # Snapshot KPIs: latest snapshot_week among available snapshots with forecast
        snap_available = snap_merge[snap_merge["has_forecast"]]
        if not snap_available.empty:
            snap_year = snap_available["snapshot_year"].max()
            snap_week = snap_available[snap_available["snapshot_year"] == snap_year]["snapshot_week"].max()
            snap_row = snap_available[
                (snap_available["snapshot_year"] == snap_year) & (snap_available["snapshot_week"] == snap_week)
            ].iloc[0]
            cov_snap = snap_row["coverage"]
            fe_snap = snap_row["forecast_error_pct"]
            gap_snap = snap_row["gap_units"]
        else:
            snap_week = None
            cov_snap = None
            fe_snap = None
            gap_snap = None

        cols_kpi = st.columns(3)
        week_label = f"FW{int(snap_week)}" if snap_week is not None else "N/A"
        cols_kpi[0].metric(f"Coverage ({week_label})", f"{cov_snap:.2%}" if cov_snap is not None else "N/A")
        cols_kpi[1].metric(f"Forecast Error% ({week_label})", f"{fe_snap:.2%}" if fe_snap is not None else "N/A")
        cols_kpi[2].metric(f"Gap Units ({week_label})", f"{int(round_half_up(gap_snap)):,}" if gap_snap is not None else "N/A")
        if snap_available.empty or snap_row.get("order_missing", False):
            st.warning(
                f"Data Quality Warning: Order Projection (Grand Total) is missing for {week_label}. "
                "Coverage cannot be computed and is shown as N/A. Please verify the forecast sheet Order Projection block."
            )

        # Snapshot-week time series (forecast/order/actual and coverage/error)
        snapshot_series = snap_merge.sort_values(["snapshot_year", "snapshot_week"])
        cov_chart_data = snapshot_series[snapshot_series["has_forecast"]]
        if not cov_chart_data.empty:
            cov_chart = (
                alt.Chart(cov_chart_data)
                .mark_bar()
                .encode(
                    x=alt.X("snapshot_week:O", title="Snapshot Week"),
                    y=alt.Y("coverage:Q", axis=alt.Axis(format=".0%")),
                    color=alt.condition("datum.coverage < 1", alt.value("#ef4444"), alt.value("#10b981")),
                    tooltip=[
                        alt.Tooltip("snapshot_year:O", title="Snapshot FY"),
                        alt.Tooltip("snapshot_week:O", title="Snapshot Week"),
                        alt.Tooltip("coverage:Q", format=".2%", title="Coverage"),
                        alt.Tooltip("forecast_units_gt:Q", format=",", title="Forecast Units"),
                        alt.Tooltip("order_units_gt:Q", format=",", title="Order Units"),
                        alt.Tooltip("actual_units_gt:Q", format=",", title="Actual Units"),
                    ],
                )
            )
            cov_ref = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(color="#9ca3af", strokeDash=[4, 4]).encode(y="y:Q")
            err_chart = (
                alt.Chart(cov_chart_data)
                .mark_line(point=True, color="#2563eb")
                .encode(
                    x=alt.X("snapshot_week:O", title="Snapshot Week"),
                    y=alt.Y("forecast_error_pct:Q", axis=alt.Axis(format=".0%"), title="Forecast Error%"),
                    tooltip=[
                        alt.Tooltip("snapshot_year:O", title="Snapshot FY"),
                        alt.Tooltip("snapshot_week:O", title="Snapshot Week"),
                        alt.Tooltip("forecast_error_pct:Q", format=".2%", title="Forecast Error%"),
                    ],
                )
            )
            st.altair_chart(cov_chart + cov_ref, use_container_width=True)
            st.altair_chart(err_chart, use_container_width=True)
        else:
            st.info("No snapshot-week forecast data available for coverage/error charts.")

        # Snapshot-week level units line chart
        units_combo_frames = []
        if not snapshot_series.empty:
            tmp = snapshot_series.copy()
            units_combo_frames.append(
                tmp[["snapshot_week", "forecast_units_gt"]].rename(columns={"forecast_units_gt": "value"}).assign(series="Forecast GT")
            )
            units_combo_frames.append(
                tmp[["snapshot_week", "order_units_gt"]].rename(columns={"order_units_gt": "value"}).assign(series="Order Projection GT")
            )
            units_combo_frames.append(
                tmp[["snapshot_week", "actual_units_gt"]].rename(columns={"actual_units_gt": "value"}).assign(series="Actual Units GT")
            )
        if units_combo_frames:
            units_combo = pd.concat(units_combo_frames, ignore_index=True).dropna(subset=["value"])
            units_chart = (
                alt.Chart(units_combo)
                .mark_line(point=True)
                .encode(
                    x=alt.X("snapshot_week:Q", title="Snapshot Week"),
                    y=alt.Y("value:Q"),
                    color="series:N",
                    tooltip=["snapshot_week", "series", alt.Tooltip("value:Q", format=",")],
                )
                .properties(height=260)
            )
            st.altair_chart(units_chart, use_container_width=True)

        # Plan history for a chosen target week
        st.subheader("Plan History for Target Week (Grand Total)")
        available_targets = sorted(fc["target_week"].dropna().unique()) if not fc.empty else []
        if not available_targets:
            st.info("No forecast data available for plan history.")
        else:
            selected_target_week = st.selectbox("Target Fiscal Week", options=available_targets, index=len(available_targets) - 1)
            fc_hist = fc[fc["target_week"] == selected_target_week]
            op_hist = op[op["target_week"] == selected_target_week]
            hist = fc_hist.rename(columns={"forecast_units": "forecast_units_gt"}).merge(
                op_hist.rename(columns={"order_units": "order_units_gt"}),
                on=["snapshot_year", "snapshot_week", "target_year", "target_week"],
                how="outer",
            )
            if hist.empty:
                st.info("No data for selected target week.")
            else:
                hist = hist.sort_values(["snapshot_year", "snapshot_week"])
                hist_long = pd.concat(
                    [
                        hist[["snapshot_week", "forecast_units_gt"]].assign(series="Forecast GT").rename(
                            columns={"forecast_units_gt": "value"}
                        ),
                        hist[["snapshot_week", "order_units_gt"]].assign(series="Order Projection GT").rename(
                            columns={"order_units_gt": "value"}
                        ),
                    ],
                    ignore_index=True,
                ).dropna(subset=["value"])
                hist_chart = (
                    alt.Chart(hist_long)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("snapshot_week:Q", title="Snapshot Week"),
                        y=alt.Y("value:Q"),
                        color="series:N",
                        tooltip=["snapshot_week", "series", alt.Tooltip("value:Q", format=",")],
                    )
                    .properties(height=260)
                )
                st.altair_chart(hist_chart, use_container_width=True)

        st.caption(
            "Coverage = Orders ÷ Forecast (supply vs demand). Forecast Error% = (Actual − Forecast) ÷ Forecast. "
            "Gap Units = Forecast − Orders. All values use Forecast sheet Grand Total cells (no SKU summation)."
        )


# ----- Supply Health -----
with tab_supply:
    st.subheader("Service KPIs")
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

        years = [current_fy, current_fy - 1, current_fy - 2]
        trend = cpfr_base[cpfr_base["fiscal_year"].isin(years)].sort_values(["fiscal_year", "fiscal_week"])

        st.subheader(f"Shipped Units Trend (FY{years[0]}, FY{years[1]}, FY{years[2]})")
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

        st.subheader(f"Fill Rate Trend (FY{years[0]}, FY{years[1]}, FY{years[2]})")
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

        # Fill Rate Red Flags from Fill Rate Red Flags sheet (latest snapshot only)
        st.subheader(
            "Fill Rate Red Flags",
            help="LFW = last fiscal week; L4FW = last 4 fiscal weeks (last month); L52FW = last 52 fiscal weeks (past year). Values are Not Shipped units.",
        )
        if redflags.empty:
            st.info("No Fill Rate Red Flags data available.")
        else:
            rf_base = redflags.copy()
            rf_base = rf_base.dropna(subset=["part_number"])
            if rf_base.empty:
                st.info("Fill Rate Red Flags sheet is present but contains no valid product rows.")
            else:
                cur_year = (
                    int(rf_base["snapshot_year"].max()) if "snapshot_year" in rf_base.columns and rf_base["snapshot_year"].notna().any() else None
                )
                if cur_year is None:
                    cur_week = None
                    rf_cur = rf_base
                else:
                    cur_week = int(rf_base[rf_base["snapshot_year"] == cur_year]["snapshot_week"].max())
                    rf_cur = rf_base[(rf_base["snapshot_year"] == cur_year) & (rf_base["snapshot_week"] == cur_week)]

                def top_card(metric: str, title: str, col):
                    sub = rf_cur.dropna(subset=[metric])
                    if sub.empty:
                        col.metric(title, "N/A", help="No data for this window.")
                        return
                    row = sub.loc[sub[metric].idxmax()]
                    val = row[metric]
                    prod = row["part_number"]
                    col.metric(title, prod, delta=f"{int(val):,} Not Shipped")

                cols_rf = st.columns(3)
                top_card("not_shipped_lfw", "Top Red Flag — LFW", cols_rf[0])
                top_card("not_shipped_l4w", "Top Red Flag — L4FW", cols_rf[1])
                top_card("not_shipped_l52w", "Top Red Flag — L52FW", cols_rf[2])


# ----- Inventory & Outs -----
with tab_inventory:
    st.subheader("Out-of-Stock Exposure")
    if filtered_outs.empty and (filtered_outs_totals is None or filtered_outs_totals.empty):
        st.info("No Outs data available.")
    else:
        total_exposure = filtered_outs_totals["store_oos_total"].sum() if not filtered_outs_totals.empty else 0
        if filtered_outs.empty:
            st.info("No Outs detail available for SKU ranking.")
            top_sku = pd.DataFrame()
        else:
            agg_parts = (
                filtered_outs.groupby("part_number")["store_oos_exposure"]
                .sum()
                .reset_index()
                .sort_values("store_oos_exposure", ascending=False)
            )
            top_sku = agg_parts.head(1)
        cols = st.columns(3)
        cols[0].metric(
            "Total OOS Exposure (stores)",
            f"{total_exposure:,.0f}",
            help="Sum of Stores Out of Stock across selected weeks.",
        )
        if not top_sku.empty:
            cols[1].metric(
                "Highest OOS SKU",
                str(top_sku.iloc[0]["part_number"]),
                delta=f"{int(top_sku.iloc[0]['store_oos_exposure'])} stores",
                help="SKU with highest store OOS exposure (sum across selected weeks).",
            )
        outs_top = filtered_outs.sort_values("store_oos_exposure", ascending=False).head(top_n)
        st.dataframe(outs_top, use_container_width=True, hide_index=True)
        bar = (
            alt.Chart(outs_top)
            .mark_bar()
            .encode(
                x=alt.X("part_number:N", title="Part"),
                y=alt.Y("store_oos_exposure:Q", title="Store OOS Exposure"),
                color="part_number:N",
                tooltip=["part_number", "item_id", "store_oos_exposure"],
            )
        )
        st.altair_chart(bar, use_container_width=True)

    st.subheader("Weeks on Hand Distribution")
    if filtered_inventory.empty:
        st.info("No inventory data available.")
    else:
        inv_rows = filtered_inventory.copy()
        if "is_grand_total" in inv_rows.columns:
            inv_rows_data = inv_rows[~inv_rows["is_grand_total"]]
        else:
            inv_rows_data = inv_rows
        hist = (
            alt.Chart(inv_rows_data)
            .mark_bar()
            .encode(x=alt.X("weeks_on_hand:Q", bin=True), y="count()")
        )
        st.altair_chart(hist, use_container_width=True)
        low = inv_rows_data[inv_rows_data["weeks_on_hand"] < thresholds["woh_min"]]
        high = inv_rows_data[inv_rows_data["weeks_on_hand"] > thresholds["woh_max"]]
        cols = st.columns(3)
        if "is_grand_total" in filtered_inventory.columns:
            gt_row = filtered_inventory[filtered_inventory["is_grand_total"]]
            avg_woh_val = gt_row["weeks_on_hand"].iloc[0] if not gt_row.empty else inv_rows_data["weeks_on_hand"].mean()
        else:
            avg_woh_val = inv_rows_data["weeks_on_hand"].mean()
        cols[0].metric("Avg WOH", f"{avg_woh_val:.2f}" if pd.notna(avg_woh_val) else "N/A", help="Average weeks on hand from Inventory Grand Total (if available).")
        cols[1].metric("Below Min WOH", f"{len(low)}", help="Count of rows with WOH below the min threshold.")
        cols[2].metric("Above Max WOH", f"{len(high)}", help="Count of rows with WOH above the max threshold.")
        st.dataframe(inv_rows_data.sort_values("weeks_on_hand"), use_container_width=True, hide_index=True)

        st.subheader("WOH Threshold Exceptions")
        min_woh_thr = st.number_input("Min WOH threshold", value=2.0, step=0.5)
        max_woh_thr = st.number_input("Max WOH threshold", value=10.0, step=0.5)

        low_exc = inv_rows_data[inv_rows_data["weeks_on_hand"] < min_woh_thr]
        high_exc = inv_rows_data[inv_rows_data["weeks_on_hand"] > max_woh_thr]
        total_rows = len(inv_rows_data)
        low_pct = (len(low_exc) / total_rows * 100) if total_rows else 0
        high_pct = (len(high_exc) / total_rows * 100) if total_rows else 0

        exc_cols = st.columns(2)
        exc_cols[0].metric("Low WOH Minor Depts", f"{len(low_exc)} ({low_pct:.1f}%)", help="Count and % of minor depts below min threshold.")
        exc_cols[1].metric("High WOH Minor Depts", f"{len(high_exc)} ({high_pct:.1f}%)", help="Count and % of minor depts above max threshold.")

        st.markdown("**Top Low WOH Minor Depts**")
        low_top = low_exc.sort_values("weeks_on_hand").head(15)[["minor_dept", "weeks_on_hand"]]
        st.dataframe(low_top, use_container_width=True, hide_index=True)
        if len(low_exc) > 15:
            with st.expander("Show all low WOH SKUs"):
                st.dataframe(low_exc.sort_values("weeks_on_hand")[["minor_dept", "weeks_on_hand"]], use_container_width=True, hide_index=True)

        st.markdown("**Top High WOH Minor Depts**")
        high_top = high_exc.sort_values("weeks_on_hand", ascending=False).head(15)[["minor_dept", "weeks_on_hand"]]
        st.dataframe(high_top, use_container_width=True, hide_index=True)
        if len(high_exc) > 15:
            with st.expander("Show all high WOH SKUs"):
                st.dataframe(high_exc.sort_values("weeks_on_hand", ascending=False)[["minor_dept", "weeks_on_hand"]], use_container_width=True, hide_index=True)

    if not filtered_returns.empty:
        st.info("Returns detail moved to 'Returns Analysis' tab.")


# ----- Returns Analysis -----
with tab_returns_tab:
    st.subheader("Returns KPIs")
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
        # --- Top Risk SKU must be MAX %DAMAGED (damaged_rate), volume-gated, valid SKU only ---
        latest = latest.copy()

        # Clean valid part_number rows only (no totals, no blanks)
        latest = latest.dropna(subset=["part_number"])
        latest = latest[latest["part_number"].astype(str).str.strip().ne("")]
        latest = latest[~latest["part_number"].astype(str).str.contains("TOTAL", case=False, na=False)]

        # Ensure damaged_units exists; if missing, estimate = damaged_rate * gross_units
        if "damaged_units" not in latest.columns or latest["damaged_units"].isna().all():
            latest["damaged_units_calc"] = latest["damaged_rate"].fillna(0) * latest["gross_units"].fillna(0)
            dmg_units_col = "damaged_units_calc"
        else:
            latest["damaged_units_calc"] = latest["damaged_units"].fillna(0)
            dmg_units_col = "damaged_units_calc"

        # Aggregate to SKU-level (in case duplicates)
        sku_latest = (
            latest.groupby("part_number", dropna=False)
            .agg(
                damaged_rate_avg=("damaged_rate", "mean"),        # ranking metric
                damaged_units_sum=(dmg_units_col, "sum"),
                gross_units_sum=("gross_units", "sum"),
            )
            .reset_index()
        )

        # Top Risk SKU = max %damaged (tie-breaker: damaged_units_sum, then gross_units_sum)
        top_risk = sku_latest.sort_values(
            ["damaged_rate_avg", "damaged_units_sum", "gross_units_sum"],
            ascending=[False, False, False],
        ).head(1)

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

        if not top_risk.empty:
            part = top_risk["part_number"].iloc[0]
            curr_dmg_units = float(top_risk["damaged_units_sum"].iloc[0]) if pd.notna(top_risk["damaged_units_sum"].iloc[0]) else None

            weeks_in_year = (
                filtered_returns.loc[filtered_returns["fiscal_year"] == latest_year, "fiscal_week"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            prev_week = max([w for w in weeks_in_year if w < int(latest_week)], default=None)

            wow_dmg_units_pct = None
            if prev_week is not None and curr_dmg_units is not None:
                prev_df = filtered_returns[
                    (filtered_returns["fiscal_year"] == latest_year)
                    & (filtered_returns["fiscal_week"] == prev_week)
                    & (filtered_returns["part_number"] == part)
                    & (filtered_returns["gross_units"] >= volume_gate)
                ].copy()

                if not prev_df.empty:
                    # compute damaged units in prev week (use actual column if present, else estimate)
                    if "damaged_units" in prev_df.columns and prev_df["damaged_units"].notna().any():
                        prev_df["damaged_units_calc"] = prev_df["damaged_units"].fillna(0)
                    else:
                        prev_df["damaged_units_calc"] = prev_df["damaged_rate"].fillna(0) * prev_df["gross_units"].fillna(0)

                    prev_dmg_units = float(prev_df["damaged_units_calc"].sum())
                    if prev_dmg_units != 0:
                        wow_dmg_units_pct = (curr_dmg_units - prev_dmg_units) / prev_dmg_units

            pill_text = (
                f"Damaged Units WoW%: {wow_dmg_units_pct:+.2%}"
                if wow_dmg_units_pct is not None
                else "Damaged Units WoW%: N/A"
            )
            arrow = ""
            color = "#6b7280"
            if wow_dmg_units_pct is not None:
                if wow_dmg_units_pct > 0:
                    arrow, color = "▲", "#065f46"
                elif wow_dmg_units_pct < 0:
                    arrow, color = "▼", "#b91c1c"

            if wow_dmg_units_pct is None:
                pill_body = "Damaged Units WoW N/A"
            else:
                pill_body = f"{arrow} Damaged Units WoW {wow_dmg_units_pct:+.2%}"

            wow_pill = (
                f"<span style='background-color:#e0f2fe;color:{color};"
                "padding:4px 10px;border-radius:14px;font-size:12px;font-weight:600;'>"
                f"{pill_body}</span>"
            )

            card_html = (
                "<div style='border:1px solid #e5e7eb;border-radius:10px;padding:12px;'>"
                "<div style='font-weight:600;font-size:14px;color:#374151;margin-bottom:6px;'>Top Risk SKU</div>"
                f"<div style='font-size:22px;font-weight:700;color:#111827;margin-bottom:6px;'>{str(part)}</div>"
                f"{wow_pill}"
                "</div>"
            )
            cols[2].markdown(card_html, unsafe_allow_html=True)

        st.subheader("Damaged Rate Trend")
        # Apply volume gate consistently
        returns_vg = filtered_returns[filtered_returns["gross_units"] >= volume_gate]
        returns_vg = returns_vg.copy()
        if "damaged_units" not in returns_vg.columns:
            returns_vg["damaged_units"] = pd.NA
        if "undamaged_units" not in returns_vg.columns:
            returns_vg["undamaged_units"] = pd.NA
        returns_vg["damaged_units"] = returns_vg["damaged_units"].fillna(0)
        returns_vg["undamaged_units"] = returns_vg["undamaged_units"].fillna(0)
        # Backfill missing units
        if "damaged_units" in returns_vg.columns and "undamaged_units" in returns_vg.columns:
            missing_undmg = returns_vg["undamaged_units"].isna() | returns_vg["undamaged_units"].eq(0)
            missing_dmg = returns_vg["damaged_units"].isna() | returns_vg["damaged_units"].eq(0)
            returns_vg.loc[missing_undmg & (~returns_vg["gross_units"].isna()), "undamaged_units"] = (
                returns_vg["gross_units"] - returns_vg["damaged_units"]
            )
            returns_vg.loc[missing_dmg & (~returns_vg["gross_units"].isna()), "damaged_units"] = (
                returns_vg["gross_units"] - returns_vg["undamaged_units"]
            )
        if not returns_vg.empty:
            part_week = (
                returns_vg.groupby(["part_number", "fiscal_year", "fiscal_week"])
                .agg(damaged_units=("damaged_units", "sum"), undamaged_units=("undamaged_units", "sum"), gross_units=("gross_units", "sum"))
                .reset_index()
            )
            part_week["damaged_rate_calc"] = part_week.apply(
                lambda r: r["damaged_units"] / r["gross_units"] if r["gross_units"] and r["gross_units"] != 0 else pd.NA,
                axis=1,
            )
            part_week["undamaged_units"] = part_week.apply(
                lambda r: r["undamaged_units"] if pd.notna(r["undamaged_units"]) else r["gross_units"] - r["damaged_units"],
                axis=1,
            )
            part_week["undamaged_rate_calc"] = part_week.apply(
                lambda r: r["undamaged_units"] / r["gross_units"] if r["gross_units"] and r["gross_units"] != 0 else pd.NA,
                axis=1,
            )
            total_week = (
                part_week.groupby(["fiscal_year", "fiscal_week"])[["damaged_units", "undamaged_units", "gross_units"]]
                .sum()
                .reset_index()
            )
            total_week["damaged_rate_calc"] = total_week.apply(
                lambda r: r["damaged_units"] / r["gross_units"] if r["gross_units"] and r["gross_units"] != 0 else pd.NA,
                axis=1,
            )
            total_week["undamaged_rate_calc"] = total_week.apply(
                lambda r: r["undamaged_units"] / r["gross_units"] if r["gross_units"] and r["gross_units"] != 0 else pd.NA,
                axis=1,
            )
            total_week["series"] = "Total"
            total_week_dmg = total_week[["series", "fiscal_year", "fiscal_week", "damaged_rate_calc"]]
            total_week_undmg = total_week[["series", "fiscal_year", "fiscal_week", "undamaged_rate_calc"]]
            part_week["series"] = part_week["part_number"]
            part_week_dmg = part_week[["series", "fiscal_year", "fiscal_week", "damaged_rate_calc"]]
            part_week_undmg = part_week[["series", "fiscal_year", "fiscal_week", "undamaged_rate_calc"]]
            trend_long = pd.concat([total_week_dmg, part_week_dmg], ignore_index=True)
            undmg_long = pd.concat([total_week_undmg, part_week_undmg], ignore_index=True)
            # Fallback if units missing: use weighted percent if available
            if trend_long.empty and "damaged_rate" in returns_vg.columns:
                tmp = (
                    returns_vg.groupby(["part_number", "fiscal_year", "fiscal_week"])
                    .apply(lambda g: (g["damaged_rate"] * g["gross_units"]).sum() / g["gross_units"].sum() if g["gross_units"].sum() else pd.NA)
                    .reset_index(name="damaged_rate_calc")
                )
                tmp["series"] = tmp["part_number"]
                total_tmp = (
                    returns_vg.groupby(["fiscal_year", "fiscal_week"])
                    .apply(lambda g: (g["damaged_rate"] * g["gross_units"]).sum() / g["gross_units"].sum() if g["gross_units"].sum() else pd.NA)
                    .reset_index(name="damaged_rate_calc")
                )
                total_tmp["series"] = "Total"
                trend_long = pd.concat([total_tmp, tmp], ignore_index=True)
            if undmg_long.empty and "undamaged_rate" in returns_vg.columns:
                tmpu = (
                    returns_vg.groupby(["part_number", "fiscal_year", "fiscal_week"])
                    .apply(lambda g: (g["undamaged_rate"] * g["gross_units"]).sum() / g["gross_units"].sum() if g["gross_units"].sum() else pd.NA)
                    .reset_index(name="undamaged_rate_calc")
                )
                tmpu["series"] = tmpu["part_number"]
                total_tmpu = (
                    returns_vg.groupby(["fiscal_year", "fiscal_week"])
                    .apply(lambda g: (g["undamaged_rate"] * g["gross_units"]).sum() / g["gross_units"].sum() if g["gross_units"].sum() else pd.NA)
                    .reset_index(name="undamaged_rate_calc")
                )
                total_tmpu["series"] = "Total"
                undmg_long = pd.concat([total_tmpu, tmpu], ignore_index=True)
            trend_long = trend_long.dropna(subset=["damaged_rate_calc"])
            undmg_long = undmg_long.dropna(subset=["undamaged_rate_calc"])
            if not trend_long.empty:
                sel = alt.selection_point(fields=["series"], bind="legend")
                chart = (
                    alt.Chart(trend_long)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("fiscal_week:O", title="Fiscal Week", axis=alt.Axis(tickMinStep=1, format="d")),
                        y=alt.Y("damaged_rate_calc:Q", axis=alt.Axis(format=".2%"), title="Damaged Rate"),
                        color="series:N",
                        opacity=alt.condition(sel, alt.value(1), alt.value(0.2)),
                        tooltip=[
                            "series",
                            "fiscal_year",
                            "fiscal_week",
                            alt.Tooltip("damaged_rate_calc:Q", format=".2%"),
                        ],
                    )
                    .add_params(sel)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No damaged rate data after applying volume gate.")

            st.subheader("Undamaged Rate Trend")
            if not undmg_long.empty:
                sel2 = alt.selection_point(fields=["series"], bind="legend")
                chart_undmg = (
                    alt.Chart(undmg_long)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("fiscal_week:O", title="Fiscal Week", axis=alt.Axis(tickMinStep=1, format="d")),
                        y=alt.Y("undamaged_rate_calc:Q", axis=alt.Axis(format=".2%"), title="Undamaged Rate"),
                        color="series:N",
                        opacity=alt.condition(sel2, alt.value(1), alt.value(0.2)),
                        tooltip=[
                            "series",
                            "fiscal_year",
                            "fiscal_week",
                            alt.Tooltip("undamaged_rate_calc:Q", format=".2%"),
                        ],
                    )
                    .add_params(sel2)
                )
                st.altair_chart(chart_undmg, use_container_width=True)
            else:
                st.info("No undamaged rate data after applying volume gate.")
        else:
            st.info("No returns data after applying volume gate.")

        st.subheader("Top Risk SKUs")
        ret_top = (
            filtered_returns[filtered_returns["gross_units"] >= volume_gate]
            .sort_values("damaged_rate", ascending=False)
            .head(top_n)
        )
        ret_top = ret_top.reset_index(drop=True)
        ret_top.insert(0, "rank", ret_top.index + 1)
        st.dataframe(ret_top, use_container_width=True, hide_index=True)


# ----- Profitability & Billbacks -----
with tab_finance:
    st.subheader("Gross Margin by SKU")
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
        prod_gm = (
            margin.dropna(subset=["part_number"])
            .groupby("part_number")
            .agg(gm=("gross_margin", "sum"), revenue=("revenue", "sum"))
            .reset_index()
            .sort_values("gm", ascending=False)
        )
        cat_gm = (
            margin.groupby("major_category")
            .agg(gm=("gross_margin", "sum"), revenue=("revenue", "sum"))
            .reset_index()
            .sort_values("gm", ascending=False)
            if "major_category" in margin.columns
            else pd.DataFrame()
        )
        extra = st.columns(2)
        if not prod_gm.empty:
            extra[0].metric("Top GM Product", str(prod_gm.iloc[0]["part_number"]), delta=f"${prod_gm.iloc[0]['gm']:,.0f}")
        if not cat_gm.empty:
            extra[1].metric("Top GM Category", str(cat_gm.iloc[0]["major_category"]), delta=f"${cat_gm.iloc[0]['gm']:,.0f}")
        top_margin = margin.sort_values("gross_margin", ascending=False).head(top_n)
        top_margin = top_margin.reset_index(drop=True)
        top_margin.insert(0, "rank", top_margin.index + 1)
        display_df = top_margin[
            ["rank", "part_number", "item_id", "description", "revenue", "cogs", "gross_margin", "gross_margin_pct"]
        ]
        display_df = format_currency_columns(display_df, ["revenue", "cogs", "gross_margin"])
        display_df = format_percent_columns(display_df, ["gross_margin_pct"], decimals=1)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Billbacks Trend")
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
        # Explicit fallback for known P9 code
        p9_mask = mapped["code_norm"] == "P9"
        mapped.loc[p9_mask, "bucket"] = mapped.loc[p9_mask, "bucket"].fillna("Promotional allowance")
        mapped.loc[p9_mask, "title"] = mapped.loc[p9_mask, "title"].fillna("Promotional allowance")
        mapped.loc[p9_mask, "direction"] = mapped.loc[p9_mask, "direction"].fillna("allowance")
        if "bucket" not in mapped.columns:
            mapped["bucket"] = None
        mapped["bucket"] = mapped["bucket"].fillna("Unmapped")
        weekly = (
            mapped.assign(invoice_week_start=lambda d: d["invoice_date"] - pd.to_timedelta(d["invoice_date"].dt.weekday, unit="D"))
            .groupby(["invoice_week_start", "bucket"])["billback_amount"]
            .sum()
            .reset_index()
            .sort_values("invoice_week_start")
        )
        if not weekly.empty:
            # Optional invoice date filter
            min_dt = weekly["invoice_week_start"].min().date()
            max_dt = weekly["invoice_week_start"].max().date()
            inv_range = st.date_input("Invoice date range", value=(min_dt, max_dt))
            if isinstance(inv_range, tuple) and len(inv_range) == 2:
                start_dt, end_dt = inv_range
                weekly = weekly[
                    (weekly["invoice_week_start"].dt.date >= start_dt) & (weekly["invoice_week_start"].dt.date <= end_dt)
                ]
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
        st.caption("Billbacks by normalized reason bucket; use mapping file for consistent grouping.")
        billback_total = mapped["billback_amount"].sum()
        sales_total = filtered_sales["revenue"].sum()
        billback_rate = billback_total / sales_total if sales_total else None
        st.metric("Billbacks % of Sales (selected weeks)", f"{billback_rate:.2%}" if billback_rate else "NA")
        unmapped = mapped[mapped["bucket"] == "Unmapped"]
        if not unmapped.empty:
            st.markdown("**Unmapped Type Codes (by $ impact)**")
            um = (
                unmapped.groupby("type_code")["billback_amount"]
                .sum()
                .reset_index()
                .sort_values("billback_amount", ascending=False)
            )
            st.dataframe(um, use_container_width=True, hide_index=True)

st.caption("Add new FYxxFWxx files to extend the time series; dashboards refresh automatically.")

with st.expander("Data Quality (debug)"):
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
            st.markdown("**All unmapped product numbers (unique)**")
            st.dataframe(
                unmapped_details[["part_number"]].drop_duplicates().rename(columns={"part_number": "unmapped_part_number"}),
                hide_index=True,
            )
    if not cpfr.empty:
        st.markdown("**CPFR requested/shipped non-null %**")
        req_pct = cpfr["requested_units"].notna().mean() * 100
        ship_pct = cpfr["shipped_units"].notna().mean() * 100
        st.write({"requested_units_non_null_pct": f"{req_pct:.1f}%", "shipped_units_non_null_pct": f"{ship_pct:.1f}%"})
