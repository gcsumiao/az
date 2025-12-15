from __future__ import annotations

import re
from dataclasses import asdict
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from core.filters import DashboardFilters, normalize_filters


DATA_DIR = Path(__file__).resolve().parents[1]
FILE_GLOB = "Innova-AZ FY*FW*.xlsx"

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


def get_source_files() -> List[Path]:
    return sorted(DATA_DIR.glob(FILE_GLOB))


def file_signature(files: List[Path]) -> Tuple[Tuple[str, float], ...]:
    return tuple((f.name, f.stat().st_mtime) for f in files)


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


def find_header_row(df: pd.DataFrame, keywords: Iterable[str], search_rows: int = 25) -> Optional[int]:
    lowered = [k.lower() for k in keywords]
    for idx in range(min(search_rows, len(df))):
        row = df.iloc[idx].astype(str).str.lower().tolist()
        if any(k in " ".join(row) for k in lowered):
            return idx
    return None


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()]


def column_as_series(df: pd.DataFrame, col: str) -> pd.Series:
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
    return drop_duplicate_columns(df)


def numericize(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def coerce_str_safe(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = series.astype("string").str.strip()
            series = series.replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
            df[col] = series
    return df


def normalize_sku(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "<na>", "na", "n/a"}:
        return None
    return s


def normalize_sku_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(normalize_sku).astype("string")
    return df


def drop_invalid_products(df: pd.DataFrame, *, drop_blank_part: bool = True) -> pd.DataFrame:
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
        df = df.dropna(subset=["part_number", "item_id"], how="all")
    return df


def ensure_week_cols(df: pd.DataFrame, week_col: str = "fiscal_week") -> pd.DataFrame:
    if week_col in df.columns:
        df[week_col] = pd.to_numeric(df[week_col], errors="coerce").astype("Int64")
    return df


def round_half_up(value: object, ndigits: int = 0) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    q = Decimal(10) ** -ndigits
    return float(Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP))


def format_currency_0(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${float(value):,.0f}"


def format_currency_columns(df: pd.DataFrame, cols: Iterable[str], decimals: int = 0) -> pd.DataFrame:
    formatted = df.copy()
    for c in cols:
        if c in formatted.columns:
            formatted[c] = formatted[c].apply(lambda v: f"${float(v):,.{decimals}f}" if pd.notna(v) else "")
    return formatted


def format_percent_columns(df: pd.DataFrame, cols: Iterable[str], decimals: int = 0) -> pd.DataFrame:
    formatted = df.copy()
    for c in cols:
        if c in formatted.columns:
            formatted[c] = formatted[c].apply(lambda v: f"{float(v)*100:.{decimals}f}%" if pd.notna(v) else "")
    return formatted


def get_snapshot_context(gt_df: pd.DataFrame) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if gt_df.empty or "fiscal_year" not in gt_df.columns or "fiscal_week" not in gt_df.columns:
        return None, None, None
    latest_year = int(pd.to_numeric(gt_df["fiscal_year"], errors="coerce").max())
    year_df = gt_df[gt_df["fiscal_year"] == latest_year]
    weeks = sorted(pd.to_numeric(year_df["fiscal_week"], errors="coerce").dropna().astype(int).unique())
    if not weeks:
        return latest_year, None, None
    snapshot_week = weeks[-1]
    prev_week = weeks[-2] if len(weeks) >= 2 else None
    return latest_year, snapshot_week, prev_week


def normalize_fee_code(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    s = re.sub(r"\s+", "", str(value).strip()).upper()
    if not s:
        return None
    return s


# ---------------- Loaders ----------------
def load_sales_actual_tyly(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    frames: List[pd.DataFrame] = []
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


def load_cpfr(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    records: List[Dict[str, float]] = []
    for path in files:
        snapshot_year, snapshot_week = parse_fiscal_from_name(path.name)
        raw = pd.read_excel(path, sheet_name="CPFR", header=None)
        if raw.empty or len(raw) < 13:
            continue
        week_row = raw.iloc[8]
        fill_row = raw.iloc[9]
        shipped_row = raw.iloc[10]
        requested_row = raw.iloc[11]
        not_shipped_row = raw.iloc[12]
        for col_idx, week_code in enumerate(week_row):
            fiscal_year, fiscal_week = parse_week_code(week_code)
            if fiscal_year is None or fiscal_week is None:
                continue
            records.append(
                {
                    "snapshot_year": snapshot_year,
                    "snapshot_week": snapshot_week,
                    "fiscal_year": fiscal_year,
                    "fiscal_week": fiscal_week,
                    "fill_rate": pd.to_numeric(fill_row.iloc[col_idx], errors="coerce"),
                    "shipped_units": pd.to_numeric(shipped_row.iloc[col_idx], errors="coerce"),
                    "requested_units": pd.to_numeric(requested_row.iloc[col_idx], errors="coerce"),
                    "not_shipped_units": pd.to_numeric(not_shipped_row.iloc[col_idx], errors="coerce"),
                }
            )
    return pd.DataFrame(records)


def load_cpfr_detail(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    frames: List[pd.DataFrame] = []
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


def load_redflags(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    frames: List[pd.DataFrame] = []
    for path in files:
        try:
            raw = pd.read_excel(path, sheet_name="Fill Rate Red Flags", header=None)
        except Exception:
            continue
        if raw.empty or raw.shape[0] < 10 or raw.shape[1] < 6:
            continue
        header_row_idx = 8
        hdr = raw.iloc[header_row_idx].astype(str).str.upper().fillna("")

        def _col_idx(pattern: str, fallback: int) -> int:
            hits = [i for i, v in enumerate(hdr) if re.search(pattern, v)]
            return hits[0] if hits else fallback

        col_part = _col_idx(r"\bPART\s*NUMBER\b", 0)
        col_lfw = _col_idx(r"\bLFW\b.*NOT\s*SHIP", 3)
        col_l4w = _col_idx(r"\bL4FW\b.*NOT\s*SHIP", 4)
        col_l52w = _col_idx(r"\bL52FW\b.*NOT\s*SHIP", 5)
        needed = [col_part, col_lfw, col_l4w, col_l52w]
        if max(needed) >= raw.shape[1]:
            continue

        end_row = min(46, raw.shape[0])
        df = raw.iloc[header_row_idx + 1 : end_row, needed].copy()
        df.columns = ["part_number", "not_shipped_lfw", "not_shipped_l4w", "not_shipped_l52w"]
        df = coerce_str_safe(df, ["part_number"])
        df = normalize_sku_columns(df, ["part_number"])
        df = numericize(df, ["not_shipped_lfw", "not_shipped_l4w", "not_shipped_l52w"])
        df = drop_invalid_products(df)
        df["snapshot_file"] = path.name
        fiscal_year, fiscal_week = parse_fiscal_from_name(path.name)
        df["fiscal_year"] = fiscal_year
        df["fiscal_week"] = fiscal_week
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_outs(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    frames: List[pd.DataFrame] = []
    for path in files:
        raw = pd.read_excel(path, sheet_name="Outs", header=None)
        if raw.empty:
            continue
        header_row = find_header_row(raw, ["Part", "Exposure"], search_rows=25) or 1
        df = pd.read_excel(path, sheet_name="Outs", header=header_row)
        df.columns = [str(c).strip() for c in df.columns]
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if "part" in cl and "number" in cl:
                rename[c] = "part_number"
            elif "exposure" in cl:
                rename[c] = "store_oos_exposure"
        df = df.rename(columns=rename)
        if "part_number" not in df.columns or "store_oos_exposure" not in df.columns:
            continue
        df = df[["part_number", "store_oos_exposure"]].copy()
        df = coerce_str_safe(df, ["part_number"])
        df = normalize_sku_columns(df, ["part_number"])
        df = numericize(df, ["store_oos_exposure"])
        df = drop_invalid_products(df)
        fiscal_year, fiscal_week = parse_fiscal_from_name(path.name)
        df["fiscal_year"] = fiscal_year
        df["fiscal_week"] = fiscal_week
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_outs_totals(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    records: List[Dict[str, object]] = []
    for path in files:
        raw = pd.read_excel(path, sheet_name="Outs", header=None)
        if raw.empty:
            continue
        # Attempt to locate a Grand Total row for exposure
        raw_str = raw.astype("string").fillna("").apply(lambda s: s.str.strip())
        gt_rows = raw_str.apply(lambda row: row.str.contains("grand total", case=False, na=False).any(), axis=1)
        if not gt_rows.any():
            continue
        gt_idx = int(gt_rows.idxmax())
        exposure_vals = pd.to_numeric(raw.iloc[gt_idx], errors="coerce")
        total_exposure = float(exposure_vals.max()) if exposure_vals.notna().any() else None
        fiscal_year, fiscal_week = parse_fiscal_from_name(path.name)
        records.append({"fiscal_year": fiscal_year, "fiscal_week": fiscal_week, "store_oos_exposure_total": total_exposure})
    return pd.DataFrame(records)


def load_inventory(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    frames: List[pd.DataFrame] = []
    for path in files:
        raw = pd.read_excel(path, sheet_name="Inventory", header=None)
        if raw.empty:
            continue
        header_row = find_header_row(raw, ["WOH", "Weeks"], search_rows=25) or 1
        df = pd.read_excel(path, sheet_name="Inventory", header=header_row)
        df.columns = [str(c).strip() for c in df.columns]
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if "dept" in cl:
                rename[c] = "dept"
            elif "weeks" in cl and ("hand" in cl or "on" in cl or "woh" in cl):
                rename[c] = "weeks_on_hand"
        df = df.rename(columns=rename)
        if "weeks_on_hand" not in df.columns:
            continue
        keep = [c for c in ["dept", "weeks_on_hand"] if c in df.columns]
        df = df[keep].copy()
        df = numericize(df, ["weeks_on_hand"])
        fiscal_year, fiscal_week = parse_fiscal_from_name(path.name)
        df["fiscal_year"] = fiscal_year
        df["fiscal_week"] = fiscal_week
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_returns(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    frames: List[pd.DataFrame] = []
    for path in files:
        raw = pd.read_excel(path, sheet_name="Returns", header=None)
        if raw.empty:
            continue
        header_row = find_header_row(raw, ["Part", "Gross"], search_rows=25) or 1
        df = pd.read_excel(path, sheet_name="Returns", header=header_row)
        df.columns = [str(c).strip() for c in df.columns]
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if cl.strip() in {"part", "part number", "part_number"} or ("part" in cl and "number" in cl):
                rename[c] = "part_number"
            elif cl.strip() in {"item", "item id", "item_id"} or ("item" in cl and "id" in cl):
                rename[c] = "item_id"
            elif "description" in cl:
                rename[c] = "description"
            elif "gross" in cl and "unit" in cl:
                rename[c] = "gross_units"
            elif "undamaged" in cl and ("rate" in cl or "%" in cl or "pct" in cl):
                rename[c] = "undamaged_rate"
            elif "damaged" in cl and ("rate" in cl or "%" in cl or "pct" in cl):
                rename[c] = "damaged_rate"
        df = df.rename(columns=rename)
        needed = {"part_number", "gross_units", "damaged_rate", "undamaged_rate"}
        if not needed.issubset(df.columns):
            continue
        keep = ["part_number"]
        for extra in ["item_id", "description"]:
            if extra in df.columns:
                keep.append(extra)
        keep += ["gross_units", "damaged_rate", "undamaged_rate"]
        df = df[keep].copy()
        df = coerce_str_safe(df, [c for c in ["part_number", "item_id", "description"] if c in df.columns])
        df = normalize_sku_columns(df, [c for c in ["part_number", "item_id"] if c in df.columns])
        df = numericize(df, ["gross_units", "damaged_rate", "undamaged_rate"])
        df = drop_invalid_products(df)
        fiscal_year, fiscal_week = parse_fiscal_from_name(path.name)
        df["fiscal_year"] = fiscal_year
        df["fiscal_week"] = fiscal_week
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_tyly_grand_total(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    records: List[Dict[str, object]] = []
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


def load_cost_dim(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    files = [DATA_DIR / name for name, _ in files_sig]
    if not files:
        return pd.DataFrame()
    latest = files[-1]
    df = pd.read_excel(latest, sheet_name="Cost File", header=13)
    base_cols = [c for c in df.columns if isinstance(c, (int, float, str)) and str(c).isdigit()]
    df["unit_cost"] = df[base_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) if base_cols else pd.NA
    df = df.rename(columns={"Part Nbr": "part_number", "Item Nbr": "item_id", "Description": "description"})
    keep = [c for c in ["part_number", "item_id", "description", "unit_cost"] if c in df.columns]
    df = df[keep]
    df = df.dropna(subset=["part_number", "item_id"])
    df = numericize(df, ["unit_cost"])
    df = coerce_str_safe(df, ["part_number", "item_id", "description"])
    df = normalize_sku_columns(df, ["part_number", "item_id"])
    df = drop_invalid_products(df)
    return df


def load_billbacks(files_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
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
                if float(val) <= 60000:
                    return pd.to_datetime(val, unit="D", origin="1899-12-30", errors="coerce")
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
    dedupe_cols = [c for c in ["invoice_number", "type_code_norm", "invoice_date", "billback_amount"] if c in out.columns]
    if dedupe_cols:
        out = out.drop_duplicates(subset=dedupe_cols, keep="first")
    return out


def load_category_dim() -> pd.DataFrame:
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

    if "major_category" not in df.columns and "minor_dept" in df.columns and not minor_dept_bridge.empty:
        bridge = minor_dept_bridge.rename(columns=rename_map)
        if {"minor_dept", "major_category"}.issubset(bridge.columns):
            df = df.merge(bridge[["minor_dept", "major_category"]].drop_duplicates(), on="minor_dept", how="left")

    if "major_category" not in df.columns and "minor_dept" in df.columns:
        df["major_category"] = df["minor_dept"]

    required = {"major_category", "part_number"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    cols = ["major_category", "part_number"] + (["item_id"] if "item_id" in df.columns else [])
    df = coerce_str_safe(df, cols)
    df = normalize_sku_columns(df, [c for c in ["part_number", "item_id"] if c in df.columns])
    df = drop_invalid_products(df)
    out_cols = ["major_category", "part_number"] + (["item_id"] if "item_id" in df.columns else [])
    return df[out_cols]


def load_billback_reason_dim() -> pd.DataFrame:
    if BILLBACK_REASON_XLSX.exists():
        df = pd.read_excel(BILLBACK_REASON_XLSX, dtype=str, engine="openpyxl")
    elif BILLBACK_REASON_PATH.exists():
        df = pd.read_csv(BILLBACK_REASON_PATH, dtype=str, encoding="utf-8-sig")
    else:
        return pd.DataFrame()

    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {
        "code_product": "code_product",
        "code product": "code_product",
        "product code": "code_product",
        "code_pr": "code_product",
        "code": "code_product",
        "title": "title",
        "bucket": "bucket",
        "direction": "direction",
        "all_codes": "all_codes",
        "codes": "all_codes",
    }
    df = df.rename(columns=rename_map)
    if "code_product" not in df.columns:
        return pd.DataFrame()

    df["code_norm"] = df["code_product"].apply(normalize_fee_code)
    if "all_codes" in df.columns:
        df["all_codes_norm"] = df["all_codes"].fillna("").astype(str).apply(
            lambda s: [normalize_fee_code(x) for x in re.split(r"[;,\s]+", s) if normalize_fee_code(x)]
        )
    else:
        df["all_codes_norm"] = df["code_norm"].apply(lambda x: [x] if x else [])
    return df


# ---------------- Compute helpers ----------------
def compute_margin(fact_sales: pd.DataFrame, cost_dim: pd.DataFrame, fiscal_weeks: List[int]) -> pd.DataFrame:
    if fact_sales.empty or cost_dim.empty:
        return pd.DataFrame()
    df = fact_sales.copy()
    if fiscal_weeks and "fiscal_week" in df.columns:
        df = df[df["fiscal_week"].isin(fiscal_weeks)]
    df = df.dropna(subset=["part_number", "item_id"])
    df = df[~df["part_number"].astype(str).str.contains("TOTAL", case=False, na=False)]
    df = df.merge(cost_dim[["part_number", "item_id", "unit_cost"]], on=["part_number", "item_id"], how="left")
    df["cogs"] = df["units"] * df["unit_cost"]
    grouped = (
        df.groupby(["part_number", "item_id", "description"])
        .agg(revenue=("revenue", "sum"), units=("units", "sum"), cogs=("cogs", "sum"))
        .reset_index()
    )
    grouped["gross_margin"] = grouped["revenue"] - grouped["cogs"]
    grouped["gross_margin_pct"] = grouped["gross_margin"] / grouped["revenue"].replace({0: pd.NA})
    return grouped


def compute_comparable_yoy(df: pd.DataFrame, snapshot_week: int, dimension: str, min_ly_rev_floor: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    wk = df[df["fiscal_week"] == snapshot_week].copy()
    if dimension not in wk.columns:
        return pd.DataFrame()
    wk = wk.dropna(subset=[dimension])
    wk[dimension] = wk[dimension].astype(str).str.strip()
    wk = wk[wk[dimension].ne("")]
    grouped = (
        wk.groupby([dimension])
        .agg(ty_rev=("revenue", "sum"), ly_rev=("ly_revenue", "sum"), ty_units=("units", "sum"), ly_units=("ly_units", "sum"))
        .reset_index()
    )
    grouped["yoy_rev_pct_comp"] = (grouped["ty_rev"] - grouped["ly_rev"]) / grouped["ly_rev"]
    bad_mask = (grouped["ly_rev"] <= 0) & (grouped["ty_rev"] > 0)
    zero_mask = grouped["ly_rev"] == 0
    small_mask = grouped["ly_rev"].abs() < float(min_ly_rev_floor)
    grouped.loc[bad_mask | zero_mask | small_mask, "yoy_rev_pct_comp"] = pd.NA
    return grouped


def compute_wow_pct(df: pd.DataFrame, snap_week: int, prev_week: Optional[int], dimension: str) -> pd.DataFrame:
    if df.empty or prev_week is None:
        return pd.DataFrame()
    if dimension not in df.columns:
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
    if df.empty or dimension not in df.columns:
        return pd.DataFrame()
    wk = df[df["fiscal_week"] == week].copy()
    wk = wk.dropna(subset=[dimension])
    wk[dimension] = wk[dimension].astype(str).str.strip()
    wk = wk[wk[dimension].ne("")]
    grouped = (
        wk.groupby([dimension])
        .agg(fytd_rev=("fytd_revenue", "sum"), fytd_yoy_pct=("fytd_revenue_diff_pct", "max"))
        .reset_index()
    )
    grouped["ly_est"] = grouped.apply(
        lambda r: r["fytd_rev"] / (1 + r["fytd_yoy_pct"])
        if pd.notna(r["fytd_yoy_pct"]) and (1 + r["fytd_yoy_pct"]) != 0
        else pd.NA,
        axis=1,
    )
    bad_mask = grouped["ly_est"].le(0) | grouped["ly_est"].abs().lt(float(min_ly_rev_floor))
    grouped.loc[bad_mask, "fytd_yoy_pct"] = pd.NA
    return grouped


def compute_ytd_wow(df: pd.DataFrame, snap_week: int, prev_week: Optional[int], dimension: str) -> pd.DataFrame:
    if df.empty or prev_week is None or dimension not in df.columns:
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


def compute_alerts(
    *,
    fact_sales: pd.DataFrame,
    cpfr: pd.DataFrame,
    outs: pd.DataFrame,
    inventory: pd.DataFrame,
    forecast: pd.DataFrame,
    orders: pd.DataFrame,
    billbacks: pd.DataFrame,
    thresholds: dict,
) -> List[Dict[str, str]]:
    alerts: List[Dict[str, str]] = []

    latest_week = None
    if not fact_sales.empty and "fiscal_week" in fact_sales.columns:
        latest_week = int(pd.to_numeric(fact_sales["fiscal_week"], errors="coerce").max())

    if latest_week and not cpfr.empty:
        latest_rows = cpfr[cpfr["fiscal_week"] == latest_week]
        fill_rate = latest_rows["fill_rate"].mean(skipna=True) if "fill_rate" in latest_rows.columns else None
        if fill_rate is not None and pd.notna(fill_rate) and fill_rate < thresholds["fill_rate"]:
            alerts.append(
                {
                    "alert_type": "Fill Rate",
                    "severity": "high",
                    "message": f"Fill rate {float(fill_rate):.1%} below target {thresholds['fill_rate']:.0%} for FW{latest_week}.",
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
        # forecast/orders use target_week; prefer those if present
        fc_week = "target_week" if "target_week" in forecast.columns else "fiscal_week"
        op_week = "target_week" if "target_week" in orders.columns else "fiscal_week"
        fc = forecast[forecast[fc_week] == latest_week]["forecast_units"].sum()
        op = orders[orders[op_week] == latest_week]["order_units"].sum()
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


# ---------------- Public API (Streamlit parity + FastAPI use) ----------------
@lru_cache(maxsize=4)
def _load_dashboard_data_cached(files_sig: Tuple[Tuple[str, float], ...]) -> Dict[str, object]:
    fact_sales = load_sales_actual_tyly(files_sig)
    units_yoy = load_units_yoy(files_sig)
    store_counts = load_store_counts(files_sig)
    forecast_totals, order_totals = load_forecast(files_sig)
    cpfr = load_cpfr(files_sig)
    cpfr_detail = load_cpfr_detail(files_sig)
    redflags = load_redflags(files_sig)
    outs = load_outs(files_sig)
    outs_totals = load_outs_totals(files_sig)
    inventory = load_inventory(files_sig)
    returns = load_returns(files_sig)
    cost_dim = load_cost_dim(files_sig)
    billbacks = load_billbacks(files_sig)
    dim_category = load_category_dim()
    dim_billback_reason = load_billback_reason_dim()
    tyly_grand_total = load_tyly_grand_total(files_sig)

    weeks = sorted(pd.to_numeric(fact_sales.get("fiscal_week", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique())

    return {
        "files": [name for name, _ in files_sig],
        "weeks": weeks,
        "fact_sales": fact_sales,
        "units_yoy": units_yoy,
        "store_counts": store_counts,
        "forecast_totals": forecast_totals,
        "order_totals": order_totals,
        "cpfr": cpfr,
        "cpfr_detail": cpfr_detail,
        "redflags": redflags,
        "outs": outs,
        "outs_totals": outs_totals,
        "inventory": inventory,
        "returns": returns,
        "cost_dim": cost_dim,
        "billbacks": billbacks,
        "dim_category": dim_category,
        "dim_billback_reason": dim_billback_reason,
        "tyly_grand_total": tyly_grand_total,
    }


def load_dashboard_data() -> Dict[str, object]:
    files = get_source_files()
    if not files:
        return {"files": [], "weeks": [], "fact_sales": pd.DataFrame()}
    return _load_dashboard_data_cached(file_signature(files))


def prepare_context(filters: dict | DashboardFilters, data_ctx: Dict[str, object]) -> Dict[str, object]:
    fact_sales: pd.DataFrame = data_ctx.get("fact_sales", pd.DataFrame()).copy()
    dim_category: pd.DataFrame = data_ctx.get("dim_category", pd.DataFrame()).copy()

    dq_removed_rows = 0
    dq_bad_tokens_remaining = 0

    # Attach categories if possible.
    if not fact_sales.empty:
        if not dim_category.empty and {"part_number", "major_category"}.issubset(dim_category.columns):
            dim = dim_category.copy()
            dim_cols = ["part_number", "major_category"] + (["item_id"] if "item_id" in dim.columns else [])
            dim = coerce_str_safe(dim, dim_cols)
            dim = normalize_sku_columns(dim, [c for c in ["part_number", "item_id"] if c in dim.columns])

            if "item_id" in dim.columns and "item_id" in fact_sales.columns and dim["item_id"].notna().any():
                key_dim = dim.dropna(subset=["part_number", "item_id"]).drop_duplicates(subset=["part_number", "item_id"])
                fact_sales = fact_sales.merge(
                    key_dim[["part_number", "item_id", "major_category"]],
                    on=["part_number", "item_id"],
                    how="left",
                )

            # Fallback mapping by part number only (handles item_id mismatch or missing item_id).
            part_map = (
                dim.dropna(subset=["part_number", "major_category"])
                .drop_duplicates(subset=["part_number"])
                .rename(columns={"major_category": "major_category_part"})[["part_number", "major_category_part"]]
            )
            fact_sales = fact_sales.merge(part_map, on="part_number", how="left")
            if "major_category" in fact_sales.columns and "major_category_part" in fact_sales.columns:
                fact_sales["major_category"] = fact_sales["major_category"].fillna(fact_sales["major_category_part"])
            fact_sales = fact_sales.drop(columns=["major_category_part"], errors="ignore")
        if "major_category" not in fact_sales.columns:
            fact_sales["major_category"] = pd.NA
        fact_sales["major_category"] = fact_sales["major_category"].fillna("Unmapped")

    available_weeks = data_ctx.get("weeks") or sorted(pd.to_numeric(fact_sales.get("fiscal_week", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique())
    filt = filters if isinstance(filters, DashboardFilters) else normalize_filters(filters, available_weeks=available_weeks)

    filtered_sales = fact_sales.copy()
    if not filtered_sales.empty and filt.selected_weeks and "fiscal_week" in filtered_sales.columns:
        filtered_sales = filtered_sales[filtered_sales["fiscal_week"].isin(filt.selected_weeks)]

    cats = [c for c in filt.selected_categories if c and c != "All Categories"]
    if cats and "major_category" in filtered_sales.columns:
        filtered_sales = filtered_sales[filtered_sales["major_category"].isin(cats)]

    if filt.selected_parts:
        filtered_sales = filtered_sales[filtered_sales["part_number"].astype(str).isin(set(filt.selected_parts))]

    if filt.sku_query:
        q = filt.sku_query.lower()
        mask = (
            filtered_sales["part_number"].astype(str).str.lower().str.contains(q, na=False)
            | filtered_sales.get("item_id", pd.Series(dtype=str)).astype(str).str.lower().str.contains(q, na=False)
        )
        filtered_sales = filtered_sales[mask]

    def filter_by_parts(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not filt.selected_parts or "part_number" not in df.columns:
            return df
        return df[df["part_number"].astype(str).isin(set(filt.selected_parts))]

    filtered_units_yoy = data_ctx.get("units_yoy", pd.DataFrame()).copy()
    if not filtered_units_yoy.empty and filt.selected_weeks and "fiscal_week" in filtered_units_yoy.columns:
        filtered_units_yoy = filtered_units_yoy[filtered_units_yoy["fiscal_week"].isin(filt.selected_weeks)]

    filtered_store_counts = data_ctx.get("store_counts", pd.DataFrame()).copy()
    if not filtered_store_counts.empty and filt.selected_weeks and "fiscal_week" in filtered_store_counts.columns:
        filtered_store_counts = filtered_store_counts[filtered_store_counts["fiscal_week"].isin(filt.selected_weeks)]
    filtered_store_counts = filter_by_parts(filtered_store_counts)

    forecast_totals = data_ctx.get("forecast_totals", pd.DataFrame()).copy()
    order_totals = data_ctx.get("order_totals", pd.DataFrame()).copy()
    # Prefer filtering by target_week when available.
    filtered_forecast = forecast_totals
    if not filtered_forecast.empty and filt.selected_weeks:
        wcol = "target_week" if "target_week" in filtered_forecast.columns else "fiscal_week"
        if wcol in filtered_forecast.columns:
            filtered_forecast = filtered_forecast[filtered_forecast[wcol].isin(filt.selected_weeks)]
    filtered_orders = order_totals
    if not filtered_orders.empty and filt.selected_weeks:
        wcol = "target_week" if "target_week" in filtered_orders.columns else "fiscal_week"
        if wcol in filtered_orders.columns:
            filtered_orders = filtered_orders[filtered_orders[wcol].isin(filt.selected_weeks)]

    forecast_actual_src = pd.DataFrame()
    if not filtered_forecast.empty and not filtered_sales.empty and filt.show_forecast_overlay:
        fc_week = "target_week" if "target_week" in filtered_forecast.columns else "fiscal_week"
        actual = filtered_sales.groupby("fiscal_week")[["units", "revenue"]].sum().reset_index()
        
        # Filter for "Diagonal" forecast (Target == Snapshot) to avoid summing duplicates
        fc_df = filtered_forecast.copy()
        if {"snapshot_year", "snapshot_week", "target_year", "target_week"}.issubset(fc_df.columns):
            fc_df = fc_df[(fc_df["snapshot_year"] == fc_df["target_year"]) & (fc_df["snapshot_week"] == fc_df["target_week"])]
            
        fc = fc_df.groupby(fc_week)[["forecast_units"]].sum().reset_index().rename(columns={fc_week: "fiscal_week"})
        forecast_actual_src = actual.merge(fc, on="fiscal_week", how="left")

    cpfr = data_ctx.get("cpfr", pd.DataFrame()).copy()
    filtered_cpfr = cpfr
    if not filtered_cpfr.empty and filt.selected_weeks and "fiscal_week" in filtered_cpfr.columns:
        filtered_cpfr = filtered_cpfr[filtered_cpfr["fiscal_week"].isin(filt.selected_weeks)]

    cpfr_detail = data_ctx.get("cpfr_detail", pd.DataFrame()).copy()
    cpfr_detail_filtered = cpfr_detail
    if not cpfr_detail_filtered.empty and filt.selected_weeks and "fiscal_week" in cpfr_detail_filtered.columns:
        cpfr_detail_filtered = cpfr_detail_filtered[cpfr_detail_filtered["fiscal_week"].isin(filt.selected_weeks)]
    cpfr_detail_filtered = filter_by_parts(cpfr_detail_filtered)

    redflags = data_ctx.get("redflags", pd.DataFrame()).copy()
    filtered_redflags = redflags
    if not filtered_redflags.empty and filt.selected_weeks and "fiscal_week" in filtered_redflags.columns:
        filtered_redflags = filtered_redflags[filtered_redflags["fiscal_week"].isin(filt.selected_weeks)]
    filtered_redflags = filter_by_parts(filtered_redflags)

    outs = data_ctx.get("outs", pd.DataFrame()).copy()
    filtered_outs = outs
    if not filtered_outs.empty and filt.selected_weeks and "fiscal_week" in filtered_outs.columns:
        filtered_outs = filtered_outs[filtered_outs["fiscal_week"].isin(filt.selected_weeks)]
    filtered_outs = filter_by_parts(filtered_outs)

    outs_totals = data_ctx.get("outs_totals", pd.DataFrame()).copy()
    filtered_outs_totals = outs_totals
    if not filtered_outs_totals.empty and filt.selected_weeks and "fiscal_week" in filtered_outs_totals.columns:
        filtered_outs_totals = filtered_outs_totals[filtered_outs_totals["fiscal_week"].isin(filt.selected_weeks)]

    inventory = data_ctx.get("inventory", pd.DataFrame()).copy()
    filtered_inventory = inventory
    if not filtered_inventory.empty and filt.selected_weeks and "fiscal_week" in filtered_inventory.columns:
        filtered_inventory = filtered_inventory[filtered_inventory["fiscal_week"].isin(filt.selected_weeks)]

    returns = data_ctx.get("returns", pd.DataFrame()).copy()
    filtered_returns = returns
    if not filtered_returns.empty and filt.selected_weeks and "fiscal_week" in filtered_returns.columns:
        filtered_returns = filtered_returns[filtered_returns["fiscal_week"].isin(filt.selected_weeks)]
    filtered_returns = filter_by_parts(filtered_returns)

    billbacks = data_ctx.get("billbacks", pd.DataFrame()).copy()
    filtered_billbacks = billbacks

    cost_dim = data_ctx.get("cost_dim", pd.DataFrame()).copy()
    filtered_cost = cost_dim
    filtered_cost = filter_by_parts(filtered_cost)

    # Exec (Overview) aliases
    exec_sales = fact_sales.copy()
    exec_sales_filtered = filtered_sales.copy()
    exec_tyly_gt = data_ctx.get("tyly_grand_total", pd.DataFrame()).copy()
    exec_tyly_gt_filtered = exec_tyly_gt
    if not exec_tyly_gt_filtered.empty and filt.selected_weeks and "fiscal_week" in exec_tyly_gt_filtered.columns:
        exec_tyly_gt_filtered = exec_tyly_gt_filtered[exec_tyly_gt_filtered["fiscal_week"].isin(filt.selected_weeks)]

    alerts = compute_alerts(
        fact_sales=filtered_sales,
        cpfr=filtered_cpfr,
        outs=filtered_outs,
        inventory=filtered_inventory,
        forecast=filtered_forecast,
        orders=filtered_orders,
        billbacks=filtered_billbacks,
        thresholds=asdict(filt.thresholds),
    )

    return {
        "filters": filt,
        "filtered_sales": filtered_sales,
        "filtered_units_yoy": filtered_units_yoy,
        "filtered_store_counts": filtered_store_counts,
        "filtered_forecast": filtered_forecast,
        "filtered_orders": filtered_orders,
        "forecast_actual_src": forecast_actual_src,
        "filtered_cpfr": filtered_cpfr,
        "cpfr_detail_filtered": cpfr_detail_filtered,
        "filtered_redflags": filtered_redflags,
        "filtered_outs": filtered_outs,
        "filtered_outs_totals": filtered_outs_totals,
        "filtered_inventory": filtered_inventory,
        "filtered_returns": filtered_returns,
        "filtered_billbacks": filtered_billbacks,
        "filtered_cost": filtered_cost,
        "exec_sales": exec_sales,
        "exec_sales_filtered": exec_sales_filtered,
        "exec_tyly_gt": exec_tyly_gt,
        "exec_tyly_gt_filtered": exec_tyly_gt_filtered,
        "forecast_totals": data_ctx.get("forecast_totals", pd.DataFrame()),
        "order_totals": data_ctx.get("order_totals", pd.DataFrame()),
        "cpfr": data_ctx.get("cpfr", pd.DataFrame()),
        "outs": data_ctx.get("outs", pd.DataFrame()),
        "outs_totals": data_ctx.get("outs_totals", pd.DataFrame()),
        "inventory": data_ctx.get("inventory", pd.DataFrame()),
        "returns": data_ctx.get("returns", pd.DataFrame()),
        "billbacks": data_ctx.get("billbacks", pd.DataFrame()),
        "dim_category": data_ctx.get("dim_category", pd.DataFrame()),
        "dim_billback_reason": data_ctx.get("dim_billback_reason", pd.DataFrame()),
        "dq_removed_rows": dq_removed_rows,
        "dq_bad_tokens_remaining": dq_bad_tokens_remaining,
        "alerts": alerts,
    }
