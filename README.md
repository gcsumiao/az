# AZ Data FW Dashboard

## Snapshot pipeline (A1)
- Export aggregated, privacy-safe JSON with `python3 export_dashboard_data.py`.
- Outputs land in `ui/public/data/{kpis,trend,alerts,heroes,actions}.json` using the latest fiscal week by default.
- Computation is shared with the Streamlit app via `python/dashboard_core.py`; no raw sales rows or identifiers are written.

## Null handling
- UI is defensive against `null`/missing fields (see `ui/lib/safe.ts`) to avoid client runtime crashes from snapshot anomalies.
- Action Center treats missing/blank categories as `Uncategorized` (shown in the category filter when present).
- Exporter normalizes `actions.json` by filling `category`, defaulting `oosRisk` to `0` (clamped to `0..1`), and dropping non-actionable rows with empty SKUs / all-null metrics.

## Local development
- One-shot launcher: `./run_local.sh` (runs exporter then `npm run dev` in `ui/`).
- Or run inside `ui/`: `npm run update-data` then `npm run dev`.
- The Next.js pages read from `/data/*.json` at runtime; regenerate snapshots before builds or deploys.

## API development (migration path)
- Start backend: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && ./run_api.sh`
- Start UI: `cd ui && npm install && npm run dev`

## Privacy & git hygiene
- Raw XLSX/CSV inputs remain local and are `.gitignore`d (`Innova-AZ FY*.xlsx`, mapping files, etc.).
- Snapshot JSONs are also ignored by default; commit them only if explicitly approved. Recreate via the exporter when needed.

## Deployment notes
- Static snapshot site: refresh data with the exporter, then `cd ui && npm run build && npm run start` or deploy with your hosting of choice.
