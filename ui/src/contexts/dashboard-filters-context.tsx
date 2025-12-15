"use client"

import * as React from "react"
import { getJson } from "@/lib/api"
import type { DashboardFilters } from "@/lib/schema"

type MetaWeeksResponse = { weeks: number[] }
type MetaCategoriesResponse = { categories: string[] }

type DashboardFiltersContextValue = {
  filters: DashboardFilters
  setFilters: (next: DashboardFilters) => void
  refresh: () => void
  refreshKey: number
  meta: {
    weeks: number[]
    categories: string[]
  }
}

const DashboardFiltersContext = React.createContext<DashboardFiltersContextValue | null>(null)

const defaultFilters: DashboardFilters = {
  selected_weeks: [],
  selected_categories: ["All Categories"],
  selected_parts: [],
  sku_query: "",
  top_n: 15,
  thresholds: {
    fill_rate: 0.9,
    outs_exposure: 200,
    coverage: 1.0,
    woh_min: 2.0,
    woh_max: 12.0,
    billbacks_pct: 0.08,
    min_ly_rev_floor: 100,
  },
  show_forecast_overlay: true,
}

export function DashboardFiltersProvider({ children }: { children: React.ReactNode }) {
  const [filters, setFilters] = React.useState<DashboardFilters>(defaultFilters)
  const [refreshKey, setRefreshKey] = React.useState(0)
  const [meta, setMeta] = React.useState<{ weeks: number[]; categories: string[] }>({ weeks: [], categories: [] })

  React.useEffect(() => {
    let cancelled = false
    async function loadMeta() {
      const [weeks, categories] = await Promise.all([
        getJson<MetaWeeksResponse>("/meta/weeks"),
        getJson<MetaCategoriesResponse>("/meta/categories"),
      ])
      if (cancelled) return
      setMeta({ weeks: weeks.weeks ?? [], categories: categories.categories ?? [] })
      setFilters((prev) => {
        if (prev.selected_weeks.length) return prev
        const w = (weeks.weeks ?? []).slice(-4)
        return { ...prev, selected_weeks: w }
      })
    }
    loadMeta().catch(() => {
      // Keep defaults; UI will still render.
    })
    return () => {
      cancelled = true
    }
  }, [])

  const refresh = React.useCallback(() => setRefreshKey((k) => k + 1), [])

  const value = React.useMemo(
    () => ({ filters, setFilters, refresh, refreshKey, meta }),
    [filters, refresh, refreshKey, meta],
  )

  return <DashboardFiltersContext.Provider value={value}>{children}</DashboardFiltersContext.Provider>
}

export function useDashboardFilters() {
  const ctx = React.useContext(DashboardFiltersContext)
  if (!ctx) throw new Error("useDashboardFilters must be used within DashboardFiltersProvider")
  return ctx
}

