"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingUp, TrendingDown, DollarSign, Package, ShoppingCart, Target } from "lucide-react"
import { cn } from "@/lib/utils"
import { postJson } from "@/lib/api"
import { OverviewResponseSchema } from "@/lib/schema"
import { useDashboardFilters } from "@/contexts/dashboard-filters-context"

type KPI = {
  label: string
  value: number | null
  deltaPct?: number | null
  status?: "good" | "warn" | "bad"
  helper?: string
  subtext?: string | null
  format?: "percent" | "currency" | "number"
}

const iconMap = {
  "Weekly Revenue": DollarSign,
  "Units Sold": ShoppingCart,
  "Avg Selling Price": Package,
  "FY Range Revenue": Target,
}

function formatCurrency0(value: number | null) {
  if (value === null || Number.isNaN(value)) return "N/A"
  return `$${Math.round(value).toLocaleString()}`
}

export function KpiCards() {
  const { filters, refreshKey } = useDashboardFilters()
  const [data, setData] = React.useState<ReturnType<typeof OverviewResponseSchema.parse> | null>(null)

  React.useEffect(() => {
    let cancelled = false
    async function run() {
      const raw = await postJson("/overview", filters)
      const parsed = OverviewResponseSchema.parse(raw)
      if (cancelled) return
      setData(parsed)
    }
    run().catch(() => {
      if (!cancelled) setData(null)
    })
    return () => {
      cancelled = true
    }
  }, [filters, refreshKey])

  const snapshotWeek = data?.snapshot.snapshot_week
  const kpis: KPI[] = [
    {
      label: "Weekly Revenue",
      value: data?.kpis.revenue ?? null,
      deltaPct: data?.kpis.revenue_wow ?? null,
      status: (data?.kpis.revenue_wow ?? 0) >= 0 ? "good" : "bad",
      helper: snapshotWeek ? `FW${snapshotWeek} vs prev week` : "vs prev week",
    },
    {
      label: "Units Sold",
      value: data?.kpis.units ?? null,
      deltaPct: data?.kpis.units_wow ?? null,
      status: (data?.kpis.units_wow ?? 0) >= 0 ? "good" : "bad",
      helper: snapshotWeek ? `FW${snapshotWeek} vs prev week` : "vs prev week",
    },
    { label: "Avg Selling Price", value: data?.kpis.asp ?? null, helper: "Revenue / Units" },
    { label: "FY Range Revenue", value: data?.range_totals.revenue ?? null, helper: "FY total (imported weeks)" },
    {
      label: "Gross Margin",
      value: data?.gm.gross_margin ?? null,
      helper: "Total GM (Selected)",
    },
    {
      label: "GM %",
      value: data?.gm.gross_margin_pct ?? null,
      helper: "Margin / Revenue",
      format: "percent",
    },
    {
      label: "Top GM Product",
      value: data?.gm.top_product?.gross_margin ?? null,
      helper: data?.gm.top_product?.part_number ?? "N/A",
      subtext: data?.gm.top_product?.description,
    },
    {
      label: "Top GM Category",
      value: data?.gm.top_category?.gross_margin ?? null,
      helper: data?.gm.top_category?.major_category ?? "N/A",
    },
  ]

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {kpis.map((kpi) => {
        const Icon = iconMap[kpi.label as keyof typeof iconMap] || DollarSign
        const delta = kpi.deltaPct
        const isPositive = (delta ?? 0) >= 0
        const showTrend = delta !== undefined && delta !== null

        return (
          <Card key={kpi.label}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{kpi.label}</CardTitle>
              <Icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {(kpi.label === "Top GM Product" || kpi.label === "Top GM Category") ? (
                  <span className="text-lg md:text-xl truncate block" title={kpi.helper || ""}>
                    {kpi.helper}
                  </span>
                ) : (
                  kpi.format === "percent"
                    ? kpi.value === null
                      ? "N/A"
                      : `${(Number(kpi.value) * 100).toFixed(1)}%`
                    : kpi.label.includes("Revenue") || kpi.label.includes("Margin")
                      ? formatCurrency0(kpi.value)
                      : kpi.label === "Avg Selling Price"
                        ? kpi.value === null
                          ? "N/A"
                          : `$${Number(kpi.value).toFixed(2)}`
                        : kpi.value === null
                          ? "N/A"
                          : kpi.value.toLocaleString()
                )}
              </div>
              <div className="flex flex-col gap-0.5 mt-1">
                {showTrend && (
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span
                      className={cn(
                        "flex items-center gap-1 font-medium",
                        kpi.status === "good"
                          ? "text-green-600 dark:text-green-400"
                          : kpi.status === "bad"
                            ? "text-red-600 dark:text-red-400"
                            : "text-yellow-600 dark:text-yellow-400",
                      )}
                    >
                      {isPositive ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                      {isPositive ? "+" : ""}
                      {Math.round((delta ?? 0) * 1000) / 10}%
                    </span>
                  </div>
                )}
                {(kpi.label === "Top GM Product" || kpi.label === "Top GM Category") ? (
                  <span className="text-sm font-medium text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30 px-2 py-0.5 rounded-full w-fit">
                    {formatCurrency0(kpi.value)}
                  </span>
                ) : (
                  kpi.helper && <span className="text-xs text-muted-foreground">{kpi.helper}</span>
                )}
                {kpi.subtext && <span className="text-[10px] text-muted-foreground truncate max-w-[180px]" title={kpi.subtext}>{kpi.subtext}</span>}
              </div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
