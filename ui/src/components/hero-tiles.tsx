"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowUpRight, ArrowDownRight, Star } from "lucide-react"
import { postJson } from "@/lib/api"
import { OverviewResponseSchema } from "@/lib/schema"
import { useDashboardFilters } from "@/contexts/dashboard-filters-context"

function pct(value: unknown) {
  const n = typeof value === "number" ? value : null
  if (n === null || Number.isNaN(n)) return "N/A"
  return `${Math.round(n * 1000) / 10}%`
}

export function HeroTiles() {
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

  const rising = data?.heroes.product.rising
  const declining = data?.heroes.product.declining
  const hero = data?.heroes.product.hero

  const risingSku = rising?.part_number ?? "N/A"
  const risingName = (rising as any)?.description as string | undefined
  const risingDelta = rising?.wow_rev_pct ?? null
  const decliningSku = declining?.part_number ?? "N/A"
  const decliningName = (declining as any)?.description as string | undefined
  const decliningDelta = declining?.yoy_rev_pct_comp ?? null
  const heroSku = hero?.part_number ?? "N/A"
  const heroName = (hero as any)?.description as string | undefined
  const heroDelta = hero?.yoy_rev_pct_comp ?? null

  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Card className="border-green-200 dark:border-green-900 bg-green-50/50 dark:bg-green-950/20">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-bold text-green-700 dark:text-green-400">Rising Star</CardTitle>
            <ArrowUpRight className="h-4 w-4 text-green-700 dark:text-green-400" />
          </div>
          <div className="text-4xl font-bold text-green-700 dark:text-green-400 leading-tight break-words mt-2">
            {risingSku}
          </div>
          <div className="text-sm text-green-600/80 font-mono mt-1 font-medium">{risingName || risingSku}</div>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-green-700 dark:text-green-400 mb-2">{pct(risingDelta)}</div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Revenue WoW% <span className="text-xs opacity-70 block mt-1">Product with the highest week-over-week revenue growth rate.</span>
          </p>
        </CardContent>
      </Card>

      <Card className="border-red-200 dark:border-red-900 bg-red-50/50 dark:bg-red-950/20">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-bold text-red-700 dark:text-red-400">Declining</CardTitle>
            <ArrowDownRight className="h-4 w-4 text-red-700 dark:text-red-400" />
          </div>
          <div className="text-4xl font-bold text-red-700 dark:text-red-400 leading-tight break-words mt-2">
            {decliningSku}
          </div>
          <div className="text-sm text-red-600/80 font-mono mt-1 font-medium">{decliningName || decliningSku}</div>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-red-700 dark:text-red-400 mb-2">{pct(decliningDelta)}</div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Revenue YoY% <span className="text-xs opacity-70 block mt-1">Product with the lowest comparable revenue in this week.</span>
          </p>
        </CardContent>
      </Card>

      <Card className="border-blue-200 dark:border-blue-900 bg-blue-50/50 dark:bg-blue-950/20">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-bold text-blue-700 dark:text-blue-400">Hero Product</CardTitle>
            <Star className="h-4 w-4 text-blue-700 dark:text-blue-400" />
          </div>
          <div className="text-4xl font-bold text-blue-700 dark:text-blue-400 leading-tight break-words mt-2">
            {heroSku}
          </div>
          <div className="text-sm text-blue-600/80 font-mono mt-1 font-medium">{heroName || heroSku}</div>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-blue-700 dark:text-blue-400 mb-2">{pct(heroDelta)}</div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Revenue YoY% <span className="text-xs opacity-70 block mt-1">Product with the highest revenue in this week.</span>
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
