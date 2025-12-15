"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { postJson } from "@/lib/api"
import { OverviewResponseSchema } from "@/lib/schema"
import { useDashboardFilters } from "@/contexts/dashboard-filters-context"
import { VegaChart } from "@/components/charts/VegaChart"

export function TrendCharts() {
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

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Revenue Trend</CardTitle>
          <p className="text-sm text-muted-foreground">Weekly revenue trend (imported weeks)</p>
        </CardHeader>
        <CardContent>
          <VegaChart spec={(data?.charts as any)?.revenue_trend} className="w-full" />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Units Trend</CardTitle>
          <p className="text-sm text-muted-foreground">Weekly units trend (imported weeks)</p>
        </CardHeader>
        <CardContent>
          <VegaChart spec={(data?.charts as any)?.units_trend} className="w-full" />
        </CardContent>
      </Card>
    </div>
  )
}
