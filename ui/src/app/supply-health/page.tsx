"use client"

import * as React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useDashboardFilters } from "@/contexts/dashboard-filters-context"
import { postJson } from "@/lib/api"
import { VegaChart } from "@/components/charts/VegaChart"

type SupplyResponse = {
  service_kpis: {
    label?: string
    shipped_units?: number | null
    fill_rate?: number | null
    not_shipped_units?: number | null
  }
  charts: Record<string, any>
  exceptions: { redflags: Record<string, any>[]; outs: Record<string, any>[] }
}

export default function SupplyHealthPage() {
  const { filters, refreshKey } = useDashboardFilters()
  const [data, setData] = React.useState<SupplyResponse | null>(null)

  React.useEffect(() => {
    let cancelled = false
    async function run() {
      const res = await postJson<SupplyResponse>("/supply-health", filters)
      if (!cancelled) setData(res)
    }
    run().catch(() => {
      if (!cancelled) setData(null)
    })
    return () => {
      cancelled = true
    }
  }, [filters, refreshKey])

  const kpis = data?.service_kpis ?? {}
  const redflags = data?.exceptions?.redflags ?? []
  const outs = data?.exceptions?.outs ?? []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Supply Health</h1>
        <p className="text-muted-foreground">Monitor CPFR, fill rate, and OOS exposure</p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <KpiCard title="Shipped Units" value={kpis.shipped_units} helper={kpis.label} />
        <KpiCard
          title="Fill Rate"
          value={kpis.fill_rate !== undefined && kpis.fill_rate !== null ? `${(kpis.fill_rate * 100).toFixed(1)}%` : "N/A"}
          helper="Current snapshot"
        />
        <KpiCard title="Not Shipped Units" value={kpis.not_shipped_units} helper="Last snapshot" />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Shipped Units Trend</CardTitle>
            <CardDescription>Weekly shipped units (CPFR)</CardDescription>
          </CardHeader>
          <CardContent>
            <VegaChart spec={data?.charts?.shipped_trend} className="w-full" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Fill Rate Trend</CardTitle>
            <CardDescription>Weekly fill rate vs target</CardDescription>
          </CardHeader>
          <CardContent>
            <VegaChart spec={data?.charts?.fill_rate_trend} className="w-full" />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Red Flags (fill rate)</CardTitle>
            <CardDescription>Parts with not shipped exposure</CardDescription>
          </CardHeader>
          <CardContent>
            {redflags.length === 0 ? (
              <p className="text-sm text-muted-foreground">No red flags for current filters.</p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>SKU</TableHead>
                    <TableHead className="text-right">Not Shipped LFW</TableHead>
                    <TableHead className="text-right">Not Shipped L4W</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {redflags.slice(0, 20).map((row, idx) => (
                    <TableRow key={`rf-${row.part_number ?? idx}`}>
                      <TableCell className="font-mono text-xs">{row.part_number}</TableCell>
                      <TableCell className="text-right">{row.not_shipped_lfw ?? 0}</TableCell>
                      <TableCell className="text-right">{row.not_shipped_l4w ?? 0}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>OOS Risk (Outs)</CardTitle>
            <CardDescription>Top outs by store exposure</CardDescription>
          </CardHeader>
          <CardContent>
            {outs.length === 0 ? (
              <p className="text-sm text-muted-foreground">No outs for current filters.</p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>SKU</TableHead>
                    <TableHead className="text-right">Store OOS Exposure</TableHead>
                    <TableHead>Severity</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {outs.slice(0, 20).map((row, idx) => (
                    <TableRow key={`out-${row.part_number ?? idx}`}>
                      <TableCell className="font-mono text-xs">{row.part_number}</TableCell>
                      <TableCell className="text-right">{Math.round(row.store_oos_exposure ?? 0).toLocaleString()}</TableCell>
                      <TableCell>
                        <Badge variant={(row.severity ?? "").toLowerCase() === "high" ? "destructive" : "secondary"}>
                          {row.severity ?? "NA"}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function KpiCard({ title, value, helper }: { title: string; value: number | string | null | undefined; helper?: string }) {
  const display =
    value === null || value === undefined
      ? "N/A"
      : typeof value === "string"
        ? value
        : Math.round(Number(value)).toLocaleString()
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{display}</div>
        {helper && <p className="text-xs text-muted-foreground">{helper}</p>}
      </CardContent>
    </Card>
  )
}
