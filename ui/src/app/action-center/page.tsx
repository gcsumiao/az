"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useDashboardFilters } from "@/contexts/dashboard-filters-context"
import { postJson } from "@/lib/api"
import { VegaChart } from "@/components/charts/VegaChart"

type ActionCenterResponse = {
  kpis: { skus_needing_action: number; high_severity: number; revenue_at_risk: number }
  table: { part_number: string; store_oos_exposure?: number; severity?: string; source?: string; revenue_impact?: number }[]
  sku_options: string[]
  drilldown: { sku: string | null; chart: any }
}

export default function ActionCenterPage() {
  const { filters, refreshKey } = useDashboardFilters()
  const [severity, setSeverity] = React.useState("All")
  const [q, setQ] = React.useState("")
  const [selectedSku, setSelectedSku] = React.useState<string | null>(null)
  const [data, setData] = React.useState<ActionCenterResponse | null>(null)

  React.useEffect(() => {
    let cancelled = false
    async function run() {
      const search = new URLSearchParams({
        severity,
        q,
        selected_sku: selectedSku ?? "",
      })
      const res = await postJson<ActionCenterResponse>(`/action-center?${search.toString()}`, filters)
      if (!cancelled) {
        setData(res)
        if (!selectedSku && res.sku_options?.length) {
          setSelectedSku(res.sku_options[0])
        }
      }
    }
    run().catch(() => {
      if (!cancelled) setData(null)
    })
    return () => {
      cancelled = true
    }
  }, [filters, refreshKey, severity, q, selectedSku])

  const kpis = data?.kpis ?? { skus_needing_action: 0, high_severity: 0, revenue_at_risk: 0 }
  const table = data?.table ?? []
  const skuOptions = data?.sku_options ?? []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Action Center</h1>
          <p className="text-muted-foreground">Prioritize SKUs with service risk and revenue at stake</p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={severity} onValueChange={setSeverity}>
            <SelectTrigger className="w-[160px]">
              <SelectValue placeholder="Severity" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="All">All</SelectItem>
              <SelectItem value="High">High</SelectItem>
              <SelectItem value="Medium">Medium</SelectItem>
            </SelectContent>
          </Select>
          <Input
            placeholder="Search SKU…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            className="w-[200px]"
          />
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <KpiCard title="SKUs needing action" value={kpis.skus_needing_action} />
        <KpiCard title="High severity" value={kpis.high_severity} />
        <KpiCard title="Revenue at risk" value={kpis.revenue_at_risk} prefix="$" />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Actionable SKUs</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>SKU</TableHead>
                  <TableHead>Source</TableHead>
                  <TableHead className="text-right">OOS Exposure</TableHead>
                  <TableHead className="text-right">Revenue Impact</TableHead>
                  <TableHead>Severity</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {table.map((row) => (
                  <TableRow
                    key={`${row.part_number}-${row.source}`}
                    className={row.part_number === selectedSku ? "bg-accent/40" : ""}
                    onClick={() => setSelectedSku(row.part_number)}
                  >
                    <TableCell className="font-mono text-xs">{row.part_number}</TableCell>
                    <TableCell>{row.source}</TableCell>
                    <TableCell className="text-right">
                      {Math.round(row.store_oos_exposure ?? 0).toLocaleString()}
                    </TableCell>
                    <TableCell className="text-right">
                      ${Math.round(row.revenue_impact ?? 0).toLocaleString()}
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant={(row.severity ?? "").toLowerCase() === "high" ? "destructive" : "secondary"}
                        className="capitalize"
                      >
                        {row.severity ?? "NA"}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            {table.length === 0 && <p className="text-sm text-muted-foreground mt-2">No actions for current filters.</p>}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>SKU Drilldown</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Select
              value={selectedSku ?? undefined}
              onValueChange={(val) => setSelectedSku(val)}
              disabled={!skuOptions.length}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select SKU" />
              </SelectTrigger>
              <SelectContent>
                {skuOptions.map((sku) => (
                  <SelectItem key={sku} value={sku}>
                    {sku}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <VegaChart spec={data?.drilldown?.chart} className="w-full" />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function KpiCard({ title, value, prefix }: { title: string; value: number | null | undefined; prefix?: string }) {
  const display = value === null || value === undefined ? "N/A" : `${prefix ?? ""}${Math.round(value).toLocaleString()}`
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{display}</div>
      </CardContent>
    </Card>
  )
}
