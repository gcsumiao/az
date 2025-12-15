"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useDashboardFilters } from "@/contexts/dashboard-filters-context"
import { postJson } from "@/lib/api"
import { VegaChart } from "@/components/charts/VegaChart"

type Metric = "revenue" | "units"
type View = "product" | "category"

type TopRow = {
  rank: number
  part_number?: string
  description?: string
  major_category?: string
  revenue: number
  units: number
}

type PerformanceResponse = {
  metric: Metric
  view: View
  top: TopRow[]
  options?: string[]
  product_detail?: TopRow[]
  charts: Record<string, any>
}

export default function PerformancePage() {
  const { filters, refreshKey } = useDashboardFilters()
  const [metric, setMetric] = React.useState<Metric>("revenue")
  const [view, setView] = React.useState<View>("product")
  const [data, setData] = React.useState<PerformanceResponse | null>(null)

  React.useEffect(() => {
    let cancelled = false
    async function run() {
      const res = await postJson<PerformanceResponse>(`/performance?metric=${metric}&view=${view}`, filters)
      if (!cancelled) setData(res)
    }
    run().catch(() => {
      if (!cancelled) setData(null)
    })
    return () => {
      cancelled = true
    }
  }, [filters, refreshKey, metric, view])

  const top = data?.top ?? []
  const productDetail = data?.product_detail ?? []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Performance</h1>
          <p className="text-muted-foreground">Analyze product and category performance trends</p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={metric} onValueChange={(val) => setMetric(val as Metric)}>
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="Metric" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="revenue">Revenue</SelectItem>
              <SelectItem value="units">Units</SelectItem>
            </SelectContent>
          </Select>
          <Select value={view} onValueChange={(val) => setView(val as View)}>
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="View" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="product">Product</SelectItem>
              <SelectItem value="category">Category</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex flex-col gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Trend</CardTitle>
            <p className="text-sm text-muted-foreground">Weekly {metric} trend</p>
          </CardHeader>
          <CardContent>
            <VegaChart key={`trend-${metric}-${view}`} spec={data?.charts?.trend} className="w-full" />
          </CardContent>
        </Card>

        {view === "product" && data?.charts?.breakdown && (
          <Card>
            <CardHeader>
              <CardTitle>Category Breakdown</CardTitle>
            </CardHeader>
            <CardContent>
              <VegaChart key={`breakdown-${metric}-${view}`} spec={data.charts.breakdown} className="w-full" />
            </CardContent>
          </Card>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Top {view === "product" ? "Products" : "Categories"}</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-12">#</TableHead>
                  {view === "product" && <TableHead>SKU</TableHead>}
                  {view === "product" && <TableHead>Category</TableHead>}
                  <TableHead>{view === "product" ? "Description" : "Category"}</TableHead>
                  <TableHead className="text-right">{metric === "revenue" ? "Revenue" : "Units"}</TableHead>
                  <TableHead className="text-right">{metric === "revenue" ? "Units" : "Revenue"}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {top.map((row) => (
                  <TableRow key={`${view}-${row.rank}-${row.part_number ?? row.major_category}`}>
                    <TableCell className="font-medium">{row.rank}</TableCell>
                    {view === "product" && <TableCell className="font-mono text-xs">{row.part_number}</TableCell>}
                    {view === "product" && <TableCell className="text-xs">{row.major_category ?? "Unmapped"}</TableCell>}
                    <TableCell className="truncate">
                      {view === "product" ? row.description ?? row.part_number : row.major_category}
                    </TableCell>
                    <TableCell className="text-right">
                      {metric === "revenue" ? `$${Math.round(row.revenue).toLocaleString()}` : row.units.toLocaleString()}
                    </TableCell>
                    <TableCell className="text-right">
                      {metric === "revenue" ? row.units.toLocaleString() : `$${Math.round(row.revenue).toLocaleString()}`}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Highlights</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {top.slice(0, 3).map((row) => (
              <div key={`hi-${row.rank}`} className="flex items-center justify-between rounded-lg border p-3">
                <div>
                  <p className="text-sm font-medium">
                    {view === "product" ? row.part_number : row.major_category}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {view === "product" ? row.description : "Category leader"}
                  </p>
                </div>
                <Badge variant="secondary">
                  {metric === "revenue" ? `$${Math.round(row.revenue).toLocaleString()}` : row.units.toLocaleString()}
                </Badge>
              </div>
            ))}
            {top.length === 0 && <p className="text-sm text-muted-foreground">No data for current filters.</p>}
          </CardContent>
        </Card>
      </div>

      {view === "category" && productDetail.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Top Products (detail)</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>SKU</TableHead>
                  <TableHead>Description</TableHead>
                  <TableHead className="text-right">Revenue</TableHead>
                  <TableHead className="text-right">Units</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {productDetail.map((row) => (
                  <TableRow key={`detail-${row.part_number}`}>
                    <TableCell className="font-mono text-xs">{row.part_number}</TableCell>
                    <TableCell>{row.description}</TableCell>
                    <TableCell className="text-right">$ {Math.round(row.revenue).toLocaleString()}</TableCell>
                    <TableCell className="text-right">{row.units.toLocaleString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
