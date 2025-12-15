"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useDashboardFilters } from "@/contexts/dashboard-filters-context"
import { postJson } from "@/lib/api"
import { VegaChart } from "@/components/charts/VegaChart"

type ReturnsResponse = {
    kpis: {
        avg_damaged_rate?: number | null
        avg_undamaged_rate?: number | null
        top_risk?: { part_number: string; damaged_rate: number } | null
    }
    top: Array<{
        rank: number
        part_number: string
        description?: string
        damaged_rate: number
        gross_units: number
    }>
    charts: Record<string, any>
}

export default function ReturnsPage() {
    const { filters, refreshKey } = useDashboardFilters()
    const [data, setData] = React.useState<ReturnsResponse | null>(null)

    React.useEffect(() => {
        let cancelled = false
        async function run() {
            const res = await postJson<ReturnsResponse>("/returns", filters)
            if (!cancelled) setData(res)
        }
        run().catch(() => {
            if (!cancelled) setData(null)
        })
        return () => {
            cancelled = true
        }
    }, [filters, refreshKey])

    const kpis = [
        { label: "Avg Damaged Rate", value: data?.kpis?.avg_damaged_rate, format: "pct" },
        { label: "Avg Undamaged Rate", value: data?.kpis?.avg_undamaged_rate, format: "pct" },
        { label: "Top Risk SKU", value: data?.kpis?.top_risk?.part_number, format: "text", sub: data?.kpis?.top_risk?.damaged_rate ? `${(data.kpis.top_risk.damaged_rate * 100).toFixed(1)}%` : null },
    ]

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-3xl font-bold tracking-tight">Return Analysis</h1>
                <p className="text-muted-foreground">Monitor return rates and identify risk SKUs</p>
            </div>

            <div className="grid gap-4 md:grid-cols-3">
                {kpis.map((k) => (
                    <Card key={k.label}>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">{k.label}</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">
                                {k.format === "pct" && k.value != null
                                    ? `${(Number(k.value) * 100).toFixed(2)}%`
                                    : k.value ?? "N/A"}
                            </div>
                            {k.sub && <p className="text-xs text-muted-foreground mt-1">Rate: {k.sub}</p>}
                        </CardContent>
                    </Card>
                ))}
            </div>

            <Card>
                <CardHeader>
                    <CardTitle>Return Rate Trend</CardTitle>
                    <p className="text-sm text-muted-foreground">Damaged vs Undamaged Rates over time</p>
                </CardHeader>
                <CardContent>
                    <VegaChart spec={data?.charts?.return_rate_trend} className="w-full" />
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Top Risk SKUs</CardTitle>
                    <p className="text-sm text-muted-foreground">Highest damaged return rates (Volume Gated)</p>
                </CardHeader>
                <CardContent>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead className="w-12">Rank</TableHead>
                                <TableHead>SKU</TableHead>
                                <TableHead className="text-right">Damaged Rate</TableHead>
                                <TableHead className="text-right">Units</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {(data?.top ?? []).map((row) => (
                                <TableRow key={row.part_number}>
                                    <TableCell className="font-medium">{row.rank}</TableCell>
                                    <TableCell className="font-mono">{row.part_number}</TableCell>
                                    <TableCell className="text-right">{(row.damaged_rate * 100).toFixed(2)}%</TableCell>
                                    <TableCell className="text-right">{row.gross_units.toLocaleString()}</TableCell>
                                </TableRow>
                            ))}
                            {(!data?.top || data.top.length === 0) && (
                                <TableRow>
                                    <TableCell colSpan={4} className="h-24 text-center">No data available.</TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>
        </div>
    )
}
