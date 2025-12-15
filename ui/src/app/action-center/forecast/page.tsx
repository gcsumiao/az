"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useDashboardFilters } from "@/contexts/dashboard-filters-context"
import { postJson } from "@/lib/api"
import { VegaChart } from "@/components/charts/VegaChart"

type CoverageResponse = {
    kpis: {
        avg_coverage?: number | null
        latest_coverage?: number | null
        mape?: number | null
    }
    charts: Record<string, any>
    table: any[]
    dq_warning?: string | null
}

export default function ForecastPage() {
    const { filters, refreshKey } = useDashboardFilters()
    const [data, setData] = React.useState<CoverageResponse | null>(null)

    React.useEffect(() => {
        let cancelled = false
        async function run() {
            const res = await postJson<CoverageResponse>("/coverage", filters)
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
        { label: "Avg Coverage", value: data?.kpis?.avg_coverage, format: "pct" },
        { label: "Latest Coverage", value: data?.kpis?.latest_coverage, format: "pct" },
        { label: "Forecast Error (MAPE)", value: data?.kpis?.mape, format: "pct" },
    ]

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-3xl font-bold tracking-tight">Forecast & Orders</h1>
                <p className="text-muted-foreground">Analyze forecast coverage and accuracy</p>
            </div>

            {data?.dq_warning && (
                <div className="rounded-md bg-red-50 p-4 border border-red-200">
                    <div className="flex">
                        <div className="flex-shrink-0">
                            {/* Warning Icon */}
                            <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                                <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                            </svg>
                        </div>
                        <div className="ml-3">
                            <h3 className="text-sm font-medium text-red-800">Data Quality Warning</h3>
                            <div className="mt-2 text-sm text-red-700">
                                <p>{data.dq_warning}</p>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            <div className="grid gap-4 md:grid-cols-3">
                {kpis.map((k) => (
                    <Card key={k.label}>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">{k.label}</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">
                                {k.value != null ? (k.format === "pct" ? `${(Number(k.value) * 100).toFixed(1)}%` : k.value) : "N/A"}
                            </div>
                        </CardContent>
                    </Card>
                ))}
            </div>

            <div className="grid gap-4 md:grid-cols-2">
                <Card>
                    <CardHeader>
                        <CardTitle>Coverage Trend</CardTitle>
                        <p className="text-sm text-muted-foreground">Order Units / Forecast Units</p>
                    </CardHeader>
                    <CardContent>
                        <VegaChart spec={data?.charts?.coverage} className="w-full" />
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader>
                        <CardTitle>Forecast vs Orders</CardTitle>
                        <p className="text-sm text-muted-foreground">Volume comparison</p>
                    </CardHeader>
                    <CardContent>
                        <VegaChart spec={data?.charts?.forecast_vs_orders} className="w-full" />
                    </CardContent>
                </Card>
            </div>

            <Card>
                <CardHeader>
                    <CardTitle>Forecast Error %</CardTitle>
                    <p className="text-sm text-muted-foreground">Absolute Percentage Error (MAPE) trend</p>
                </CardHeader>
                <CardContent>
                    <VegaChart spec={data?.charts?.forecast_error} className="w-full" />
                </CardContent>
            </Card>
        </div>
    )
}
