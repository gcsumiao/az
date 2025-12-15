"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { AlertTriangle } from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { postJson } from "@/lib/api"
import { OverviewResponseSchema } from "@/lib/schema"
import { useDashboardFilters } from "@/contexts/dashboard-filters-context"

type AlertItem = {
  alert_type: string
  severity: string
  message: string
  action: string
}

const severityLabel = { high: "High", medium: "Medium", low: "Low" }

export function AlertsPanel() {
  const { filters, refreshKey } = useDashboardFilters()
  const [data, setData] = React.useState<ReturnType<typeof OverviewResponseSchema.parse> | null>(null)
  const [filterSeverity, setFilterSeverity] = React.useState("all")
  const [searchTerm, setSearchTerm] = React.useState("")
  const [showAllDialog, setShowAllDialog] = React.useState(false)

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

  const alerts = data?.alerts ?? []
  const filteredAlerts = alerts.filter((a) => {
    const matchesSeverity = filterSeverity === "all" || a.severity === filterSeverity
    const q = searchTerm.toLowerCase()
    const matchesSearch =
      !q ||
      a.alert_type.toLowerCase().includes(q) ||
      a.message.toLowerCase().includes(q) ||
      a.action.toLowerCase().includes(q)
    return matchesSeverity && matchesSearch
  })

  const severityOrder: Record<string, number> = { high: 0, medium: 1, low: 2 }
  const sorted = [...filteredAlerts].sort(
    (a, b) => (severityOrder[a.severity] ?? 3) - (severityOrder[b.severity] ?? 3),
  )
  const topAlerts = sorted.slice(0, 5)

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <div>
          <CardTitle>Alerts & Notifications</CardTitle>
          <p className="text-sm text-muted-foreground mt-1">Top 5 items requiring attention</p>
        </div>
        <Dialog open={showAllDialog} onOpenChange={setShowAllDialog}>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm">
              View All ({alerts.length})
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden flex flex-col">
            <DialogHeader>
              <DialogTitle>All Alerts</DialogTitle>
              <DialogDescription>Filter and search through all alerts</DialogDescription>
            </DialogHeader>

            <div className="flex gap-2 pt-4">
              <Input
                placeholder="Search alerts..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="flex-1"
              />
              <Select value={filterSeverity} onValueChange={setFilterSeverity}>
                <SelectTrigger className="w-[140px]">
                  <SelectValue placeholder="Severity" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="low">Low</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="overflow-y-auto flex-1 pr-2">
              <div className="space-y-3 py-4">
                {sorted.map((alert, idx) => (
                  <AlertRow key={`${alert.alert_type}-${idx}`} alert={alert} />
                ))}
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </CardHeader>

      <CardContent className="grid gap-3 md:grid-cols-2">
        {topAlerts.length === 0 && <p className="text-sm text-muted-foreground">No alerts for current filters.</p>}
        {topAlerts.map((alert, idx) => (
          <AlertRow key={`${alert.alert_type}-${idx}`} alert={alert} />
        ))}
      </CardContent>
    </Card>
  )
}

function AlertRow({ alert }: { alert: AlertItem }) {
  const badgeVariant = alert.severity === "high" ? "destructive" : alert.severity === "medium" ? "default" : "secondary"
  return (
    <div className="flex items-start gap-3 p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors">
      <div
        className={`p-2 rounded-md flex-shrink-0 ${alert.severity === "high" ? "bg-red-100 dark:bg-red-950" : alert.severity === "medium" ? "bg-yellow-100 dark:bg-yellow-950" : "bg-blue-100 dark:bg-blue-950"}`}
      >
        <AlertTriangle
          className={`h-4 w-4 ${alert.severity === "high" ? "text-red-600 dark:text-red-400" : alert.severity === "medium" ? "text-yellow-600 dark:text-yellow-400" : "text-blue-600 dark:text-blue-400"}`}
        />
      </div>
      <div className="flex-1 space-y-1 min-w-0">
        <div className="flex items-center gap-2">
          <p className="text-sm font-medium leading-none">{alert.alert_type}</p>
          <Badge variant={badgeVariant} className="text-xs capitalize">
            {severityLabel[alert.severity as keyof typeof severityLabel] ?? alert.severity}
          </Badge>
        </div>
        <p className="text-sm text-muted-foreground">{alert.message}</p>
        <p className="text-xs text-muted-foreground">Action: {alert.action}</p>
      </div>
    </div>
  )
}
