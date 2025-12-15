import { KpiCards } from "@/components/kpi-cards"
import { HeroTiles } from "@/components/hero-tiles"
import { TrendCharts } from "@/components/trend-charts"
import { AlertsPanel } from "@/components/alerts-panel"

export default function Dashboard() {
  return (
    <div className="p-6 space-y-6">
      {/* Row 1: KPI Cards */}
      <KpiCards />

      {/* Row 2: Hero Tiles */}
      <HeroTiles />

      {/* Row 3: Over-time Trends */}
      <TrendCharts />

      {/* Row 4: Alerts */}
      <AlertsPanel />
    </div>
  )
}
