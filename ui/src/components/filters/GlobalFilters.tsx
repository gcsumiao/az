"use client"

import * as React from "react"
import { Calendar, Filter } from "lucide-react"

import { useDashboardFilters } from "@/contexts/dashboard-filters-context"
import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Checkbox } from "@/components/ui/checkbox"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Input } from "@/components/ui/input"

function formatWeeksLabel(weeks: number[]) {
  if (!weeks.length) return "Weeks: All"
  const sorted = [...weeks].sort((a, b) => a - b)
  if (sorted.length === 1) return `Weeks: FW${sorted[0]}`
  return `Weeks: FW${sorted[0]}–FW${sorted[sorted.length - 1]}`
}

export function GlobalFilters() {
  const { filters, setFilters, meta } = useDashboardFilters()
  const weeksLabel = formatWeeksLabel(filters.selected_weeks)
  const categoriesLabel =
    !filters.selected_categories.length || filters.selected_categories.includes("All Categories")
      ? "Category: All"
      : `Category: ${filters.selected_categories.length}`

  return (
    <div className="space-y-2">
      <Popover>
        <PopoverTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start gap-3 h-9 px-3 text-sm font-medium text-muted-foreground hover:text-foreground"
          >
            <Calendar className="h-4 w-4" />
            {weeksLabel}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-72 p-3" align="start">
          <div className="text-sm font-medium mb-2">Fiscal Weeks</div>
          <ScrollArea className="h-48 pr-2">
            <div className="space-y-2">
              {meta.weeks.map((w) => {
                const checked = filters.selected_weeks.includes(w)
                return (
                  <label key={w} className="flex items-center gap-2 text-sm">
                    <Checkbox
                      checked={checked}
                      onCheckedChange={(val) => {
                        const next = new Set(filters.selected_weeks)
                        if (val) next.add(w)
                        else next.delete(w)
                        setFilters({ ...filters, selected_weeks: Array.from(next).sort((a, b) => a - b) })
                      }}
                    />
                    FW{w}
                  </label>
                )
              })}
            </div>
          </ScrollArea>
        </PopoverContent>
      </Popover>

      <Popover>
        <PopoverTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start gap-3 h-9 px-3 text-sm font-medium text-muted-foreground hover:text-foreground"
          >
            <Filter className="h-4 w-4" />
            {categoriesLabel}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-72 p-3" align="start">
          <div className="text-sm font-medium mb-2">Major Category</div>
          <ScrollArea className="h-56 pr-2">
            <div className="space-y-2">
              <label className="flex items-center gap-2 text-sm">
                <Checkbox
                  checked={filters.selected_categories.includes("All Categories")}
                  onCheckedChange={(val) =>
                    setFilters({ ...filters, selected_categories: val ? ["All Categories"] : [] })
                  }
                />
                All Categories
              </label>
              {meta.categories.map((c) => {
                const checked = filters.selected_categories.includes(c)
                return (
                  <label key={c} className="flex items-center gap-2 text-sm">
                    <Checkbox
                      checked={checked}
                      disabled={filters.selected_categories.includes("All Categories")}
                      onCheckedChange={(val) => {
                        const next = new Set(filters.selected_categories.filter((x) => x !== "All Categories"))
                        if (val) next.add(c)
                        else next.delete(c)
                        setFilters({ ...filters, selected_categories: Array.from(next) })
                      }}
                    />
                    {c}
                  </label>
                )
              })}
            </div>
          </ScrollArea>
        </PopoverContent>
      </Popover>

      <Input
        value={filters.sku_query}
        onChange={(e) => setFilters({ ...filters, sku_query: e.target.value })}
        placeholder="SKU search…"
        className="h-9"
      />
    </div>
  )
}

