"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { ThemeToggle } from "./theme-toggle"
import { Notifications } from "./notifications"
import { useSettings } from "@/contexts/settings-context"
import { RefreshCw, Download, HelpCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import React from "react"
import { useDashboardFilters } from "@/contexts/dashboard-filters-context"

export function TopNav() {
  const pathname = usePathname()
  const pathSegments = pathname.split("/").filter(Boolean)
  const { settings } = useSettings()
  const { filters, refresh } = useDashboardFilters()

  const weeksLabel = React.useMemo(() => {
    if (!filters.selected_weeks.length) return "Weeks: All"
    const sorted = [...filters.selected_weeks].sort((a, b) => a - b)
    return sorted.length === 1 ? `FW${sorted[0]}` : `FW${sorted[0]}–FW${sorted[sorted.length - 1]}`
  }, [filters.selected_weeks])

  const categoriesLabel = React.useMemo(() => {
    if (!filters.selected_categories.length || filters.selected_categories.includes("All Categories")) return "Categories: All"
    if (filters.selected_categories.length === 1) return `Category: ${filters.selected_categories[0]}`
    return `Categories: ${filters.selected_categories.length} selected`
  }, [filters.selected_categories])

  const partsLabel = React.useMemo(() => {
    if (filters.sku_query) return `SKU search: "${filters.sku_query}"`
    if (filters.selected_parts.length) return `${filters.selected_parts.length} SKU focus`
    return "SKU: All"
  }, [filters.sku_query, filters.selected_parts])

  return (
    <header className="sticky top-0 z-40 border-b bg-background">
      <div className="flex h-14 items-center justify-between px-6">
        {/* Breadcrumb */}
        <nav className="flex items-center space-x-2 text-sm">
          <Link href="/" className="font-medium text-muted-foreground hover:text-foreground transition-colors">
            Home
          </Link>
          {pathSegments.map((segment, index) => (
            <React.Fragment key={segment}>
              <span className="text-muted-foreground">/</span>
              <Link
                href={`/${pathSegments.slice(0, index + 1).join("/")}`}
                className="font-medium hover:text-foreground transition-colors"
              >
                {segment
                  .split("-")
                  .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
                  .join(" ")}
              </Link>
            </React.Fragment>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" className="gap-2" onClick={refresh}>
            <RefreshCw className="h-4 w-4" />
            <span className="hidden sm:inline">Refresh</span>
          </Button>
          <Button variant="ghost" size="sm" className="gap-2">
            <Download className="h-4 w-4" />
            <span className="hidden sm:inline">Export</span>
          </Button>
          <Button variant="ghost" size="sm">
            <HelpCircle className="h-4 w-4" />
            <span className="sr-only">Help</span>
          </Button>
          <div className="h-4 w-px bg-border mx-1" />
          <Notifications />
          <ThemeToggle />
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="relative h-8 w-8 rounded-full">
                <Avatar className="h-8 w-8">
                  <AvatarFallback className="bg-primary/10 text-primary">
                    GC
                  </AvatarFallback>
                </Avatar>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56" align="end" forceMount>
              <DropdownMenuLabel className="font-normal">
                <div className="flex flex-col space-y-1">
                  <p className="text-sm font-medium leading-none">ginnyc@innova.com</p>
                  <p className="text-xs leading-none text-muted-foreground">ginnyc@innova.com</p>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem asChild>
                <Link href="/settings">Profile</Link>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link href="/settings">Settings</Link>
              </DropdownMenuItem>
              <DropdownMenuItem>Log out</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      <div className="flex items-center gap-2 px-6 py-2 border-t bg-muted/30">
        <span className="text-xs font-medium text-muted-foreground">Filters:</span>
        <div className="flex items-center gap-2 flex-wrap">
          <Badge variant="secondary" className="gap-1.5 pr-1 pl-2.5">
            {weeksLabel}
          </Badge>
          <Badge variant="secondary" className="gap-1.5 pr-1 pl-2.5">
            {categoriesLabel}
          </Badge>
          <Badge variant="secondary" className="gap-1.5 pr-1 pl-2.5">
            {partsLabel}
          </Badge>
        </div>
      </div>
    </header>
  )
}
