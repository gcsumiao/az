"use client"

import { useState } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  LayoutDashboard,
  TrendingUp,
  Package,
  AlertCircle,
  Settings,
  HelpCircle,
  Menu,
  ChevronLeft,
  ChevronDown,
  ChevronRight,
  BarChart,
  RotateCcw,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip"
import { Badge } from "@/components/ui/badge"
import { GlobalFilters } from "@/components/filters/GlobalFilters"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@radix-ui/react-collapsible"

const navigation = [
  { name: "Overview", href: "/", icon: LayoutDashboard },
  { name: "Performance", href: "/performance", icon: TrendingUp },
  { name: "Supply Health", href: "/supply-health", icon: Package },
  { name: "Returns Analysis", href: "/supply-health/returns", icon: RotateCcw },
  { name: "Action Center", href: "/action-center", icon: AlertCircle, badge: 8 },
  { name: "Forecast & Orders", href: "/action-center/forecast", icon: BarChart },
]

const bottomNavigation = [
  { name: "Settings", href: "/settings", icon: Settings },
  { name: "Help", href: "/help", icon: HelpCircle },
]

export function Sidebar() {
  const pathname = usePathname()
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [isMobileOpen, setIsMobileOpen] = useState(false)
  // Track open states for collapsible items. Default open for simplicity or based on active path.
  const [openItems, setOpenItems] = useState<Record<string, boolean>>({ "Supply Health": true, "Action Center": true })

  const toggleOpen = (name: string) => {
    setOpenItems((prev) => ({ ...prev, [name]: !prev[name] }))
  }

  const NavItem = ({ item, isBottom = false }: { item: (typeof navigation)[0]; isBottom?: boolean }) => {
    const isActive = pathname === item.href
    // const hasChildren = "children" in item && item.children && item.children.length > 0
    // Simplified: Flat list for now per request, or we keep structure but render differently?
    // Request: "return analysis and forecast pages should be on the same level as other pages"
    // So no children needed in the array.

    const LinkContent = (
      <Link
        href={item.href}
        className={cn(
          "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all hover:bg-accent",
          isActive ? "bg-accent text-accent-foreground shadow-sm" : "text-muted-foreground hover:text-foreground",
          isCollapsed && "justify-center px-2",
        )}
      >
        <item.icon className={cn("h-5 w-5 flex-shrink-0")} />
        {!isCollapsed && (
          <>
            <span className="flex-1">{item.name}</span>
            {"badge" in item && item.badge && (
              <Badge variant="destructive" className="h-5 min-w-5 px-1 text-xs">
                {item.badge}
              </Badge>
            )}
          </>
        )}
      </Link>
    )

    if (isCollapsed) {
      return (
        <Tooltip delayDuration={0}>
          <TooltipTrigger asChild>{LinkContent}</TooltipTrigger>
          <TooltipContent side="right" className="flex items-center gap-2">
            {item.name}
            {"badge" in item && item.badge && (
              <Badge variant="destructive" className="h-5 min-w-5 px-1 text-xs">
                {item.badge}
              </Badge>
            )}
          </TooltipContent>
        </Tooltip>
      )
    }

    return LinkContent
  }

  return (
    <TooltipProvider>
      <>
        <button
          className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-background rounded-md shadow-md border"
          onClick={() => setIsMobileOpen(!isMobileOpen)}
          aria-label="Toggle sidebar"
        >
          <Menu className="h-5 w-5" />
        </button>

        <div
          className={cn(
            "fixed inset-y-0 left-0 z-40 flex flex-col border-r bg-background transition-all duration-300 ease-in-out lg:static",
            isCollapsed ? "w-[72px]" : "w-64",
            isMobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0",
          )}
        >
          <div className="flex h-16 items-center gap-2 border-b px-4">
            {!isCollapsed && (
              <Link href="/" className="flex items-center font-semibold text-lg">
                Navigation
              </Link>
            )}
            <Button
              variant="ghost"
              size="icon"
              className={cn("h-8 w-8 ml-auto", isCollapsed && "ml-0")}
              onClick={() => setIsCollapsed(!isCollapsed)}
            >
              <ChevronLeft className={cn("h-4 w-4 transition-transform", isCollapsed && "rotate-180")} />
              <span className="sr-only">{isCollapsed ? "Expand" : "Collapse"} Sidebar</span>
            </Button>
          </div>

          <div className="flex-1 overflow-auto py-4">
            <nav className="space-y-1 px-3">
              {navigation.map((item) => (
                <NavItem key={item.name} item={item} />
              ))}
            </nav>

            {!isCollapsed && (
              <div className="mt-6 px-3">
                <div className="px-3 mb-2 flex items-center gap-2">
                  <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Filters</h3>
                </div>
                <GlobalFilters />
              </div>
            )}
          </div>

          <div className="border-t p-3">
            <nav className="space-y-1">
              {bottomNavigation.map((item) => (
                <NavItem key={item.name} item={item} isBottom />
              ))}
            </nav>
          </div>
        </div>

        {isMobileOpen && (
          <div
            className="fixed inset-0 z-30 bg-background/80 backdrop-blur-sm lg:hidden"
            onClick={() => setIsMobileOpen(false)}
          />
        )}
      </>
    </TooltipProvider>
  )
}
