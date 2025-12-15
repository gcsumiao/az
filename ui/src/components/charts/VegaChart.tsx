"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

type VegaSpec = Record<string, unknown>

export function VegaChart({ spec, className }: { spec: VegaSpec | null | undefined; className?: string }) {
  const ref = React.useRef<HTMLDivElement | null>(null)

  React.useEffect(() => {
    let view: { finalize?: () => void } | null = null
    let resizeObserver: ResizeObserver | null = null
    async function run() {
      if (!ref.current || !spec) return
      const embed = (await import("vega-embed")).default

      const rect = ref.current.getBoundingClientRect()
      const fallbackHeight = Math.max(240, Math.round(rect.height || 320))

      const cloned: any = JSON.parse(JSON.stringify(spec))
      cloned.background = null
      cloned.autosize = { type: "fit", contains: "padding", resize: true }
      cloned.width = "container"
      if (cloned.height === undefined || cloned.height === null || cloned.height === "container") {
        cloned.height = fallbackHeight
      }
      cloned.config = {
        ...(cloned.config ?? {}),
        axis: { ...(cloned.config?.axis ?? {}), grid: false },
        axisX: { ...(cloned.config?.axisX ?? {}), grid: false },
        axisY: { ...(cloned.config?.axisY ?? {}), grid: false },
        legend: { ...(cloned.config?.legend ?? {}), orient: "bottom", direction: "horizontal" },
        view: { ...(cloned.config?.view ?? {}), stroke: null },
      }

      const result = await embed(ref.current, cloned, { actions: false, renderer: "canvas" })
      view = result.view
      try {
        ; (view as any)?.resize?.()
          ; (view as any)?.runAsync?.()
      } catch {
        // Ignore initial resize errors.
      }

      if (typeof ResizeObserver !== "undefined") {
        resizeObserver = new ResizeObserver(() => {
          try {
            // Vega view can be resized when container dimensions change (e.g., filter panel toggles).
            ; (view as any)?.resize?.()
              ; (view as any)?.runAsync?.()
          } catch {
            // Ignore resize errors.
          }
        })
        resizeObserver.observe(ref.current)
      }
    }
    run().catch(() => {
      // Ignore render errors; component will remain empty.
    })
    return () => {
      resizeObserver?.disconnect()
      view?.finalize?.()
      if (ref.current) ref.current.innerHTML = ""
    }
  }, [spec])

  return <div ref={ref} className={cn("w-full min-h-[320px] animate-wipe-right", className)} />
}
