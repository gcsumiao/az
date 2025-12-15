import { z } from "zod"

export const ThresholdsSchema = z.object({
  fill_rate: z.number(),
  outs_exposure: z.number(),
  coverage: z.number(),
  woh_min: z.number(),
  woh_max: z.number(),
  billbacks_pct: z.number(),
  min_ly_rev_floor: z.number(),
})

export const DashboardFiltersSchema = z.object({
  selected_weeks: z.array(z.number()),
  selected_categories: z.array(z.string()),
  selected_parts: z.array(z.string()),
  sku_query: z.string(),
  top_n: z.number(),
  thresholds: ThresholdsSchema,
  show_forecast_overlay: z.boolean(),
})

export type DashboardFilters = z.infer<typeof DashboardFiltersSchema>

export const OverviewResponseSchema = z.object({
  snapshot: z.object({
    fiscal_year: z.number().nullable(),
    snapshot_week: z.number().nullable(),
    prev_week: z.number().nullable(),
  }),
  kpis: z.object({
    units: z.number().nullable(),
    revenue: z.number().nullable(),
    asp: z.number().nullable(),
    units_wow: z.number().nullable(),
    revenue_wow: z.number().nullable(),
  }),
  range_totals: z.object({
    units: z.number().nullable(),
    revenue: z.number().nullable(),
  }),
  gm: z.object({
    gross_margin: z.number().nullable(),
    gross_margin_pct: z.number().nullable(),
    top_product: z
      .object({
        part_number: z.string(),
        description: z.string().nullable(),
        gross_margin: z.number().nullable(),
      })
      .nullable(),
    top_category: z
      .object({
        major_category: z.string(),
        gross_margin: z.number().nullable(),
      })
      .nullable(),
  }),
  heroes: z.object({
    product: z.object({
      hero: z.record(z.any()).nullable(),
      rising: z.record(z.any()).nullable(),
      declining: z.record(z.any()).nullable(),
    }),
    category: z.object({
      hero: z.record(z.any()).nullable(),
      rising: z.record(z.any()).nullable(),
      declining: z.record(z.any()).nullable(),
    }),
  }),
  charts: z.record(z.any()),
  alerts: z.array(
    z.object({
      alert_type: z.string(),
      severity: z.string(),
      message: z.string(),
      action: z.string(),
    }),
  ),
})

export type OverviewResponse = z.infer<typeof OverviewResponseSchema>

