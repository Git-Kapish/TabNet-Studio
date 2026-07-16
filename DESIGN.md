# TabNet Studio — Design System

**Version:** 1.0
**Theme reference:** [Amber Minimal](https://tweakcn.com/editor/theme?theme=amber-minimal) (shadcn/ui token set)
**Status:** Draft for implementation

---

## 1. Overview

TabNet Studio is a research and experimentation tool for training and interpreting TabNet models. The UI should read as an **engineering instrument**, not a marketing surface: dense with information, calm in color, and fast to scan.

**Aesthetic pillars:** minimal · clean · professional · technical · calm · spacious · information-first.

**Explicitly avoid:** gradients, glassmorphism, drop-shadow-heavy cards, decorative illustration, bouncy motion, marketing-style hero sections.

**Reference products:** Weights & Biases, TensorBoard, Vercel Dashboard, Linear, GitHub, Hugging Face.

---

## 2. Design Principles

1. **Content first.** Charts, metrics, feature importance, attention masks, and model architecture always take visual priority over chrome. Use generous whitespace instead of borders/shadows to separate sections.
2. **Professional over decorative.** If a choice makes the product feel more like a landing page and less like a dashboard, discard it.
3. **Color is functional, not decorative.** Amber and status colors are reserved for meaning: active state, success, warning, error, selection, attention. Neutral gray/white carries everything else.
4. **Every screen should answer four questions at a glance:** How does this model work? How is it training? What features does it use? Why did it make this prediction?

---

## 3. Design Tokens

These are the literal values from the Amber Minimal `index.css` export (Tailwind v4 `@theme` token format). Implementation should import this file as-is rather than re-deriving values.

### 3.1 Color tokens (CSS custom properties)

| Token | Light | Dark | Role |
|---|---|---|---|
| `--background` | `#ffffff` | `#171717` | App background |
| `--foreground` | `#262626` | `#e5e5e5` | Primary text |
| `--card` | `#ffffff` | `#262626` | Card surface |
| `--card-foreground` | `#262626` | `#e5e5e5` | Text on cards |
| `--popover` | `#ffffff` | `#262626` | Popover/menu surface |
| `--popover-foreground` | `#262626` | `#e5e5e5` | Text on popovers |
| `--primary` | `#f59e0b` | `#f59e0b` | Amber accent — buttons, active states, links |
| `--primary-foreground` | `#000000` | `#000000` | Text/icons on primary fill |
| `--secondary` | `#f3f4f6` | `#262626` | Secondary surfaces/buttons |
| `--secondary-foreground` | `#4b5563` | `#e5e5e5` | Text on secondary |
| `--muted` | `#f9fafb` | `#1f1f1f` | De-emphasized backgrounds |
| `--muted-foreground` | `#6b7280` | `#a3a3a3` | De-emphasized text |
| `--accent` | `#fffbeb` | `#92400e` | Hover / subtle highlight surface |
| `--accent-foreground` | `#92400e` | `#fde68a` | Text on accent surface |
| `--destructive` | `#ef4444` | `#ef4444` | Errors, danger actions |
| `--destructive-foreground` | `#ffffff` | `#ffffff` | Text on destructive fill |
| `--border` | `#e5e7eb` | `#404040` | Dividers, card borders |
| `--input` | `#e5e7eb` | `#404040` | Input borders |
| `--ring` | `#f59e0b` | `#f59e0b` | Focus ring |

**Chart palette:**

| Token | Light | Dark |
|---|---|---|
| `--chart-1` | `#f59e0b` | `#fbbf24` |
| `--chart-2` | `#d97706` | `#d97706` |
| `--chart-3` | `#b45309` | `#92400e` |
| `--chart-4` | `#92400e` | `#b45309` |
| `--chart-5` | `#78350f` | `#92400e` |

The chart palette is a **monochrome amber ramp**, not a categorical multi-hue palette. This works for 2–3 ordered series (e.g. train/val loss) but is risky for unordered categorical data (e.g. 5 unrelated feature groups) where adjacent steps can be hard to tell apart — see §13.

**Sidebar tokens** (the sidebar is themed independently from the main surface):

| Token | Light | Dark |
|---|---|---|
| `--sidebar` | `#f9fafb` | `#0f0f0f` |
| `--sidebar-foreground` | `#262626` | `#e5e5e5` |
| `--sidebar-primary` | `#f59e0b` | `#f59e0b` |
| `--sidebar-primary-foreground` | `#ffffff` | `#ffffff` |
| `--sidebar-accent` | `#fffbeb` | `#92400e` |
| `--sidebar-accent-foreground` | `#92400e` | `#fde68a` |
| `--sidebar-border` | `#e5e7eb` | `#404040` |
| `--sidebar-ring` | `#f59e0b` | `#f59e0b` |

**Status colors** — success and info are not defined by the Amber Minimal theme and must be added as custom tokens (they don't exist in the base export):

| State | Token | Suggested value |
|---|---|---|
| Success | `--success` | Green, not in theme — add e.g. `#16a34a` (verify against `--background`/`--card` for AA) |
| Warning | — | Reuses `--primary` / `--accent` (amber is already the warning color) |
| Error | `--destructive` | `#ef4444` (defined) |
| Info | `--info` | Blue, not in theme — add e.g. `#3b82f6` (verify contrast) |

**Rule:** amber never appears purely decoratively — only for active/selected/primary-action states. If more than ~10% of a screen is amber, that's a signal something is mis-scoped.

### 3.2 Typography

| Token | Font | Use |
|---|---|---|
| `--font-sans` | **Inter** (fallback `sans-serif`) | Default UI text |
| `--font-mono` | **JetBrains Mono** (fallback `monospace`) | Hyperparameters, file paths, metric values, config, tabular numbers |
| `--font-serif` | **Source Serif 4** (fallback `serif`) | Not used in the current spec — available in the theme but no component calls for serif text. Leave unused rather than introducing it decoratively; flag if a future need (e.g. long-form docs/reports) arises. |

All three are loaded via `next/font/google` in `layout.tsx` and exposed as CSS variables (`--font-sans`, `--font-serif`, `--font-mono`), matching the `@theme inline` mapping in the theme file — no separate font-loading setup needed.

| Style | Size | Weight | Line-height | Use |
|---|---|---|---|---|
| H1 | 28–32px | Bold (700) | 1.2 | Page titles |
| H2 | 22–24px | Semibold (600) | 1.3 | Section headers |
| H3 | 18px | Semibold (600) | 1.4 | Card/subsection titles |
| Body | 16px | Regular (400) | 1.5 | Default text |
| Body small | 14px | Regular (400) | 1.5 | Secondary content |
| Caption | 13–14px | Regular (400) | 1.4 | Labels, timestamps, helper text |
| Code / mono | 13–14px | Regular (400) | 1.5 | Hyperparameters, file paths, metric values, config |

Use tabular (monospace) figures for any numeric column that updates live (loss, accuracy) so digits don't jitter horizontally.

### 3.3 Spacing

8-point grid: `4 · 8 · 12 · 16 · 24 · 32 · 48 · 64`

- Card padding: `24`
- Section gaps: `32`–`48`
- Inline control gaps: `8`–`12`
- Page margins (desktop): `32`–`48`

### 3.4 Radius & Elevation

Base radius token: `--radius: 0.375rem` (6px), with derived scale:

| Token | Value |
|---|---|
| `--radius-sm` | `calc(0.375rem - 4px)` = 2px |
| `--radius-md` | `calc(0.375rem - 2px)` = 4px |
| `--radius-lg` | `0.375rem` = 6px |
| `--radius-xl` | `calc(0.375rem + 4px)` = 10px |

This is slightly tighter than the original 8px spec — use `--radius-lg` (6px) as the default for cards/inputs/buttons, `--radius-xl` (10px) for larger containers if needed. Full pill (`9999px`) reserved for badges/tags only, as before.

Shadows are genuinely subtle in the export — all shadow steps share the same near-black, low-opacity base and mostly vary in blur/spread, not intensity:

| Token | Value |
|---|---|
| `--shadow-2xs` / `--shadow-xs` | `0px 4px 8px -1px rgba(0,0,0,0.05)` |
| `--shadow-sm` / `--shadow` | `0px 4px 8px -1px rgba(0,0,0,0.10), 0px 1px 2px -2px rgba(0,0,0,0.10)` |
| `--shadow-md` | `0px 4px 8px -1px rgba(0,0,0,0.10), 0px 2px 4px -2px rgba(0,0,0,0.10)` |
| `--shadow-lg` | `0px 4px 8px -1px rgba(0,0,0,0.10), 0px 4px 6px -2px rgba(0,0,0,0.10)` |
| `--shadow-xl` | `0px 4px 8px -1px rgba(0,0,0,0.10), 0px 8px 10px -2px rgba(0,0,0,0.10)` |

Default to `--shadow-xs`/`--shadow-sm` for cards; reserve `--shadow-md`+ for transient/overlay elements (dropdowns, modals, toasts) that need to visually separate from the whole page, not just from adjacent cards. Separation between static cards should still come primarily from spacing and `--border`, per §2.

### 3.5 Motion

- Allowed: fade, slide, expand/collapse.
- Duration: `150–250ms`, ease-out.
- Disallowed: bounce, elastic, spring overshoot, large-distance motion.
- Respect `prefers-reduced-motion`.

---

## 4. Layout

- Max content width: `1600px`, centered on ultrawide displays.
- Primary target: desktop, minimum supported width `1280px`.
- Tablet: functional, not optimized.
- Mobile: out of scope for v1 (note this explicitly to avoid silent scope creep during implementation).

```
┌─────────────────────────────────────────────────┐
│  Top Navigation                                  │
├───────────┬───────────────────────┬─────────────┤
│           │                       │             │
│  Sidebar  │     Main Workspace    │  Inspector  │
│           │                       │  (contextual│
│           │                       │   panel)    │
│           │                       │             │
└───────────┴───────────────────────┴─────────────┘
```

The Inspector panel is contextual — it appears when a node, run, or data point is selected, and collapses when nothing is selected (don't reserve permanent empty space for it).

### Top Navigation
Logo · Project name · Dataset selector · Theme toggle · GitHub link

### Left Sidebar
Dashboard · Train · Architecture Explorer · Benchmarks · Experiments · Models · Documentation

---

## 5. Components

### Buttons

| Variant | Fill | Border | Use |
|---|---|---|---|
| Primary | Amber, filled | none | Main call-to-action (Train, Save, Run) |
| Secondary | Transparent | 1px `--border` | Secondary actions |
| Ghost | Transparent | none | Low-emphasis / toolbar actions |
| Danger | Red, filled or outlined | — | Destructive actions (Delete run) |

All buttons need visible `hover`, `active`, `focus-visible`, and `disabled` states — disabled = reduced opacity + no pointer events, never color alone.

### Cards
Title (required) · optional subtitle · content. Padding `24px`. Separated by spacing and a hairline `--border`, not shadow.

### Tables
Compact density. No alternating row backgrounds — hover highlight only. Sticky header for long tables. Numeric columns right-aligned, monospaced.

### Forms
Vertical layout, label above input. Inputs: subtle border, clear focus ring (`--ring`), no heavy shadow. Inline validation messages in red/`--destructive`, below the field.

### Charts
Minimal styling, no gradients. Preferred types: line, bar, heatmap, matrix. Subtle gridlines (`--border` at low opacity). Every chart needs a text-equivalent summary for accessibility (see §9).

### Loading & Empty States
- Loading: skeletons, not spinners, wherever a layout shape is known ahead of time.
- Empty state: always instructive, never a bare blank table.
  > "No experiments yet. Train your first TabNet model to begin." *(+ primary CTA button)*

### Notifications
Toast, top-right, auto-dismiss after 3–5 seconds, with a manual dismiss control. Errors persist until dismissed or resolved.

### Icons
Lucide, consistent stroke width (default 1.5–2px), sized to the type scale (16px inline with body text, 20px in nav/toolbars).

---

## 6. Dashboard (Homepage)

**Dataset panel** — selector, row/column counts, task type, target column.

**Training panel** — controls, current status, progress bar, estimated time remaining.

**Metrics panel** — Accuracy, Precision, Recall, F1, Loss, Validation Loss, Training Time, Inference Time.

**Live charts** — Loss, Validation Loss, Learning Rate, Accuracy (all updating during training).

**Recent runs table** — Run Name, Dataset, Accuracy, Date, Status.

---

## 7. Architecture Explorer *(signature feature)*

An interactive, click-through diagram of the TabNet forward pass:

```
Input
  → Embeddings
  → BatchNorm
  → Decision Step 1
      → Sparse Mask
      → Feature Transformer
      → Decision Output
  → Decision Step 2
      → …
  → Aggregation
  → Prediction
```

Every node is clickable and opens the Inspector panel with node-specific detail:

| Node type | Inspector shows |
|---|---|
| Feature Transformer | Dimensions, parameter count, output tensor shape |
| Sparse Mask | Selected features, sparsity ratio, mask visualization |
| Decision Step | Attention weights, contribution to final output |

---

## 8. Interpretability Views

### Feature Importance
- **Local** (current sample): interactive bar chart, updates per selected row.
- **Global** (full dataset): sorted horizontal bar chart.

### Attention Masks
Rendered as heatmaps. Support hover (value tooltip), click (pin/compare), zoom, and filter-by-decision-step.

### Hyperparameter Playground
Controls: `Nd`, `Na`, number of steps, `gamma`, `lambda_sparse`, learning rate, batch size, optimizer, scheduler. Every change should reflect immediately in any dependent visualization — no "apply" button required for preview state.

---

## 9. Benchmarks & Experiments

**Benchmark page** — compares TabNet vs. Random Forest, Logistic Regression, XGBoost across Accuracy, Precision, Recall, F1, Training Time, Inference Time, Model Size, Memory Usage. Charts support sorting by any metric.

**Experiment viewer** — per run: metadata, hyperparameters, metrics, logs, TensorBoard link, model download, full configuration.

---

## 10. Accessibility

- Minimum contrast: **WCAG AA** for all text and meaningful UI elements.
- Full keyboard navigation across nav, sidebar, tables, and Architecture Explorer nodes.
- Visible focus indicators on every interactive element (`--ring`, never suppressed).
- Every chart/heatmap needs a textual/tabular equivalent (e.g., a "View as table" toggle) — don't rely on color alone to convey selection or attention weight.
- Respect `prefers-reduced-motion` and `prefers-color-scheme`.

---

## 11. Dark Mode

Both light and dark are first-class, not an afterthought. Dark mode must preserve:
- High contrast (WCAG AA, same bar as light mode)
- The same minimal, functional use of color (amber stays purposeful, not glowing/neon)
- Readable charts — verify chart series colors independently in dark mode, don't just invert.

---

## 12. Responsive Breakpoints

| Breakpoint | Width | Support level |
|---|---|---|
| Desktop | ≥1440px | Primary target |
| Small desktop / laptop | 1280–1439px | Fully supported (minimum) |
| Tablet | 768–1279px | Functional, layout may collapse Inspector into a drawer |
| Mobile | <768px | Not prioritized for v1 |

---

## 13. Open Questions for Implementation

- [ ] **Add `--success` and `--info` tokens.** The Amber Minimal export defines `--destructive` (red) but has no success/info colors — pick values and verify AA contrast against `--background`/`--card` in both light and dark before use in metric deltas, status badges, and toasts.
- [ ] **Categorical chart risk.** `--chart-1..5` is a monochrome amber ramp (`#f59e0b → #78350f`), well-suited to ordered/sequential data (loss curves, sparsity gradients) but low-contrast for unordered categorical comparisons (e.g. 5 unrelated feature groups in one legend). Decide whether Feature Importance and Benchmark charts need a secondary qualitative palette, or whether shape/pattern/label disambiguation compensates.
- [ ] **6px radius vs. original 8px spec.** The real theme ships `--radius: 0.375rem` (6px), tighter than this doc's original recommendation — confirm 6px reads correctly at data-table and card scale, or override the token if 8px is preferred.
- [ ] **Serif font is unused.** `--font-serif` (Source Serif 4) is loaded but has no assigned role in this spec — either assign it (e.g. long-form docs/report pages) or drop the font import to save weight.
- [ ] Decide Inspector panel behavior on smaller desktop widths (overlay vs. push layout).
