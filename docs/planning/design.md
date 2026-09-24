# Design — Streamlit UI

> Revision 2 (2026-09-24): agent UI elements added in §9.

> Visual and interaction spec for `frontend/`. The UI is a thin client over the API (Architecture §8).

---

## 1. Principles

1. **Food first.** Dish images carry the recommendation; text supports them.
2. **Trust over flash.** Diet and allergen info is always visible, labelled with text (never colour alone), and carries a disclaimer.
3. **Calm and warm.** Warm neutrals and one appetising accent; no gradients, no emoji clutter.
4. **Honest system state.** Show when results are degraded, filtered to nothing, or served from cache.

## 2. Colour palette

Warm "kitchen" palette. Contrast ratios checked against the page background for text use.

| Token | Light | Dark | Use |
|---|---|---|---|
| `primary` | `#B5472F` terracotta (5.4:1 on bg) | `#E07A5F` | Buttons, active states, links, focus ring |
| `bg` | `#FFFBF5` warm off-white | `#1C1714` | Page background |
| `surface` | `#F5EDE3` | `#2A221D` | Sidebar, cards |
| `border` | `#E6DACB` | `#3D322A` | Card borders, dividers |
| `text` | `#2B2118` espresso | `#F3ECE4` | Body text |
| `text-muted` | `#6B5E53` (6.1:1) | `#BFB2A5` | Secondary text, captions |
| `veg` | `#2E7D32` | `#81C784` | Vegetarian / vegan badge (with "VEG" / "VEGAN" text) |
| `nonveg` | `#B3261E` | `#EF9A9A` | Non-vegetarian badge (with "NON-VEG" text) |
| `warning` | `#8A5300` on `#FFF3DC` | `#FFCC80` on `#3A2A12` | Allergen chips, "unverified ingredients", degraded notice |
| `info` | `#2F5D8A` | `#90CAF9` | Debug panel, cache notice |

Veg/non-veg colours follow the Indian food-labelling convention the dataset uses (green/red), always paired with a text label and an icon shape.

## 3. Typography

| Role | Font | Size / weight |
|---|---|---|
| Headings | **Fraunces** (warm serif), fallback Georgia, serif | H1 32/600, H2 24/600, card title 20/600 |
| Body / UI | **Inter**, fallback system sans-serif | 16/400; labels 14/500 |
| Numbers (price, kcal, macros) | Inter with tabular figures | 14/500 |
| Code / debug | JetBrains Mono, fallback monospace | 13/400 |

If the installed Streamlit version doesn't support custom theme fonts, fall back to `font = "sans serif"` and load no web fonts. Verify in Phase 5.

## 4. Layout

```
┌───────────────────────────── wide layout ──────────────────────────────┐
│ Sidebar (surface)            │ Main                                     │
│ ─ Logo + "Menu Finder"       │ H1  What are you craving?                │
│ ─ Filters                    │ Subtitle (muted): describe it or upload  │
│   Diet: Any/Veg/Vegan/Non-veg│      a photo of a dish                   │
│   Exclude allergens (multi)  │ ┌ chat history (newest at bottom) ─────┐ │
│   Max calories (slider)      │ │ user bubble (+ image thumbnail)      │ │
│   Max price (slider)         │ │ assistant: 1-line summary            │ │
│   Cuisine (multi)            │ │ [card][card][card]  (3 columns;      │ │
│   Sort: Relevance/Rating/    │ │  stacked on narrow screens)          │ │
│         Price/Calories       │ │ "Show different options" button      │ │
│ ─ Model: provider select     │ └──────────────────────────────────────┘ │
│ ─ Allergen disclaimer        │ [image uploader]  [chat_input ........]  │
│ ─ About / debug toggle       │                                          │
└──────────────────────────────┴──────────────────────────────────────────┘
```

- `st.set_page_config(layout="wide", page_title="Menu Finder")`.
- Use `st.chat_input` and `st.chat_message` (drop `streamlit-chat`).
- The image uploader sits above the input; once a message is sent, the attached image is shown in the user bubble and the uploader is cleared (fixes baseline defect: re-sending a stale image).

## 5. Result card

```
┌───────────────────────────┐
│ [thumbnail 16:10, rounded]│
│ Dish Name            ★4.6 │   title 20/600; rating right-aligned
│ Restaurant · Cuisine      │   muted 14
│ [VEG] [Contains: dairy,   │   diet badge + allergen chips (warning)
│  gluten]                  │
│ "Why: light, high-protein │   reason from LLM (or templated if degraded)
│  and under 400 kcal"      │
│ 350 kcal · P 12g C 30g F15│   tabular numbers
│ $12 · Serves 1–2   👍 👎   │   feedback buttons
└───────────────────────────┘
```

- Card: `surface` background, 1 px `border`, 12 px radius, 16 px padding, 12 px gap.
- Image `alt`/caption = dish name. When an image is missing, show a neutral placeholder tile.
- Allergen chips with `source=inferred` get a dotted underline and a tooltip ("inferred from ingredients"). `allergen_unverified` items show a ⚠ "Ingredients not fully verifiable" chip.

## 6. States and copy

| State | Treatment |
|---|---|
| Empty (first load) | 3 example prompt chips: "Spicy vegetarian under $10", "Something like sushi but vegan", "High-protein, low-calorie dinner" |
| Loading | `st.status` with steps: "Understanding your request → Finding dishes → Writing suggestions" |
| No matches | Warning box: "No dishes match all filters. The strictest one is *Max calories 150*." + a "Relax filters" button |
| Degraded | Info box: "Showing best matches without AI summaries (the model service is busy)." |
| Budget/rate limit (429) | "We've hit today's demo limit for this model. Try the retrieval-only search or come back later." |
| Error (5xx/timeout) | "Something went wrong on our side. Please try again." + request ID in small muted text |
| General question | Plain assistant message, no cards |

Tone: friendly, brief, concrete. No exclamation marks in error states. Never blame the user.

## 7. Accessibility

- Text contrast ≥ 4.5:1; badges carry text as well as colour.
- All interactive controls have visible labels (no placeholder-only inputs).
- Numbers carry units (kcal, g, $).
- Layout holds at 360 px width (columns stack).

## 8. Theme config (`frontend/.streamlit/config.toml`)

```toml
[theme]
base = "light"
primaryColor = "#B5472F"
backgroundColor = "#FFFBF5"
secondaryBackgroundColor = "#F5EDE3"
textColor = "#2B2118"
font = "sans serif"   # upgrade to Inter/Fraunces if supported (see §3)
```
Streamlit's built-in dark mode uses the dark column values via a `[theme.dark]` section if the version supports it; otherwise use the light theme only.

## 9. Agent UI (revision 2)

| Element | Spec |
|---|---|
| **Tool timeline** | Under each assistant turn, a collapsible `st.status` list of steps as they stream: "Searching dishes (vegan, ≤ 400 kcal)", "Checking allergens", "Computing meal totals", "Verifying plan ✓". Muted text, `info` colour icons; failed/retried steps shown honestly. |
| **Meal-plan card** | One row per course (starter / main / dessert) using the result-card layout (§5, compact); footer with **totals vs limits** as progress bars (calories, price) in `primary`, turning `warning` above 90% of a limit. Totals come from `meal_totals`, labelled "calculated". |
| **Confirmation (interrupt)** | Inline `warning` box: "Tiramisu has ingredients we can't fully verify for *egg*. Include it anyway?" with **Include** / **Find an alternative** buttons. No modal dialogs. |
| **Applied constraints chips** | Row of removable chips above results showing every active constraint, marked "from you" or "inferred"; inferred chips have a dotted border and must be confirmable (A15). |
| **Preferences panel** (sidebar) | "Remembered about you": allergies, diet, dislikes, each with a remove ×, plus **Forget everything**. Only items the user stated. |
| **Model / trace footer** | Small muted line per answer: model used (and "fallback" if used), latency, trace ID (copy button). |
| **AI-interaction disclosure** | Persistent caption under the title: "You're chatting with an AI assistant. Always confirm allergens with the restaurant." |
| **Data-use notice** | Under the uploader: "Photos are sent to a free AI service that may use them to improve its models. Don't upload personal photos." |
| **Degraded mode** | `info` box: "The AI planner is at its free limit right now. Showing matching dishes without a plan." |
