# Labelling guide: retrieval queries

How to grade dishes for `retrieval.yaml` (Phase 3 plan, §1). One person grades every query, so consistency matters more than any single call. When unsure between two grades, pick the lower one.

## The question to ask

**"If I typed this query, how happy would I be to see this dish?"** Judge only what the words ask for. For queries with filters (vegan, no peanuts, under 450 kcal), grade the fit to the words and ignore the filters: violations are checked separately against the reviewed allergens.

## Grades

| Grade | Meaning |
|---|---|
| 3 | Exactly what was asked: the dish named, or a dish that fully matches the description |
| 2 | Relevant: a good answer, though not the only or the most exact one |
| 1 | Related, but wrong on something that matters: the right family, the wrong dish or a missing key feature |
| 0 | Not relevant |

### Worked examples

**3, exactly what was asked**
- "butter chicken" → Butter Chicken. Items of a query's target dish are graded 3 by rule and do not appear on the page.
- "dishes cooked in coconut milk" → Prawn Moilee (prawns in a turmeric and coconut-milk curry).

**2, relevant**
- "something warm and comforting for a rainy evening" → Dal Makhani: warm, rich and homely, a good answer among several.
- "a quick bite to have with evening tea" → Osmania Biscuit: a tea-time snack, though not the only kind meant.

**1, related but wrong on something that matters**
- "butter chicken" → Chicken Tikka Masala: a similar creamy chicken curry, but not the dish asked for.
- "steamed snacks rather than fried ones" → Medu Vada: a South Indian snack like idli, but fried.

**0, not relevant**
- "a festive sweet to take to a Diwali party" → Veg Hakka Noodles.
- "made with chickpeas" → Margherita Pizza.

## Rules of thumb

- **Same dish at two restaurants:** grade both the same unless the descriptions differ in what the query asks about.
- **Colloquial and Hinglish queries:** grade by what the speaker means ("kuch meetha chahiye" means "I want something sweet"). A dish the catalog lacks ("chole bhature") makes its closest match a 1 or 2, never a 3.
- **Typos:** grade the intended query ("panner tikka" means paneer tikka).
- **Drinks and sides:** a beverage is rarely a 2 for a meal query, and a side dish rarely a 3 for "a filling meal".
- **Don't reward popularity or price.** Rating, reviews and price are not shown, on purpose.

## Consistency check

A week after grading, a seeded 10% of the graded pairs comes back for a second pass, without the first grades shown. Weighted Cohen's κ between the two passes is reported with the results.
