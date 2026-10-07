# Catalog source files

The restaurant catalog authored for this project. It is **synthetic by design**: the restaurants are fictional, and the dishes, prices, ratings and nutrition are plausible but invented. Metrics computed on it describe how the system behaves, not real-world accuracy.

The catalog is written in reviewed batches of about 50 dishes and validated by `food_concierge.ingestion.loader`.

## Files

| File | One row per | Notes |
|---|---|---|
| `restaurants.csv` | restaurant | `cuisines` is a `;`-separated list from the closed set in `ingestion/taxonomy.py` |
| `menu.csv` | dish at a restaurant | the same dish at two restaurants is two rows |
| `photo_review.csv` | photo candidate for a dish | Wikimedia Commons files under accepted licences, plus a `none` row per dish; the owner marks one row per dish `yes` in `approved` |
| `attributions.csv` | approved photo | file page, file URL, author, licence, licence URL, SHA-256 and size; written by `scripts/fetch_photos.py approve` |

## Photos

Photos come from Wikimedia Commons only, under CC0, public domain, CC BY or CC BY-SA. NC, ND, GFDL-only, non-free and unknown licences are refused, and so are files with restrictions such as trademarks or personality rights. The owner approves one photo per dish, or marks the dish as having no suitable photo.

The image files are not committed. `scripts/fetch_photos.py sync` downloads each one into the photo cache outside the repository, checks it against its pinned SHA-256, and builds a WebP thumbnail of at most 480 px. Wherever a photo is shown, its author and licence are shown with it.

## `menu.csv` columns

- **Lists** (`ingredients`, allergen columns) are separated by `;`.
- **`diet`:** `vegan`, `vegetarian` (lacto-ovo: may contain milk and eggs) or `non_vegetarian`.
- **`contains_egg`:** `yes` or `no`. It drives the `eggless` filter, and is `yes` for hidden egg too (mayonnaise, Caesar dressing, tartare sauce).
- **`serves`:** a number of people (`2`) or a range (`1-2`).
- **`price_inr`:** whole rupees.
- **`kcal`, `protein_g`, `carbs_g`, `fat_g`:** per serving, plausible values, not measurements. Calories agree with 4P + 4C + 9F within 25%.
- **`label_allergens`:** what the restaurant declares, in everyday words ("dairy", "nuts").
  - A blank cell means the dish is **not labelled**. About 30% of dishes are left that way on purpose, because real menus are like that.
  - `none` means the restaurant **declares** the dish allergen-free.
  - Some labels are deliberately incomplete or wrong (for example, a dish declared `none` whose sambar powder contains wheat-based hing). The system must never trust labels alone.
- **`true_allergens`:** the hand-checked allergens of the full recipe, including what is inside composite ingredients such as sauces, chutneys and spice mixes.
  - Only the 15 canonical keys are accepted, `none` when there are none, and wheat must be listed with gluten.
  - This column is used **only by tests and evaluation**: the loader keeps it out of the records the system uses.
  - The owner reviews every entry before a batch merges, and that review is the ground truth's authority.

## Recipe assumptions behind `true_allergens`

Composite ingredients are opaque in the ingredient list. The ground truth assumes these recipes:

| Ingredient | Assumed to contain | Why |
|---|---|---|
| hing (asafoetida), sambar powder | wheat, gluten | Commercial compounded hing is usually cut with wheat flour; sambar powder includes it |
| soy sauce | soy, wheat, gluten | Brewed with wheat |
| pav, bun, bread, puri, maida, atta, rava, semolina, noodles, pasta | wheat, gluten | Wheat flour |
| egg noodles, mayonnaise, Caesar dressing, tartare sauce | eggs | Egg or egg yolk |
| Caesar dressing | fish, mustard, milk | Anchovy, mustard and parmesan |
| tartare sauce | mustard | Mayonnaise-based with mustard |
| dry garlic chutney, farsan | peanuts | Mumbai-style recipes include roasted peanuts |
| gingelly oil | sesame | Unrefined sesame oil |
| chocolate | milk, soy | Milk solids and soy lecithin |
| coconut, coconut oil, coconut milk | nothing | Coconut is not treated as a tree nut, and coconut milk is not milk |
| chilli sauce, vinegar | nothing | Assumed free of added sulphites |
| biryani masala, chaat masala, pav bhaji masala, misal masala, chole masala, green and tamarind chutneys | nothing | Spices, herbs and fruit only |
| mustard greens (sarson), kasundi | mustard | Mustard plant and mustard sauce |
| barley | gluten (not wheat) | A gluten cereal that is not wheat |
| makki atta (maize), rice flour, rice ada | nothing | Gluten-free grains |
| dried apricots | sulphites | Usually preserved with sulphur dioxide |
| raisins | nothing | Dark raisins, assumed unsulphited |
| paratha, rotli, methi muthia | wheat, gluten | Wheat flour (muthia mixes wheat and gram flour) |
| Gujarati dal | peanuts | Traditionally cooked with peanuts |
| kadhi, khaman, shrikhand (in the thali) | milk | Made with curd |
| chicken tikka | milk | Marinated in curd |
| poppy seeds, fruit salt, papad | nothing | Not in the 15 keys (papad is urad dal) |
| balsamic vinegar | sulphites | Wine vinegar contains sulphites |
| candied fruit, tutti frutti | sulphites | Usually preserved with sulphur dioxide |
| puff pastry | wheat, gluten | Indian bakery puff pastry is made with vegetable fat, not butter |
| teriyaki sauce, gochujang, peanut satay sauce, soy dipping sauce | soy, wheat, gluten | Made with soy sauce or wheat |
| Japanese curry roux | wheat, gluten, soy, milk | Flour-based roux with soy and milk powder |
| kimchi | fish, crustaceans | Fish sauce and salted shrimp |
| fish sauce | fish | Fermented fish |
| chilli paste (tom yum) | crustaceans | Contains dried shrimp |
| wasabi | mustard | Served wasabi is mostly horseradish with mustard |
| granola | gluten (not wheat) | Oats are a gluten cereal |
| glutinous rice | nothing | "Glutinous" refers to stickiness; rice has no gluten |
| vegetarian green curry paste | nothing | Made without shrimp paste |
| falafel, hummus base, garlic toum | nothing besides sesame in tahini | Chickpeas, herbs, garlic and oil |
| chicken shawarma | milk | Marinated in yogurt |
| silver varq | nothing | Pure silver leaf |
