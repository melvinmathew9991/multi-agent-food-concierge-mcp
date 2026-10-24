"""The MCP server over the domain services (architecture §5): 6 tools, 4 resources, 2 prompts.

- Tools are read-only and idempotent; constraints are enforced inside them, so a client can't get around them.
  ``search_dishes`` and ``find_similar`` filter in SQL before ranking, then re-check every result against the
  same constraints and drop any that breaks one: a dish that violates them is never returned.
- Domain errors come back as tool results with ``isError`` and the error's message, which is written to be safe
  to show. Anything unexpected reaches the client only as "Error executing tool <name>"; the traceback is logged.
- Inputs are bounded: queries by ``max_query_chars``, dish lists by ``MAX_DISHES``, results by ``MAX_RESULTS``,
  photos by ``max_image_mb`` before they are decoded.
"""

from __future__ import annotations

import base64
import binascii
import inspect
import json
import logging
from collections.abc import Callable
from typing import Annotated, Any, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ResourceNotFoundError, ToolError
from mcp_types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from food_concierge import __version__
from food_concierge.errors import AppError, InputValidationError, NotFoundError
from food_concierge.ingestion.photos import thumbnail_path
from food_concierge.ingestion.taxonomy import EU_14, US_BIG_9, USER_TERMS, Allergen, Cuisine, Diet, parse_allergen_terms
from food_concierge.mcp_server.access import ToolAccess
from food_concierge.mcp_server.services import Services
from food_concierge.services.allergens import DISCLAIMER, AllergenCheck
from food_concierge.services.catalog import MAX_DISHES, DishDetail, dish_ids
from food_concierge.services.constraints import Constraints, parse_constraints
from food_concierge.services.nutrition import MAX_DINERS, MealTotals
from food_concierge.services.search import MAX_RESULTS, Dishes
from food_concierge.services.vision import DESCRIBE_IN_WORDS, PhotoDescription

logger = logging.getLogger(__name__)

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
# Describing a photo calls a hosted model: not idempotent in cost, and it reaches outside the server.
DESCRIBES = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True)

INSTRUCTIONS = (
    "Tools for finding dishes at the restaurants in this catalog under a diner's hard constraints (diet, "
    "allergens, calories, price, cuisine). Search never returns a dish that breaks the constraints given. Pass "
    "constraints as arguments; ones the query states in words ('peanut-free', 'vegan') are applied too, and the "
    "result's note names them. A dish "
    "marked unverified has an ingredient whose recipe can't be checked: ask the diner before recommending it to "
    "someone avoiding an allergen, then search again with include_unverified. may_contain allergens come from the "
    "photo only and are warnings. Prices are in Indian rupees. " + DISCLAIMER
)

_T = TypeVar("_T")
_F = TypeVar("_F", bound=Callable[..., Any])

Query = Annotated[str, Field(min_length=1, description="What the diner wants, in their words.")]
DishId = Annotated[str, Field(min_length=1, max_length=40, description="A dish id, such as 's05'.")]
DietArg = Annotated[Diet | None, Field(description="vegan, vegetarian (lacto-ovo) or non_vegetarian (anything).")]
Eggless = Annotated[bool, Field(description="Leave out dishes with egg.")]
AllergenTerms = Annotated[
    list[str] | None,
    Field(max_length=20, description="Allergens to avoid, in plain words: 'dairy', 'nuts', 'shellfish', 'gluten'."),
]
MaxKcal = Annotated[int | None, Field(gt=0, description="Most calories one serving of a dish may have.")]
MaxPrice = Annotated[int | None, Field(gt=0, description="Most one dish may cost, in rupees.")]
Cuisines = Annotated[list[Cuisine] | None, Field(max_length=10, description="Only these cuisines.")]
IncludeUnverified = Annotated[
    bool, Field(description="Include dishes whose recipe can't be checked; only after the diner has agreed.")
]
ExcludeIds = Annotated[list[str] | None, Field(max_length=MAX_DISHES, description="Dish ids not to return again.")]
K = Annotated[int, Field(ge=1, le=MAX_RESULTS, description="How many dishes to return.")]
DishIds = Annotated[list[str], Field(min_length=1, max_length=MAX_DISHES, description="Dish ids, such as 's05'.")]
Diners = Annotated[int, Field(ge=1, le=MAX_DINERS, description="How many people share the meal.")]


class _Out(BaseModel):
    model_config = ConfigDict(frozen=True)


class AllergenChecks(_Out):
    checks: list[AllergenCheck]
    disclaimer: str = DISCLAIMER


def _tool_errors(call: Callable[[], _T]) -> _T:
    """Run a tool body, returning domain errors to the model as ``isError`` results with their safe message."""
    try:
        return call()
    except AppError as error:
        raise ToolError(error.message) from error


def _resource_errors(call: Callable[[], _T]) -> _T:
    try:
        return call()
    except NotFoundError as error:
        raise ResourceNotFoundError(error.message) from error
    except AppError as error:
        raise ResourceError(error.message) from error


def _constraints(
    diet: Diet | None,
    eggless: bool,
    exclude_allergens: list[str] | None,
    max_kcal: int | None,
    max_price_inr: int | None,
    cuisines: list[Cuisine] | None,
    include_unverified: bool,
) -> Constraints:
    return parse_constraints(
        {
            "diet": diet,
            "eggless": eggless,
            "exclude_allergens": exclude_allergens or [],
            "max_kcal": max_kcal,
            "max_price_inr": max_price_inr,
            "cuisines": cuisines or [],
            "include_unverified": include_unverified,
        }
    )


def create_server(services: Services) -> MCPServer:
    """A server over ``services``; run it with ``.run("stdio")``, or connect ``mcp.Client(server)`` in tests."""
    settings = services.settings
    server: MCPServer = MCPServer(
        name="food-concierge",
        title="Food Concierge",
        version=__version__,
        instructions=INSTRUCTIONS,
        middleware=[ToolAccess(settings)],
    )
    search = services.search.search

    def tool(annotations: ToolAnnotations) -> Callable[[_F], _F]:
        """``server.tool`` with the docstring dedented, so clients don't see the source's indentation."""

        def register(fn: _F) -> _F:
            return server.tool(annotations=annotations, description=inspect.cleandoc(fn.__doc__ or ""))(fn)

        return register

    @tool(READ_ONLY)
    def search_dishes(
        query: Query,
        diet: DietArg = None,
        eggless: Eggless = False,
        exclude_allergens: AllergenTerms = None,
        max_kcal: MaxKcal = None,
        max_price_inr: MaxPrice = None,
        cuisines: Cuisines = None,
        include_unverified: IncludeUnverified = False,
        exclude_ids: ExcludeIds = None,
        k: K = 5,
    ) -> Dishes:
        """Find dishes for a request, ranked by relevance. Never returns a dish that breaks the constraints.

        Constraints the query states in words ("peanut-free", "no dairy", "vegan") are added to the arguments, and
        the note says which; pass them as arguments too.
        """

        def run() -> Dishes:
            constraints = _constraints(
                diet, eggless, exclude_allergens, max_kcal, max_price_inr, cuisines, include_unverified
            )
            skip = set(dish_ids(exclude_ids)) if exclude_ids else set()
            return search(query, constraints, k, skip)

        return _tool_errors(run)

    @tool(READ_ONLY)
    def find_similar(
        dish_id: DishId,
        diet: DietArg = None,
        eggless: Eggless = False,
        exclude_allergens: AllergenTerms = None,
        max_kcal: MaxKcal = None,
        max_price_inr: MaxPrice = None,
        cuisines: Cuisines = None,
        include_unverified: IncludeUnverified = False,
        k: K = 5,
    ) -> Dishes:
        """Dishes like this one (not the same dish at another restaurant), under the same kind of constraints."""

        def run() -> Dishes:
            constraints = _constraints(
                diet, eggless, exclude_allergens, max_kcal, max_price_inr, cuisines, include_unverified
            )
            dish = services.catalog.get_dish(dish_id)
            same = {item_id for item_id, key in services.store.dish_keys().items() if key == dish.dish_key}
            text = f"{dish.name}. {dish.description} Ingredients: {', '.join(dish.ingredients)}."
            return search(text[: settings.max_query_chars], constraints, k, same, read_query=False)

        return _tool_errors(run)

    @tool(READ_ONLY)
    def get_dish(dish_id: DishId) -> DishDetail:
        """One dish in full: ingredients, nutrition (estimates per serving), price for the dish, restaurant, allergen
        tags with their evidence, photo."""
        return _tool_errors(lambda: services.catalog.get_dish(dish_id))

    @tool(READ_ONLY)
    def check_allergens(dish_ids: DishIds, allergens: AllergenTerms = None) -> AllergenChecks:
        """Which of these allergens each dish contains, may contain (photo only), or can't be checked for.

        With no allergens given, every allergen is checked. ``clear`` is true only when nothing is found and the
        recipe can be checked.
        """

        def run() -> AllergenChecks:
            try:
                asked = parse_allergen_terms(";".join(allergens or []))
            except ValueError as error:  # the same words search takes, named in the policy resource
                raise InputValidationError(f"{str(error).capitalize()}. See menu://policy/allergens.") from error
            return AllergenChecks(checks=services.allergens.check(dish_ids, asked))

        return _tool_errors(run)

    @tool(READ_ONLY)
    def meal_totals(dish_ids: DishIds, diners: Diners = 1) -> MealTotals:
        """Calories, protein, carbs, fat, price in rupees and servings for a meal shared by `diners`: the whole order
        and each diner's equal share. A dish serving a range of people (1-2) makes a min-max range; list a dish twice
        to count it twice. Nutrition values are the menu's estimates per serving, not measurements."""
        return _tool_errors(lambda: services.nutrition.meal_totals(dish_ids, diners))

    @tool(DESCRIBES)
    def describe_image(
        image_base64: Annotated[str, Field(min_length=1, description="The photo's bytes, base64-encoded.")],
    ) -> PhotoDescription:
        """Describe a photo of a dish in words, for search_dishes. If it can't be described, ask for words."""

        def run() -> PhotoDescription:
            if services.vision is None:
                raise InputValidationError(DESCRIBE_IN_WORDS)
            if len(image_base64) > (settings.max_image_bytes + 2) // 3 * 4:  # before decoding anything
                raise InputValidationError(f"The image is larger than {settings.max_image_mb:g} MB.")
            try:
                data = base64.b64decode(image_base64, validate=True)
            except (binascii.Error, ValueError) as error:
                raise InputValidationError("The image is not valid base64.") from error
            return services.vision.describe(data)

        return _tool_errors(run)

    @server.resource("menu://catalog", mime_type="application/json", description="Every dish, in short form.")
    def catalog() -> str:
        dishes = [dish.model_dump(mode="json") for dish in services.catalog.all_summaries()]
        return json.dumps({"currency": "INR", "disclaimer": DISCLAIMER, "dishes": dishes})

    @server.resource("menu://dishes/{dish_id}", mime_type="application/json", description="One dish in full.")
    def dish(dish_id: str) -> str:
        return _resource_errors(lambda: services.catalog.get_dish(dish_id).model_dump_json())

    @server.resource("menu://dishes/{dish_id}/image", mime_type="image/webp", description="A dish's photo (WebP).")
    def dish_image(dish_id: str) -> bytes:
        def read() -> bytes:
            photo = services.catalog.get_dish(dish_id).photo
            if photo is None:
                raise NotFoundError(f"Dish {dish_id} has no photo.")
            path = thumbnail_path(settings.photo_cache_dir, photo)
            if not path.is_file():
                raise NotFoundError(f"The photo of {dish_id} isn't stored on this server; its photo_url links to it.")
            return path.read_bytes()

        return _resource_errors(read)

    @server.resource(
        "menu://policy/allergens", mime_type="application/json", description="The allergens, their words, the rules."
    )
    def allergen_policy() -> str:
        words: dict[str, list[str]] = {allergen.value: [] for allergen in Allergen}
        for term, keys in sorted(USER_TERMS.items()):
            for key in keys:
                words[key.value].append(term)
        return json.dumps(
            {
                "allergens": [
                    {"key": a.value, "eu_14": a in EU_14, "us_big_9": a in US_BIG_9, "words": words[a.value]}
                    for a in Allergen
                ],
                "levels": {
                    "contains": "A label or an ingredient says so. Search leaves these dishes out.",
                    "may_contain": "Only the photo suggests it. Shown as a warning, never filtered on.",
                    "unverified": "An ingredient's recipe can't be checked; ask the diner before recommending it.",
                },
                "disclaimer": DISCLAIMER,
            }
        )

    @server.prompt(description="Plan a meal under a diner's constraints, checked with the tools.")
    def plan_meal(constraints: str) -> str:
        return (
            f"Plan a meal for a diner with these constraints: {constraints}\n"
            "Use search_dishes with the constraints as arguments, not only in the query. Check the meal with "
            "meal_totals for the number of diners, against any calorie limit per person and budget for the whole "
            "order, and with check_allergens for every allergen named. Don't recommend an unverified dish to someone "
            "avoiding an allergen without asking them first. Give prices in rupees, and end with this note: "
            + DISCLAIMER
        )

    @server.prompt(description="Find catalog dishes like one the diner describes or photographed.")
    def find_similar_dish(description: str) -> str:
        return (
            f"A diner is looking for a dish like this: {description}\n"
            "Search for it with search_dishes, using the description's main ingredients and style as the query. "
            "If they mentioned a diet or allergens, pass them as arguments. Say why each result is similar, and "
            "that a photo shows a dish, not its recipe."
        )

    return server
