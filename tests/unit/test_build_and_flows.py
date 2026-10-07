"""The offline build: each step skips when current, rebuilds when an input changes; Prefect flows and CLI."""

import logging
import os
import runpy
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from prefect.states import Completed, Failed
from prefect.testing.utilities import prefect_test_harness

from food_concierge.config import REPO_ROOT, Settings, get_settings
from food_concierge.errors import CatalogValidationError
from food_concierge.flows import __main__ as cli
from food_concierge.flows import pipeline
from food_concierge.ingestion.build import build_catalog, build_search_index, source_files
from food_concierge.models.embeddings import HashingEmbedder
from food_concierge.storage.indexes import VECTORS

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "catalog"
LEXICON = REPO_ROOT / "data" / "lexicon" / "allergens.yaml"


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copytree(FIXTURE, tmp_path / "raw")
    (tmp_path / "lexicon").mkdir()
    shutil.copy(LEXICON, tmp_path / "lexicon" / "allergens.yaml")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    return tmp_path


def append_to_menu(data_dir: Path, text: str) -> None:
    with (data_dir / "raw" / "menu.csv").open("a", encoding="utf-8") as handle:
        handle.write(text)


def test_steps_skip_when_current_and_rebuild_when_inputs_change(data_dir: Path) -> None:
    settings = Settings()
    embedder = HashingEmbedder()

    first = [build_catalog(settings), build_search_index(settings, embedder)]
    again = [build_catalog(settings), build_search_index(settings, embedder)]
    (data_dir / "lexicon" / "allergens.yaml").write_text(
        LEXICON.read_text(encoding="utf-8").replace("version: 1", "version: 2"), encoding="utf-8"
    )
    changed = [build_catalog(settings), build_search_index(settings, embedder)]

    assert [r.rebuilt for r in first] == [True, True]
    assert [r.rebuilt for r in again] == [False, False]  # re-run with unchanged inputs: a no-op
    assert [r.rebuilt for r in changed] == [True, True]  # the lexicon is an input of both
    assert changed[0].data_sha256 != first[0].data_sha256 == first[1].data_sha256
    assert first[0].output == data_dir / "processed" / "catalog.db"


def test_index_rebuilds_for_another_embedder_or_changed_files(data_dir: Path) -> None:
    settings = Settings()
    build_catalog(settings)
    build_search_index(settings, HashingEmbedder())

    assert build_search_index(settings, HashingEmbedder(query_prefix="query: ")).rebuilt
    (data_dir / "processed" / "indexes" / VECTORS).write_bytes(b"tampered")
    assert build_search_index(settings, HashingEmbedder(query_prefix="query: ")).rebuilt
    assert build_search_index(settings, HashingEmbedder(query_prefix="query: "), force=True).rebuilt
    assert build_catalog(settings, force=True).rebuilt


def test_attributions_are_an_input_when_present(data_dir: Path) -> None:
    settings = Settings()
    assert [p.name for p in source_files(settings)] == ["restaurants.csv", "menu.csv", "allergens.yaml"]

    (data_dir / "raw" / "attributions.csv").write_text("item_id\n", encoding="utf-8")
    assert source_files(settings)[-1].name == "attributions.csv"


def test_catalog_warnings_are_logged(data_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    menu = data_dir / "raw" / "menu.csv"
    menu.write_text(menu.read_text(encoding="utf-8").replace(",420,470,16,55,20,", ",420,300,16,55,20,"), "utf-8")

    with caplog.at_level(logging.WARNING):
        assert build_catalog(Settings()).rebuilt

    assert any("calories differ" in str(getattr(r, "issue", "")) for r in caplog.records)


def test_invalid_catalog_fails_the_build(data_dir: Path) -> None:
    append_to_menu(data_dir, "broken,row\n")

    with pytest.raises(CatalogValidationError):
        build_catalog(Settings())


@pytest.fixture(scope="module")
def prefect_harness() -> Iterator[None]:
    with prefect_test_harness():
        yield


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "get_embedder", lambda settings: HashingEmbedder())  # no model download


@pytest.mark.usefixtures("prefect_harness", "data_dir", "fake_embedder")
def test_flows_build_then_do_nothing() -> None:
    first = pipeline.build_flow()
    second = pipeline.build_flow()

    assert [(r.step, r.rebuilt) for r in first] == [("catalog", True), ("index", True)]
    assert [(r.step, r.rebuilt) for r in second] == [("catalog", False), ("index", False)]
    assert pipeline.ingest_flow(force=True).rebuilt


@pytest.mark.usefixtures("prefect_harness", "data_dir", "fake_embedder")
def test_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ingest"]) == 0
    assert cli.main(["index"]) == 0
    assert cli.main(["build"]) == 0
    out = capsys.readouterr().out.splitlines()

    assert out[0].startswith("catalog: rebuilt")
    assert out[1].startswith("index: rebuilt")
    assert out[2:] == [line.replace("rebuilt", "up to date") for line in out[:2]]


def test_module_entry_point(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["food_concierge.flows", "--help"])
    monkeypatch.delitem(sys.modules, "food_concierge.flows.__main__")  # imported above as `cli`; run it afresh

    with pytest.raises(SystemExit) as exited:
        runpy.run_module("food_concierge.flows", run_name="__main__")

    assert exited.value.code == 0
    assert "does nothing when both are current" in capsys.readouterr().out


def test_retry_only_transient_failures() -> None:
    def failed(exc: BaseException) -> Any:
        return Failed(data=exc)

    assert pipeline._transient(None, None, failed(ConnectionError("reset")))
    assert pipeline._transient(None, None, failed(TimeoutError()))
    assert not pipeline._transient(None, None, failed(ValueError("bad catalog")))
    assert not pipeline._transient(None, None, Completed(data=1))


def test_prefect_analytics_are_off() -> None:
    assert os.environ["PREFECT_SERVER_ANALYTICS_ENABLED"] == "false"
