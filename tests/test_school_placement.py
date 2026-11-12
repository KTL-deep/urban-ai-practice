"""Known small artificial territory; never used by the Pavlovsk application."""

import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[1] / "school-placement"
sys.path.insert(0, str(ROOT))
from data_loader import DataError, load_study
from model import calculate, verify_baseline
from search import search
from export import csv_bytes, quarters_frame, geojson


@pytest.fixture
def study_dir(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    blocks = gpd.GeoDataFrame({"block_id": [0, 1, 2], "population": [2500, 2500, 2500],
        "capacity_school": [100, 0, 0]}, geometry=[box(30+i*.01,59,30.005+i*.01,59.005) for i in range(3)], crs=4326)
    blocks.to_file(data / "blocks.geojson", driver="GeoJSON")
    pd.DataFrame([[0, 5, np.inf], [7, 0, np.inf], [np.inf, np.inf, 0]],
                 index=[0, 1, 2], columns=[0, 1, 2]).to_csv(data / "accessibility_matrix_drive.csv")
    (data / "study_info.json").write_text(json.dumps({"travel_time_units":"minutes", "travel_mode":"drive (автомобиль)"}),encoding="utf-8")
    return data


def test_real_method_baseline_and_scenarios(study_dir):
    study = load_study(study_dir)
    baseline = calculate(study)
    assert baseline.totals["demand"] == 900
    assert baseline.totals["served"] == 100
    assert baseline.totals["deficit"] == 800
    zero = calculate(study, (), baseline)
    pd.testing.assert_series_equal(zero.rows.deficit, baseline.rows.deficit)
    one = calculate(study, [(1, 600)], baseline)
    assert one.totals["added_capacity"] == 600
    assert one.totals["capacity"] == 700
    assert one.rows.loc[0, "demand_within"] > baseline.rows.loc[0, "demand_within"]
    assert one.rows.loc[1, "demand_within"] > baseline.rows.loc[1, "demand_within"]
    assert one.rows.loc[2, "demand_within"] == 0
    assert one.totals["served"] <= one.totals["capacity"]
    pair = calculate(study, [(0, 300), (1, 300)], baseline)
    assert pair.totals["added_capacity"] == 600
    same = calculate(study, [(1, 300), (1, 300)], baseline)
    assert same.totals["deficit"] == one.totals["deficit"]
    assert calculate(study).totals == baseline.totals
    assert study.capacity.tolist() == [100, 0, 0]
    assert study.matrix.loc[0, 1] == 5 and study.matrix.loc[1, 0] == 7


def test_complete_search_and_manual_equivalence(study_dir):
    study = load_study(study_dir)
    baseline = calculate(study)
    progress = []
    found = search(study, baseline, 600, lambda done, total, errors: progress.append((done, total, errors)))
    assert found.checked == 3 and found.complete and found.errors == 0
    assert set(item["block_id"] for item in found.candidates) == {0, 1, 2}
    assert progress[-1] == (3, 3, 0)
    for item in found.candidates:
        assert item["reduction_places"] == calculate(study, [(item["block_id"], 600)], baseline).totals["reduction"]


def test_failures_are_not_global_optimum(study_dir):
    study = load_study(study_dir)
    baseline = calculate(study)
    def failing(study, schools, baseline):
        if schools[0][0] == 2:
            raise RuntimeError("solver error")
        return calculate(study, schools, baseline)
    found = search(study, baseline, evaluator=failing)
    assert not found.complete and found.errors == 1 and found.checked == 3
    assert found.candidates[-1]["status"] == "Ошибка"


def test_ties_and_stable_sort(study_dir):
    study = load_study(study_dir)
    baseline = calculate(study)
    def tied(study, schools, baseline):
        class Outcome:
            totals = {"deficit": 700, "reduction": 100, "reduction_pct": 12.5}
        return Outcome()
    found = search(study, baseline, evaluator=tied)
    assert found.winners == (0, 1, 2)
    assert [x["block_id"] for x in found.candidates] == [0, 1, 2]


@pytest.mark.parametrize("bad", [0, -2, 3.5, True, "600", None])
def test_capacity_validation(study_dir, bad):
    with pytest.raises((DataError, TypeError)):
        calculate(load_study(study_dir), [(1, bad)])


def test_missing_and_misaligned_inputs(study_dir):
    study_dir.joinpath("accessibility_matrix_drive.csv").unlink()
    with pytest.raises(DataError, match="Не найдены"):
        load_study(study_dir)
    pd.DataFrame([[0, 1, 2], [1, 0, 2], [2, 2, 0]], index=[0,1,2], columns=[0,1,99]).to_csv(study_dir / "accessibility_matrix_drive.csv")
    with pytest.raises(DataError, match="ID строк и столбцов"):
        load_study(study_dir)


def test_matrix_reorders_both_axes_by_id_without_symmetrizing(study_dir):
    path = study_dir / "accessibility_matrix_drive.csv"
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(int)
    frame.columns = frame.columns.astype(int)
    frame.loc[[2, 0, 1], [1, 2, 0]].to_csv(path)
    matrix = load_study(study_dir).matrix
    assert list(matrix.index) == list(matrix.columns) == [0, 1, 2]
    assert matrix.loc[0, 1] == 5 and matrix.loc[1, 0] == 7


def test_saved_indicator_geojson_is_accepted_when_raw_blocks_absent(study_dir):
    raw = study_dir / "blocks.geojson"
    raw.rename(study_dir / "blocks_indicators.geojson")
    study = load_study(study_dir)
    assert study.metadata["blocks_file"] == "blocks_indicators.geojson"
    assert calculate(study).totals["demand"] == 900


def test_missing_population_does_not_become_zero(study_dir):
    path = study_dir / "blocks.geojson"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["features"][0]["properties"]["population"] = None
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(DataError, match="Население"):
        load_study(study_dir)


def test_capacity_uses_documented_notebook_rounding(study_dir):
    path = study_dir / "blocks.geojson"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["features"][0]["properties"]["capacity_school"] = 100.8
    doc["features"][1]["properties"]["capacity_school"] = None
    path.write_text(json.dumps(doc), encoding="utf-8")
    study = load_study(study_dir)
    assert study.capacity.tolist() == [101, 0, 0]
    assert "NaN" in study.metadata["missing_capacity_policy"]


def test_saved_baseline_and_exports_are_read_only(study_dir):
    study = load_study(study_dir)
    base = calculate(study)
    path = study_dir / "provision_schools.csv"
    base.rows.to_csv(path)
    before = {p.name: p.read_bytes() for p in study_dir.iterdir()}
    assert "совпал" in verify_baseline(base, load_study(study_dir).saved_provision)
    result = calculate(study, [(1,600)], base)
    exported = csv_bytes(quarters_frame(result))
    assert exported.startswith(b"\xef\xbb\xbf") and "Дефицит, мест" in exported.decode("utf-8-sig")
    assert len(json.loads(geojson(study, result))["features"]) == 3
    assert before == {p.name: p.read_bytes() for p in study_dir.iterdir()}


def test_saved_baseline_mismatch_is_rejected(study_dir):
    study = load_study(study_dir)
    baseline = calculate(study)
    saved = baseline.rows.copy()
    saved.loc[0, "demand_within"] += 1
    saved.to_csv(study_dir / "provision_schools.csv")
    with pytest.raises(DataError, match="расходится"):
        verify_baseline(baseline, load_study(study_dir).saved_provision)


def test_zero_demand_has_undefined_fraction(study_dir):
    path = study_dir / "blocks.geojson"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["features"][2]["properties"]["population"] = 0
    path.write_text(json.dumps(doc), encoding="utf-8")
    result = calculate(load_study(study_dir))
    assert pd.isna(result.rows.loc[2, "provision_strong"])
    assert result.rows.loc[2, "deficit"] == 0


def test_browser_renders_and_map_table_share_results(study_dir):
    from app import make_app, map_data
    instance = make_app(study_dir)
    response = instance.server.test_client().get("/")
    assert response.status_code == 200
    assert b"dash" in response.data.lower()
    assert any(inp["id"] == "polygons" and inp["property"] == "clickData"
               for cb in instance.callback_map.values() for inp in cb["inputs"])
    click_callback = next(key for key, cb in instance.callback_map.items()
                          if any(inp["id"] == "polygons" and inp["property"] == "clickData" for inp in cb["inputs"]))
    clicked = instance.server.test_client().post("/_dash-update-component", json={
        "output": click_callback,
        "outputs": [{"id": "selected", "property": "data"}, {"id": "block", "property": "value"}],
        "inputs": [{"id": "polygons", "property": "clickData", "value": {"properties": {"block_id": 1}}}],
        "state": [], "changedPropIds": ["polygons.clickData"]})
    assert clicked.status_code == 200
    assert clicked.json["response"]["selected"]["data"] == 1
    assert clicked.json["response"]["block"]["value"] == 1
    assert instance.server.test_client().get("/assets/map.js").status_code == 200
    study = load_study(study_dir)
    base = calculate(study)
    scenario = calculate(study, [(1, 600)], base)
    features = map_data(study, base, scenario, "Сценарий", 1)["features"]
    assert {f["properties"]["block_id"] for f in features} == set(scenario.rows.index)
    assert features[0]["properties"]["map_color"] != map_data(study, base, scenario, "Исходное состояние", 1)["features"][0]["properties"]["map_color"]
