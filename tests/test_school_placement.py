"""Known small artificial territory; never used by the Pavlovsk application."""

import json
import base64
from pathlib import Path
import sys
import time

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


def test_default_data_directory_falls_back_to_packaged_copy(monkeypatch, tmp_path):
    import data_loader
    package_dir = tmp_path / "school-placement"
    snapshot = package_dir / "input"
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(data_loader, "__file__", str(package_dir / "data_loader.py"))
    assert data_loader.default_directory() == snapshot


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
    instance = make_app(study_dir, study_dir.parent / "results")
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


def test_reproduction_uses_only_original_notebook_code():
    from reproduce_study import CELLS
    notebook = json.loads((ROOT.parent / "examples" / "pavlovsk_study.ipynb").read_text(encoding="utf-8"))
    for stage, indices in CELLS.items():
        if stage == "verify":
            assert not indices
            continue
        assert indices
        assert all(notebook["cells"][i]["cell_type"] == "code" for i in indices)
    assert 16 in CELLS["all"] and 21 in CELLS["all"]


def test_ui_add_save_reset_and_download_on_test_territory(study_dir, tmp_path):
    """Exercise real Dash callbacks without writing into the Pavlovsk results."""
    from app import make_app
    instance = make_app(study_dir, tmp_path / "results")
    client = instance.server.test_client()

    def call(trigger, outputs, inputs, state):
        callback = next(key for key, value in instance.callback_map.items()
                        if any(item["id"] == trigger.split(".")[0] and item["property"] == trigger.split(".")[1]
                               for item in value["inputs"])
                        and outputs[0][0] in key)
        output_specs = [{"id": i, "property": p} for i, p in outputs]
        request = {"output": callback, "outputs": output_specs if len(outputs) > 1 else output_specs[0],
                   "inputs": [{"id": i, "property": p, "value": v} for i, p, v in inputs],
                   "state": [{"id": i, "property": p, "value": v} for i, p, v in state],
                   "changedPropIds": [trigger]}
        response = client.post("/_dash-update-component", json=request)
        assert response.status_code == 200, response.get_data(as_text=True)[:500]
        return response.get_json()["response"]

    remove_pattern = '{"index":["ALL"],"type":"remove"}'
    inputs = [("add", "n_clicks", 1), ("reset", "n_clicks", None), (remove_pattern, "n_clicks", [])]
    state = [("draft", "data", []), ("block", "value", 1), ("places", "value", -1)]
    bad = call("add.n_clicks", [("draft", "data"), ("status", "children")], inputs, state)
    assert "положительную" in bad["status"]["children"] and "draft" not in bad
    state[-1] = ("places", "value", 600)
    added = call("add.n_clicks", [("draft", "data"), ("status", "children")], inputs, state)
    draft = added["draft"]["data"]
    assert draft == [{"id": 1, "places": 600}]
    saved = call("run.n_clicks", [("scenarios", "data"), ("view", "value"), ("map-mode", "value"),
                                  ("status", "children")], [("run", "n_clicks", 1)],
                 [("scenario-name", "value", "А"), ("draft", "data", draft),
                  ("reason", "value", "Квартал выбран вручную до поиска"), ("scenarios", "data", {})])
    assert saved["view"]["value"] == "А" and saved["scenarios"]["data"]["А"]["schools"] == [[1, 600]]
    assert (tmp_path / "results" / "scenarios.json").is_file()

    inputs = [("export-quarters", "n_clicks", 1), ("export-ranking", "n_clicks", None),
              ("export-comparison", "n_clicks", None), ("export-geojson", "n_clicks", None)]
    exported = call("export-quarters.n_clicks", [("download", "data")], inputs,
                    [("view", "value", "А"), ("scenarios", "data", saved["scenarios"]["data"]),
                     ("ranking", "data", [])])
    contents = base64.b64decode(exported["download"]["data"]["content"])
    assert contents.startswith(b"\xef\xbb\xbf") and "Дефицит, мест" in contents.decode("utf-8-sig")

    b_schools = [{"id": 0, "places": 300}, {"id": 1, "places": 300}]
    manual_b = call("run.n_clicks", [("scenarios", "data"), ("view", "value"),
                                     ("map-mode", "value"), ("status", "children")],
                    [("run", "n_clicks", 2)],
                    [("scenario-name", "value", "Б"), ("draft", "data", b_schools),
                     ("reason", "value", "Два разных квартала"),
                     ("scenarios", "data", saved["scenarios"]["data"])])
    chosen = manual_b["scenarios"]["data"]
    started = call("automatic.n_clicks", [("job", "data"), ("timer", "disabled"),
                                           ("progress", "children")],
                   [("automatic", "n_clicks", 1)],
                   [("auto-places", "value", 600), ("scenarios", "data", chosen)])
    token = started["job"]["data"]
    assert token and started["timer"]["disabled"] is False
    poll_outputs = [("progress", "children"), ("timer", "disabled"), ("ranking", "data"),
                    ("ranking", "columns"), ("ranking-status", "children"),
                    ("scenarios", "data"), ("view", "value"), ("map-mode", "value"),
                    ("selected", "data")]
    for interval in range(40):
        finished = call("timer.n_intervals", poll_outputs, [("timer", "n_intervals", interval)],
                        [("job", "data", token), ("scenarios", "data", chosen)])
        if "ranking" in finished:
            break
        time.sleep(0.05)
    assert "ranking" in finished and len(finished["ranking"]["data"]) == 3
    assert "В" in finished["scenarios"]["data"]
    assert finished["timer"]["disabled"] is True
    assert "3" in finished["progress"]["children"]

    for button, fragment in (("export-ranking", "block_id"),
                             ("export-comparison", "Сценарий"),
                             ("export-geojson", "FeatureCollection")):
        export_inputs = [(i, "n_clicks", 1 if i == button else None) for i in
                         ("export-quarters", "export-ranking", "export-comparison", "export-geojson")]
        download = call(f"{button}.n_clicks", [("download", "data")], export_inputs,
                        [("view", "value", "В"), ("scenarios", "data", finished["scenarios"]["data"]),
                         ("ranking", "data", finished["ranking"]["data"])])
        payload = base64.b64decode(download["download"]["data"]["content"])
        assert fragment in payload.decode("utf-8-sig")

    cleared = call("reset.n_clicks", [("view", "value"), ("draft", "data"), ("map-mode", "value"),
                                      ("selected", "data"), ("block", "value"), ("reason", "value"),
                                      ("places", "value")],
                   [("reset", "n_clicks", 1)], [])
    assert cleared["draft"]["data"] == [] and cleared["view"]["value"] == "Исходное состояние"
    assert cleared["selected"]["data"] is None and cleared["block"]["value"] is None
    assert cleared["reason"]["value"] == "" and cleared["places"]["value"] == 600
    assert (tmp_path / "results" / "scenarios.json").is_file()  # reset never discards saved scenarios


def test_ui_incomplete_search_never_replaces_previous_winner(study_dir, tmp_path, monkeypatch):
    import app as app_module
    from search import SearchResult

    def failed_search(study, baseline, places, progress):
        candidates = tuple({"block_id": i, "new_capacity_places": places,
                            "baseline_deficit_places": baseline.totals["deficit"],
                            "remaining_deficit_places": None if i == 1 else 700,
                            "reduction_places": None if i == 1 else 100,
                            "reduction_pct": None if i == 1 else 12.5,
                            "status": "Ошибка" if i == 1 else "Рассчитан",
                            "error": "solver failure" if i == 1 else ""} for i in range(3))
        for done in range(1, 4):
            progress(done, 3, 1 if done >= 2 else 0)
        return SearchResult(candidates, (0, 2), False, 3, 1)

    monkeypatch.setattr(app_module, "search", failed_search)
    app = app_module.make_app(study_dir, tmp_path / "out")
    client = app.server.test_client()
    scenarios = {name: {"schools": [[i, 600]], "reason": "Ранее рассчитано"}
                 for name, i in (("А", 0), ("Б", 1), ("В", 2))}
    start = next(k for k, v in app.callback_map.items()
                 if any(x["id"] == "automatic" for x in v["inputs"]))
    started = client.post("/_dash-update-component", json={
        "output": start, "outputs": [{"id": i, "property": p} for i, p in
                                   [("job", "data"), ("timer", "disabled"), ("progress", "children")]],
        "inputs": [{"id": "automatic", "property": "n_clicks", "value": 1}],
        "state": [{"id": "auto-places", "property": "value", "value": 600},
                  {"id": "scenarios", "property": "data", "value": scenarios}],
        "changedPropIds": ["automatic.n_clicks"]})
    assert started.status_code == 200
    token = started.get_json()["response"]["job"]["data"]
    poll = next(k for k, v in app.callback_map.items() if any(x["id"] == "timer" for x in v["inputs"]))
    fields = [("progress", "children"), ("timer", "disabled"), ("ranking", "data"),
              ("ranking", "columns"), ("ranking-status", "children"), ("scenarios", "data"),
              ("view", "value"), ("map-mode", "value"), ("selected", "data")]
    for interval in range(30):
        finished = client.post("/_dash-update-component", json={
            "output": poll, "outputs": [{"id": i, "property": p} for i, p in fields],
            "inputs": [{"id": "timer", "property": "n_intervals", "value": interval}],
            "state": [{"id": "job", "property": "data", "value": token},
                      {"id": "scenarios", "property": "data", "value": scenarios}],
            "changedPropIds": ["timer.n_intervals"]})
        assert finished.status_code == 200
        response = finished.get_json()["response"]
        if "ranking" in response:
            break
        time.sleep(0.02)
    assert "ranking" in response and response["ranking"]["data"][1]["status"] == "Ошибка"
    assert "НЕ установлен" in response["ranking-status"]["children"]
    assert response["scenarios"]["data"]["В"]["schools"] == [[2, 600]]
    assert (tmp_path / "out" / "automatic_ranking_incomplete.csv").is_file()
    assert not (tmp_path / "out" / "automatic_ranking.csv").exists()
