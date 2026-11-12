"""Integration checks against locally saved Pavlovsk data, if present."""

import json
import hashlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "school-placement"))


@pytest.mark.skipif(not (ROOT / "pavlovsk_study_data" / "provision_schools.csv").is_file(),
                    reason="Сохранённые данные Павловска отсутствуют")
def test_real_baseline_and_browser_callbacks():
    from app import make_app
    app = make_app(ROOT / "pavlovsk_study_data")
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    callback = next(key for key in app.callback_map if "map-layers.children" in key)
    fields = [("map-layers", "children"), ("legend", "children"), ("quarter-detail", "children"),
              ("totals", "children"), ("quarters", "data"), ("quarters", "columns"),
              ("comparison", "data"), ("comparison", "columns")]
    response = client.post("/_dash-update-component", json={
        "output": callback,
        "outputs": [{"id": name, "property": prop} for name, prop in fields],
        "inputs": [{"id": "view", "property": "value", "value": "Исходное состояние"},
                   {"id": "map-mode", "property": "value", "value": "Исходное состояние"},
                   {"id": "selected", "property": "data", "value": None},
                   {"id": "scenarios", "property": "data", "value": {}}],
        "state": [], "changedPropIds": ["view.value"]})
    assert response.status_code == 200, response.get_data(as_text=True)[:500]
    rows = response.json["response"]["quarters"]["data"]
    assert len(rows) == 188
    assert sum(row["Дефицит, мест"] for row in rows) == 80


@pytest.mark.skipif(not (ROOT / "school-placement" / "results" / "experiment_summary.json").is_file(),
                    reason="Учебный эксперимент ещё не выполнялся")
def test_saved_scenarios_and_complete_ranking_load_into_browser():
    from app import make_app
    summary = json.loads((ROOT / "school-placement" / "results" / "experiment_summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] and summary["errors"] == 0 and summary["blocks_checked"] == 188
    manual_path = ROOT / "school-placement" / "results" / "manual_choices.json"
    assert summary["manual_choices_sha256"] == hashlib.sha256(manual_path.read_bytes()).hexdigest()
    assert {label: item["totals"]["deficit"] for label, item in summary["scenarios"].items()} == {
        "А": 32, "Б": 23, "В": 22}
    app = make_app(ROOT / "pavlovsk_study_data")
    layout = app.server.test_client().get("/_dash-layout").get_json()
    def find(component, identifier):
        if isinstance(component, dict):
            if component.get("props", {}).get("id") == identifier:
                return component["props"]
            for child in component.get("props", {}).get("children", []) if isinstance(component.get("props", {}).get("children", []), list) else []:
                found = find(child, identifier)
                if found is not None:
                    return found
        return None
    scenarios = find(layout, "scenarios")["data"]
    ranking = find(layout, "ranking")["data"]
    assert set(("А", "Б", "В")).issubset(scenarios)
    assert len(ranking) == 188 and {row["block_id"] for row in ranking} == set(range(188))
    assert scenarios["В"]["schools"] == [[163, 600]]
    assert ranking[0]["block_id"] == 163 and ranking[0]["reduction_places"] == 58
    callback = next(key for key in app.callback_map if "map-layers.children" in key)
    fields = [("map-layers", "children"), ("legend", "children"), ("quarter-detail", "children"),
              ("totals", "children"), ("quarters", "data"), ("quarters", "columns"),
              ("comparison", "data"), ("comparison", "columns")]
    response = app.server.test_client().post("/_dash-update-component", json={
        "output": callback, "outputs": [{"id": name, "property": prop} for name, prop in fields],
        "inputs": [{"id": "view", "property": "value", "value": "В"},
                   {"id": "map-mode", "property": "value", "value": "Сценарий"},
                   {"id": "selected", "property": "data", "value": 163},
                   {"id": "scenarios", "property": "data", "value": scenarios}],
        "state": [], "changedPropIds": ["view.value"]})
    assert response.status_code == 200
    view = response.get_json()["response"]
    assert "agriculture" in view["quarter-detail"]["children"]
    assert sum(row["Дефицит, мест"] for row in view["quarters"]["data"]) == 22
