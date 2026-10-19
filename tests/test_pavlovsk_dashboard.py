"""Synthetic fixtures keep tests independent of the notebook and generated datasets."""

import csv
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "build_pavlovsk_dashboard.py"
spec = importlib.util.spec_from_file_location("pavlovsk_dashboard", SCRIPT)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


@pytest.fixture
def data_dir(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    ids = ["0", "1"]
    features = [
        {"type": "Feature", "properties": {"block_id": int(identifier), "fsi": 0.5 + index,
            "gsi": .2, "mxi": .4, "area_accessibility": 8, "provision_strong": index,
            "population": 10, "travel_time_median_min": 6, "strelka_morphotype": "</script><img src=x onerror=alert(1)>",
            "spacematrix_morphotype": "low-rise"},
         "geometry": {"type": "Polygon", "coordinates": [[[30 + index, 59], [30.1 + index, 59],
                                                    [30.1 + index, 59.1], [30 + index, 59]]]}}
        for index, identifier in enumerate(ids)
    ]
    (data / "blocks_indicators.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    (data / "boundary.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [features[0]]}), encoding="utf-8")
    (data / "study_info.json").write_text(json.dumps({"territory": "<unsafe>&"}), encoding="utf-8")
    with (data / "blocks_indicators.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["", "fsi", "gsi", "mxi", "area_accessibility", "provision_strong", "population", "strelka_morphotype"])
        writer.writeheader()
        for index, identifier in enumerate(ids):
            writer.writerow({"": identifier, "fsi": "nan" if index else 0.5, "gsi": .2, "mxi": .4,
                             "area_accessibility": 8, "provision_strong": index, "population": 10,
                             "strelka_morphotype": "</script><img src=x onerror=alert(1)>"})
    with (data / "provision_schools.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["", *dashboard.SCHOOL_COLUMNS])
        writer.writeheader()
        for index, identifier in enumerate(ids):
            writer.writerow({"": identifier, "demand": 0 if index else 10, "capacity": 5,
                             "demand_left": 0 if index else 5, "demand_within": 0 if index else 5,
                             "demand_without": 0, "capacity_left": 0, "capacity_within": 0,
                             "capacity_without": 0, "provision_strong": 0 if index else .5,
                             "provision_weak": 0 if index else .5})
    return data


def test_generator_embeds_maps_charts_tables_and_escapes_inputs(data_dir, tmp_path):
    output = tmp_path / "report.html"
    assert dashboard.generate(data_dir, output) == output
    page = output.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    assert "id='blocks-table'" in page and "id='schools-table'" in page
    assert "data:image/png;base64," in page and "srcdoc='" in page
    assert "Показатель карты" in page and "Морфотипы Spacematrix" in page
    assert "50.0%" in page and "автомобиль" in page
    assert "&lt;unsafe&gt;&amp;" in page
    assert "&lt;/script&gt;&lt;img" in page
    assert "</script><img src=x onerror" not in page
    without_images = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", page)
    assert "NaN" not in without_images and "Infinity" not in without_images
    assert page.count("id='blocks-search'") == 1
    assert page.count("id='schools-search'") == 1


def test_no_overwrite_without_flag_and_cli(data_dir, tmp_path, capsys):
    output = tmp_path / "existing.html"
    output.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        dashboard.generate(data_dir, output)
    assert output.read_text(encoding="utf-8") == "original"
    dashboard.main(["--data-dir", str(data_dir), "--output", str(output), "--overwrite"])
    assert "Отчёт сохранён" in capsys.readouterr().out
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_missing_or_misaligned_inputs_fail_without_output(data_dir, tmp_path):
    output = tmp_path / "report.html"
    (data_dir / "boundary.geojson").unlink()
    with pytest.raises(ValueError, match="Отсутствует"):
        dashboard.generate(data_dir, output)
    assert not output.exists()
    (data_dir / "boundary.geojson").write_text('{"type":"FeatureCollection","features":[{"type":"Feature"}]}', encoding="utf-8")
    path = data_dir / "provision_schools.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("1,0,5", "999,0,5"), encoding="utf-8")
    with pytest.raises(ValueError, match="ID квартала"):
        dashboard.generate(data_dir, output)
    assert not output.exists()


def test_import_has_no_side_effects(tmp_path):
    assert callable(dashboard.main)
    assert not (tmp_path / "pavlovsk_map.html").exists()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is optional for JavaScript syntax checks")
def test_map_javascript_parses(data_dir):
    features, boundary, *_ = dashboard.load_data(data_dir)
    page = dashboard.build_map(features, boundary)
    assert "basemaps.cartocdn.com" in page
    assert "tile.openstreetmap.org" not in page
    assert "Без подложки" in page
    assert "basemap.on('tileerror'" in page
    assert page.index("var map_") < page.index("const data =")
    assert page.index("var geo_json_") < page.index("const data =")
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", page, re.DOTALL)
    assert scripts
    for script in scripts:
        if script.strip():
            result = subprocess.run([shutil.which("node"), "--check"], input=script, text=True,
                                    encoding="utf-8", capture_output=True, check=False)
            assert result.returncode == 0, result.stderr
