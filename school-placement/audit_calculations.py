"""Independent re-execution checks; never rewrites the submitted experiment."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from loguru import logger

from data_loader import load_study
from model import calculate, verify_baseline
from search import search


def main():
    logger.remove()
    root = Path(__file__).resolve().parent
    study = load_study()
    files = [path for path in study.directory.iterdir() if path.suffix in (".geojson", ".csv", ".json")]
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    baseline = calculate(study)
    verified = verify_baseline(baseline, study.saved_provision)
    population = study.population.copy(deep=True)
    capacity = study.capacity.copy(deep=True)
    matrix = study.matrix.copy(deep=True)
    outcomes = {"А": calculate(study, [(102, 600)], baseline),
                "Б": calculate(study, [(102, 300), (73, 300)], baseline),
                "В": calculate(study, [(163, 600)], baseline)}
    for name, result in outcomes.items():
        assert result.rows.population.equals(population)
        assert result.totals["capacity"] == baseline.totals["capacity"] + 600
        saved = pd.read_csv(root / "results" / f"scenario_{name}.csv", sep=";", encoding="utf-8-sig")
        assert dict(zip(saved["ID квартала"], saved["Дефицит, мест"])) == result.rows.deficit.to_dict()
    pd.testing.assert_frame_equal(calculate(study).rows, baseline.rows)
    pd.testing.assert_series_equal(population, study.population)
    pd.testing.assert_series_equal(capacity, study.capacity)
    pd.testing.assert_frame_equal(matrix, study.matrix)
    start = time.perf_counter()
    ranking = search(study, baseline, 600)
    elapsed = time.perf_counter() - start
    saved_ranking = pd.read_csv(root / "results" / "automatic_ranking.csv", sep=";", encoding="utf-8-sig")
    assert len(ranking.candidates) == len(study.population) and ranking.complete
    for actual, saved in zip(ranking.candidates, saved_ranking.to_dict("records"), strict=True):
        for field in ("block_id", "reduction_places", "remaining_deficit_places", "status"):
            assert actual[field] == saved[field], (field, actual, saved)
    assert ranking.winners == (163,)
    assert outcomes["В"].totals["deficit"] == ranking.candidates[0]["remaining_deficit_places"]
    assert hashes == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    report = {"baseline_verification": verified, "blocks": len(population), "matrix_shape": list(matrix.shape),
              "asymmetric_unordered_pairs": int(np.triu(~np.isclose(matrix.to_numpy(), matrix.to_numpy().T), 1).sum()),
              "source_files_unchanged": True, "input_sha256": hashes,
              "baseline": baseline.totals, "scenarios": {name: r.totals for name, r in outcomes.items()},
              "search_checked": ranking.checked, "search_errors": ranking.errors,
              "search_complete": ranking.complete, "search_winners": ranking.winners,
              "saved_ranking_matches_reexecution": True, "search_elapsed_seconds": elapsed,
              "manual_best_matches_search": True}
    path = root / "results" / "audit_checks.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "input_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
