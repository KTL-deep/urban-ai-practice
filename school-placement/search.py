"""Exhaustive, independently evaluated single-school placement."""

from dataclasses import dataclass
import math

from model import calculate, validate_schools


@dataclass(frozen=True)
class SearchResult:
    candidates: tuple
    winners: tuple
    complete: bool
    checked: int
    errors: int


def search(study, baseline, places=600, progress=None, evaluator=calculate):
    ids = list(study.population.index)
    if not ids:
        raise ValueError("Нет кварталов для перебора")
    validate_schools([(int(ids[0]), places)], set(ids))
    records = []
    for position, block_id in enumerate(ids, 1):
        record = {"block_id": int(block_id), "new_capacity_places": places,
                  "baseline_deficit_places": baseline.totals["deficit"]}
        try:
            outcome = evaluator(study, [(int(block_id), places)], baseline)
            record.update({"remaining_deficit_places": outcome.totals["deficit"],
                           "reduction_places": outcome.totals["reduction"],
                           "reduction_pct": outcome.totals["reduction_pct"], "status": "Рассчитан", "error": ""})
        except Exception as exc:
            record.update({"remaining_deficit_places": None, "reduction_places": None,
                           "reduction_pct": None, "status": "Ошибка", "error": f"{type(exc).__name__}: {exc}"})
        records.append(record)
        if progress is not None:
            progress(position, len(ids), sum(r["status"] == "Ошибка" for r in records))
    ranked = sorted(records, key=lambda r: (r["status"] != "Рассчитан",
                     -r["reduction_places"] if r["status"] == "Рассчитан" else 0, r["block_id"]))
    successes = [r for r in ranked if r["status"] == "Рассчитан"]
    winners = tuple(r["block_id"] for r in successes if math.isclose(r["reduction_places"],
                    successes[0]["reduction_places"], abs_tol=1e-7, rel_tol=1e-9)) if successes else ()
    errors = len(records) - len(successes)
    return SearchResult(tuple(ranked), winners, errors == 0, len(records), errors)
