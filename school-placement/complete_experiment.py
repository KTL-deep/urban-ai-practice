"""Reproducible Pavlovsk A/B experiment and exhaustive single-school search.

Manual choices MUST already exist in results/manual_choices.json before this
script runs. No simulated or newly downloaded data are accepted.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from data_loader import DataError, load_study
from export import compare_frame, csv_bytes, geojson, quarters_frame
from model import calculate, validate_schools, verify_baseline
from search import search


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def choices_from_file(study):
    path = RESULTS / "manual_choices.json"
    if not path.is_file():
        raise DataError("Сначала зафиксируйте ручной выбор в results/manual_choices.json ДО автопоиска")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not document.get("recorded_before_automatic_search"):
        raise DataError("Ручной выбор не отмечен как сделанный до автоматического поиска")
    scenarios = {}
    for label, expected in (("А", (1, 600)), ("Б", (2, 300))):
        item = document[label]
        schools = validate_schools([tuple(pair) for pair in item["schools"]], set(study.population.index))
        if len(schools) != expected[0] or any(capacity != expected[1] for _, capacity in schools):
            raise DataError(f"Сценарий {label}: неверное число школ или вместимость")
        if label == "Б" and schools[0][0] == schools[1][0]:
            raise DataError("Сценарий Б требует двух разных кварталов")
        reason = item["reason"].strip()
        if not reason:
            raise DataError(f"Обоснование сценария {label} не заполнено")
        scenarios[label] = {"schools": [list(pair) for pair in schools], "reason": reason}
    return scenarios, hashlib.sha256(path.read_bytes()).hexdigest()


def write_scenarios(study, baseline, scenarios, outcomes):
    payload = {"source_directory": str(study.directory.resolve()), "scenarios": scenarios}
    (RESULTS / "scenarios.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison = compare_frame({label: (outcomes[label], data["reason"]) for label, data in scenarios.items()}, baseline)
    (RESULTS / "scenario_comparison.csv").write_bytes(csv_bytes(comparison))
    for label, result in outcomes.items():
        (RESULTS / f"scenario_{label}.csv").write_bytes(csv_bytes(quarters_frame(result)))
        (RESULTS / f"scenario_{label}.geojson").write_bytes(geojson(study, result))


def main():
    study = load_study()
    baseline = calculate(study)
    print(verify_baseline(baseline, study.saved_provision), flush=True)
    scenarios, manual_hash = choices_from_file(study)
    if baseline.totals["deficit"] != 80 or len(study.population) != 188:
        raise DataError("Ручные обоснования относятся к другому исходному набору Павловска")
    existing = RESULTS / "scenarios.json"
    if existing.exists():
        prior = json.loads(existing.read_text(encoding="utf-8"))
        if set(prior.get("scenarios", {})) - {"А", "Б", "В"}:
            raise DataError("Сохранены дополнительные пользовательские сценарии; сохраните их отдельно перед повторным экспериментом")
        if prior.get("source_directory") != str(study.directory.resolve()) or any(
            name in prior.get("scenarios", {}) and prior["scenarios"][name]["schools"] != scenarios[name]["schools"]
            for name in scenarios
        ):
            raise DataError("Существующие пользовательские сценарии отличаются: результаты не перезаписаны")

    outcomes = {}
    for label in ("А", "Б"):
        outcomes[label] = calculate(study, scenarios[label]["schools"], baseline)
        print(f"{label}: {outcomes[label].totals}", flush=True)
    write_scenarios(study, baseline, scenarios, outcomes)

    def progress(done, total, errors):
        if done % 10 == 0 or done == total:
            print(f"Перебор: {done}/{total}, ошибок: {errors}", flush=True)

    ranking = search(study, baseline, 600, progress=progress)
    (RESULTS / "automatic_ranking.csv").write_bytes(csv_bytes(pd.DataFrame(ranking.candidates)))
    if not ranking.complete:
        raise DataError(f"Неполный перебор: {ranking.errors} ошибок из {ranking.checked}; глобальный оптимум не установлен")

    winner = ranking.winners[0]
    scenarios["В"] = {"schools": [[winner, 600]],
                      "reason": "Полный перебор всех 188 кварталов после фиксации ручных вариантов. "
                                + "Равноценные лучшие ID: " + ", ".join(map(str, ranking.winners))}
    outcomes["В"] = calculate(study, scenarios["В"]["schools"], baseline)
    if outcomes["В"].totals["deficit"] != ranking.candidates[0]["remaining_deficit_places"]:
        raise DataError("Ручное воспроизведение лучшего автоматического варианта не совпало с рейтингом")
    write_scenarios(study, baseline, scenarios, outcomes)

    summary = {"source_directory": str(study.directory.resolve()), "manual_choices_sha256": manual_hash,
               "blocks_checked": ranking.checked, "errors": ranking.errors,
               "complete": ranking.complete, "winners": ranking.winners,
               "baseline": baseline.totals,
               "scenarios": {label: {"schools": scenarios[label]["schools"], "totals": result.totals,
                                     "improved_blocks": int((result.rows.deficit_reduction > 0).sum()),
                                     "remaining_deficit_blocks": int((result.rows.deficit > 0).sum()),
                                     "most_improved": [[int(i), float(v)] for i, v in result.rows.deficit_reduction.nlargest(10).items() if v > 0],
                                     "remaining_deficits": [[int(i), float(v)] for i, v in result.rows.deficit.items() if v > 0]}
                             for label, result in outcomes.items()},
               "manual_A_matches_any_best": scenarios["А"]["schools"][0][0] in ranking.winners}
    (RESULTS / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "automatic_search.json").write_text(json.dumps({
        "source_directory": str(study.directory.resolve()), "capacity_places": 600,
        "blocks_checked": ranking.checked, "errors": ranking.errors,
        "complete": ranking.complete, "winners": ranking.winners,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Полный поиск: {ranking.checked} кварталов; равноценные лучшие: {ranking.winners}", flush=True)
    print(f"В: {outcomes['В'].totals}", flush=True)
    print(f"А входит в оптимальные: {summary['manual_A_matches_any_best']}", flush=True)


if __name__ == "__main__":
    main()
