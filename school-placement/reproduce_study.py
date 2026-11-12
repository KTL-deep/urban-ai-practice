"""Execute existing study notebook cells without editing the notebook.

Run from the repository root. This intentionally uses the real data source and
the exact preprocessing, matrix and provision code from practice 2. The notebook
itself is never rewritten; its outputs go to pavlovsk_study_data/.
"""

import argparse
import json
from pathlib import Path


CELLS = {
    "verify": (),
    "boundary": (2, 4, 6),
    "core": (2, 4, 6, 8, 10, 14, 16, 21),
    "all": (2, 4, 6, 8, 10, 14, 16, 21, 18, 19, 20, 25, 27),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=CELLS, default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        parser.error(f"Запускайте из корня репозитория: {root}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda: None  # notebook plots are not needed for saved data
    notebook = json.loads((root / "examples" / "pavlovsk_study.ipynb").read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    for index in CELLS[args.stage]:
        cell = notebook["cells"][index]
        if cell["cell_type"] != "code":
            raise ValueError(f"Ячейка {index} больше не содержит код: проверьте ноутбук")
        print(f"\n[Практика №2: ячейка {index}]", flush=True)
        exec(compile("".join(cell["source"]), f"pavlovsk_study.ipynb:{index}", "exec"), namespace)
    if args.stage != "boundary":
        from data_loader import load_study
        from model import calculate, verify_baseline

        study = load_study(root / "pavlovsk_study_data")
        baseline = calculate(study)
        verification = verify_baseline(baseline, study.saved_provision)
        from export import csv_bytes, quarters_frame
        output_dir = root / "school-placement" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "baseline_summary.json").write_text(json.dumps({
            "territory": study.metadata["territory"], "source_directory": str(study.directory.resolve()),
            "blocks": len(study.population), "totals": baseline.totals,
            "method": "BlocksNet 1.0.0a9 competitive_provision", "travel_mode": "drive",
            "travel_time_limit_minutes": 15, "demand_places_per_1000_residents": 120,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "baseline_quarters.csv").write_bytes(csv_bytes(quarters_frame(baseline)))
        print(f"Кварталов: {len(study.population)}; спрос: {baseline.totals['demand']}; "
              f"обслужено: {baseline.totals['served']}; дефицит: {baseline.totals['deficit']}", flush=True)
        print(verification, flush=True)


if __name__ == "__main__":
    main()
