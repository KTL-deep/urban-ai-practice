"""One BlocksNet 1.0.0a9 calculation for baseline and every scenario."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_loader import DataError, Study


@dataclass(frozen=True)
class Result:
    rows: pd.DataFrame
    totals: dict
    schools: tuple
    links: pd.DataFrame


def validate_schools(schools, ids):
    validated = []
    for block_id, places in schools:
        if isinstance(block_id, bool) or not isinstance(block_id, int) or block_id not in ids:
            raise DataError(f"Неизвестный ID квартала: {block_id}")
        if isinstance(places, bool) or not isinstance(places, int) or places <= 0:
            raise DataError("Вместимость школы должна быть положительным целым числом")
        validated.append((block_id, places))
    return tuple(validated)


def calculate(study: Study, schools=(), baseline: Result | None = None) -> Result:
    from blocksnet.analysis.provision import competitive_provision
    from blocksnet.config import service_types_config
    import blocksnet

    if blocksnet.__version__ != "1.0.0a9":
        raise DataError(f"Нужен BlocksNet 1.0.0a9; установлен {blocksnet.__version__}")
    cfg = service_types_config["school"]
    if int(cfg["demand"]) != 120 or int(cfg["accessibility"]) != 15:
        raise DataError("Норматив или порог школы отличается от методики практики №2")
    placements = validate_schools(schools, set(study.population.index))
    capacity = study.capacity.copy(deep=True)
    for block_id, places in placements:
        capacity.loc[block_id] += places
    inputs = pd.DataFrame({"population": study.population.copy(deep=True), "capacity": capacity})
    calculated, links = competitive_provision(inputs, study.matrix,
                                                accessibility=cfg["accessibility"], demand=cfg["demand"])
    rows = calculated[["demand", "capacity", "demand_within", "demand_left",
                       "capacity_within", "provision_strong"]].copy()
    rows.index.name = "block_id"
    rows.insert(0, "population", study.population.loc[rows.index])
    rows["deficit"] = rows["demand"] - rows["demand_within"]
    if baseline is not None:
        rows["baseline_provision"] = baseline.rows["provision_strong"]
        rows["baseline_deficit"] = baseline.rows["deficit"]
        rows["provision_change"] = rows["provision_strong"] - rows["baseline_provision"]
        rows["deficit_reduction"] = rows["baseline_deficit"] - rows["deficit"]
    if (rows[["demand", "capacity", "demand_within", "deficit"]] < -1e-8).any().any():
        raise DataError("Некорректные отрицательные итоги распределения")
    if (rows.demand_within > rows.demand + 1e-8).any() or rows.demand_within.sum() > rows.capacity.sum() + 1e-8:
        raise DataError("Спрос обслужен сверх потребности или вместимости")
    self_supply = pd.concat([rows.demand, rows.capacity], axis=1).min(axis=1)
    if not links.empty:
        flows = links.reset_index()
        if (flows.value < 0).any() or not flows.source.isin(rows.index).all() or not flows.target.isin(rows.index).all():
            raise DataError("Некорректные связи распределения")
        inbound = flows.groupby("target").value.sum().reindex(rows.index, fill_value=0)
        outbound = flows.groupby("source").value.sum().reindex(rows.index, fill_value=0)
        if (inbound + self_supply > rows.capacity + 1e-8).any() or (outbound + self_supply > rows.demand + 1e-8).any():
            raise DataError("Места распределены повторно с учётом обслуживания собственного квартала")
        if not np.allclose(rows.demand_within.to_numpy(dtype=float),
                           (outbound + self_supply).to_numpy(dtype=float), atol=1e-7):
            raise DataError("Обслуженный спрос не совпадает с распределёнными связями")
        for link in flows.itertuples():
            if study.matrix.loc[link.source, link.target] > cfg["accessibility"]:
                raise DataError("Место распределено за пределами доступности")
    elif not np.allclose(rows.demand_within.to_numpy(dtype=float), self_supply.to_numpy(dtype=float), atol=1e-7):
        raise DataError("Нет связей, но обслуженный спрос превышает собственные места")
    # No demand: the original BlocksNet fraction is NaN, not an invented 100%.
    total_demand = int(rows.demand.sum())
    deficit = int(round(rows.deficit.sum()))
    totals = {"population": float(rows.population.sum()), "demand": total_demand,
              "capacity": int(rows.capacity.sum()), "served": int(round(rows.demand_within.sum())),
              "deficit": deficit, "provision": float(rows.demand_within.sum() / total_demand) if total_demand else None,
              "added_capacity": sum(places for _, places in placements)}
    if baseline is not None:
        reduction = baseline.totals["deficit"] - deficit
        totals.update({"baseline_deficit": baseline.totals["deficit"], "reduction": reduction,
                       "reduction_pct": 100 * reduction / baseline.totals["deficit"] if baseline.totals["deficit"] else None})
    return Result(rows, totals, placements, links)


def verify_baseline(result: Result, saved: pd.DataFrame | None):
    if saved is None:
        return "Сохранённый provision_schools.csv отсутствует: независимая сверка недоступна."
    for column in ("demand", "capacity", "demand_within", "demand_left", "provision_strong"):
        if column not in saved:
            raise DataError(f"В сохранённом расчёте нет столбца {column}")
        actual = result.rows[column].to_numpy(dtype=float)
        expected = pd.to_numeric(saved[column], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(actual, expected, atol=1e-7, rtol=1e-7, equal_nan=True):
            raise DataError(f"Базовый расчёт расходится с практикой №2: {column}")
    return "Базовый расчёт совпал с сохранённым provision_schools.csv."
