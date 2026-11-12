"""Excel-friendly exports; only scenario results, never source data."""

import json

import pandas as pd


COLUMNS = {"block_id": "ID квартала", "population": "Население, чел.", "demand": "Потребность, мест",
           "capacity": "Вместимость школ, мест", "demand_within": "Обслужено за 15 мин, мест",
           "deficit": "Дефицит, мест", "provision_strong": "Обеспеченность, доля",
           "baseline_provision": "Исходная обеспеченность, доля", "baseline_deficit": "Исходный дефицит, мест",
           "provision_change": "Изменение обеспеченности, доля", "deficit_reduction": "Сокращение дефицита, мест"}


def csv_bytes(frame):
    return frame.to_csv(index=False, encoding="utf-8-sig", sep=";").encode("utf-8-sig")


def quarters_frame(result):
    columns = [column for column in COLUMNS if column in result.rows.reset_index().columns]
    return result.rows.reset_index()[columns].rename(columns=COLUMNS)


def compare_frame(scenarios, baseline):
    rows = [{"Сценарий": "Исходное состояние", "Школы (ID: мест)": "—", "Добавлено мест": 0,
             "Вместимость сети, мест": baseline.totals["capacity"],
             "Исходный дефицит, мест": baseline.totals["deficit"],
             "Остаточный дефицит, мест": baseline.totals["deficit"], "Сокращение, мест": 0,
             "Обоснование": "—"}]
    for name, (result, reason) in scenarios.items():
        rows.append({"Сценарий": name, "Школы (ID: мест)": ", ".join(f"{i}: {c}" for i, c in result.schools),
                     "Добавлено мест": result.totals["added_capacity"],
                     "Вместимость сети, мест": result.totals["capacity"],
                     "Исходный дефицит, мест": baseline.totals["deficit"],
                     "Остаточный дефицит, мест": result.totals["deficit"],
                     "Сокращение, мест": result.totals["reduction"], "Обоснование": reason})
    frame = pd.DataFrame(rows)
    frame["Транспорт"] = "автомобиль"
    frame["Порог доступности, мин"] = 15
    frame["Норматив, мест на 1000 жителей"] = 120
    frame["Метод"] = "BlocksNet 1.0.0a9 competitive_provision"
    return frame


def geojson(study, result):
    geo = json.loads(study.blocks.to_json())
    geo["metadata"] = {"study": study.metadata, "schools": result.schools, "totals": result.totals,
                       "provision_method": "BlocksNet 1.0.0a9 competitive_provision",
                       "travel_time_limit_minutes": 15, "demand_per_1000_residents": 120}
    for feature in geo["features"]:
        block_id = int(feature["properties"]["block_id"])
        row = result.rows.loc[block_id]
        for column in COLUMNS:
            if column in row:
                value = row[column]
                feature["properties"][column] = None if pd.isna(value) else float(value)
    return json.dumps(geo, ensure_ascii=False, allow_nan=False).encode("utf-8")
