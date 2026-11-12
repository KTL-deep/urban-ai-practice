"""Read-only access to the saved Pavlovsk study; no download or fabrication."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


REQUIRED = ("accessibility_matrix_drive.csv",)
OPTIONAL = ("provision_schools.csv", "study_info.json", "boundary.geojson")


class DataError(ValueError):
    """The saved study is absent or inconsistent."""


@dataclass(frozen=True)
class Study:
    blocks: gpd.GeoDataFrame
    matrix: pd.DataFrame
    population: pd.Series
    capacity: pd.Series
    metadata: dict
    saved_provision: pd.DataFrame | None
    boundary: dict | None
    directory: Path


def default_directory():
    original = Path(__file__).resolve().parents[1] / "pavlovsk_study_data"
    return original if original.exists() else Path(__file__).resolve().parent / "input"


def same_study_reference(directory, recorded_directory):
    """Accept a verified portable snapshot of the original read-only inputs."""
    directory = Path(directory).resolve()
    if str(directory) == recorded_directory:
        return True
    snapshot = Path(__file__).resolve().parent / "input"
    if directory != snapshot.resolve():
        return False
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("original_directory") != recorded_directory:
        return False
    return all((snapshot / name).is_file() and
               hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == digest
               for name, digest in manifest.get("sha256", {}).items()) and bool(manifest.get("sha256"))


def _ids(values, label):
    try:
        result = [int(value) for value in values]
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataError(f"{label}: ID должны быть целыми числами") from exc
    if any(str(original).strip() not in (str(value), f"{value}.0") for original, value in zip(values, result)):
        raise DataError(f"{label}: обнаружены некорректные ID")
    if len(set(result)) != len(result):
        raise DataError(f"{label}: повторяющиеся ID")
    return result


def _nonnegative(series, label, integer=False):
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any():
        raise DataError(f"{label}: пропуск, бесконечность или отрицательное значение")
    if integer and not (values == np.floor(values)).all():
        raise DataError(f"{label}: вместимость должна быть целочисленной")
    return values.astype(int if integer else float)


def load_study(directory=None):
    directory = Path(directory) if directory is not None else default_directory()
    block_file = ("blocks.geojson" if (directory / "blocks.geojson").is_file()
                  else "blocks_indicators.geojson")
    missing = [name for name in REQUIRED if not (directory / name).is_file()]
    if not (directory / block_file).is_file():
        missing.insert(0, "blocks.geojson (или blocks_indicators.geojson)")
    if missing:
        raise DataError(f"Не найдены обязательные файлы в {directory}: {', '.join(missing)}. "
                        "Укажите каталог сохранённых данных практики №2; загрузка из сети не выполняется.")
    try:
        blocks = gpd.read_file(directory / block_file)
        matrix = pd.read_csv(directory / REQUIRED[0], index_col=0)
    except Exception as exc:
        raise DataError(f"Не удалось прочитать кварталы или матрицу: {exc}") from exc
    if blocks.empty or blocks.crs is None or blocks.crs.to_epsg() != 4326:
        raise DataError("Кварталы должны быть непустым GeoJSON в EPSG:4326")
    if blocks.geometry.isna().any() or blocks.geometry.is_empty.any() or not blocks.geometry.is_valid.all():
        raise DataError("Некорректная или отсутствующая геометрия кварталов")
    if not blocks.geom_type.isin(("Polygon", "MultiPolygon")).all():
        raise DataError("Геометрия кварталов должна быть Polygon или MultiPolygon")
    for column in ("block_id", "population", "capacity_school"):
        if column not in blocks:
            raise DataError(f"В blocks.geojson нет поля {column}")
    ids = _ids(blocks.block_id.tolist(), "Кварталы")
    blocks = blocks.copy().set_axis(ids)
    blocks.index.name = "block_id"
    blocks["block_id"] = ids
    population = _nonnegative(blocks.population, "Население")
    # Reproduce the notebook exactly: missing source capacities become zero and
    # estimated fractional capacities are rounded BEFORE BlocksNet receives them.
    raw_capacity = pd.to_numeric(blocks.capacity_school, errors="coerce")
    if (raw_capacity.isna() & blocks.capacity_school.notna()).any() or (raw_capacity.dropna() < 0).any() or not np.isfinite(raw_capacity.dropna().to_numpy(dtype=float)).all():
        raise DataError("Вместимость: нечисловое, бесконечное или отрицательное значение")
    capacity = raw_capacity.fillna(0).round().astype(int)

    row_ids = _ids(matrix.index.tolist(), "Строки матрицы")
    col_ids = _ids(matrix.columns.tolist(), "Столбцы матрицы")
    if matrix.shape != (len(ids), len(ids)) or set(row_ids) != set(ids) or set(col_ids) != set(ids):
        raise DataError("ID строк и столбцов матрицы должны точно совпадать с ID кварталов")
    matrix.index = row_ids
    matrix.columns = col_ids
    matrix = matrix.loc[ids, ids].apply(pd.to_numeric, errors="coerce")
    values = matrix.to_numpy(dtype=float)
    if np.isnan(values).any() or (values < 0).any() or np.isneginf(values).any():
        raise DataError("Матрица времени содержит пропуски, отрицательные значения или -inf")
    # inf is the documented indicator of an unreachable directed journey.
    if not np.isfinite(np.diag(values)).all():
        raise DataError("Поездка внутри квартала должна иметь конечное время")

    info = {}
    if (directory / "study_info.json").is_file():
        info = json.loads((directory / "study_info.json").read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            raise DataError("study_info.json должен содержать объект")
        if info.get("travel_time_units") not in (None, "minutes") or info.get("travel_mode") not in (None, "drive (автомобиль)"):
            raise DataError("Метаданные не подтверждают автомобильные поездки в минутах")
    saved = None
    if (directory / "provision_schools.csv").is_file():
        saved = pd.read_csv(directory / "provision_schools.csv", index_col=0)
        saved.index = _ids(saved.index.tolist(), "Исходный расчёт")
        if set(saved.index) != set(ids) or len(saved) != len(ids):
            raise DataError("ID исходного расчёта не совпадают с кварталами")
        saved = saved.loc[ids]
    boundary = None
    if (directory / "boundary.geojson").is_file():
        boundary = json.loads((directory / "boundary.geojson").read_text(encoding="utf-8"))
    info.update({"territory": info.get("territory", "Павловск, Санкт-Петербург"),
                 "source_directory": str(directory.resolve()), "travel_mode": "drive (автомобиль)",
                 "travel_time_units": "minutes", "population_and_capacity": "Расчётные оценки практики №2",
                 "missing_capacity_policy": "По ноутбуку практики №2: NaN → 0; оценки округлены до целого",
                 "blocks_file": block_file})
    return Study(blocks, matrix, population, capacity, info, saved, boundary, directory)
