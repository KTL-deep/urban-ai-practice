"""Local Dash GIS: python school-placement/app.py --data-dir PATH."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Lock

import dash
from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update
import dash_leaflet as dl
from dash_extensions.javascript import Namespace
import pandas as pd

from data_loader import DataError, load_study, same_study_reference
from export import compare_frame, csv_bytes, geojson, quarters_frame
from model import calculate, verify_baseline
from search import search


STYLE = Namespace("schoolPlacement", "map")("style")
PALETTE = ["#b5424d", "#d66a50", "#e7ae6a", "#c8d8a2", "#5ca5a0", "#197a73"]
RANK_LABELS = {"block_id": "ID квартала", "new_capacity_places": "Новая школа, мест",
               "baseline_deficit_places": "Исходный дефицит, мест", "remaining_deficit_places": "Остаточный дефицит, мест",
               "reduction_places": "Сокращение, мест", "reduction_pct": "Сокращение, %",
               "status": "Статус", "error": "Ошибка"}
LAND_USE_LABELS = {"residential": "жилая", "business": "общественно-деловая",
                   "industrial": "промышленная", "agriculture": "сельскохозяйственная",
                   "recreation": "рекреационная", "transport": "транспортная",
                   "special": "специальная"}


def color(value, mode):
    if value is None or pd.isna(value):
        return "#cbd5e1"
    if mode == "Изменение":
        return "#b5424d" if value < -0.001 else "#dbe7eb" if value < 0.001 else "#96c9b7" if value < .15 else "#197a73"
    return PALETTE[min(5, max(0, int(float(value) * 6)))]


def legend_component(mode):
    if mode == "Изменение":
        items = [("#b5424d", "< −0,001"), ("#dbe7eb", "около 0"),
                 ("#96c9b7", "+0,001…+0,15"), ("#197a73", "≥ +0,15")]
        heading = "Изменение обеспеченности, доля (сценарий минус исходное)"
    else:
        items = list(zip(PALETTE, ("0–0,16", "0,17–0,33", "0,34–0,49", "0,50–0,66",
                                      "0,67–0,83", "0,84–1,00")))
        heading = "Обеспеченность, доля 0–1; шкала фиксирована между сценариями"
    items.append(("#cbd5e1", "нет спроса / нет данных"))
    return html.Div([html.Strong(heading), html.Div([
        html.Span([html.Span(className="swatch", style={"backgroundColor": shade}), label],
                  className="legend-item") for shade, label in items], className="legend-items")])


def map_data(study, baseline, result, mode, selected):
    view = baseline if mode == "Исходное состояние" else result
    geo = json.loads(study.blocks.to_json())
    for feature in geo["features"]:
        key = int(feature["properties"]["block_id"])
        value = (result.rows.loc[key, "provision_strong"] - baseline.rows.loc[key, "provision_strong"]
                 if mode == "Изменение" else view.rows.loc[key, "provision_strong"])
        feature["properties"] = {"block_id": key, "map_color": color(value, mode)}
    return geo


def _fmt(value):
    return "—" if value is None or pd.isna(value) else f"{value:,.1f}".replace(",", " ") if isinstance(value, float) else f"{value:,}".replace(",", " ")


def make_app(data_dir=None, results_dir=None):
    study = load_study(data_dir)
    baseline = calculate(study)
    verification = verify_baseline(baseline, study.saved_provision)
    app = Dash(__name__, assets_folder=str(Path(__file__).parent / "assets"),
               title="Школы · Павловск", suppress_callback_exceptions=True)
    app.server.config["study"] = study
    bounds = study.blocks.total_bounds
    # All mutable UI state is per browser session; calculation threads only share immutable input data.
    executor = ThreadPoolExecutor(max_workers=1)
    jobs = {}
    jobs_lock = Lock()
    from uuid import uuid4

    saved_path = (Path(results_dir) if results_dir is not None else Path(__file__).parent / "results") / "scenarios.json"
    saved_scenarios = {}
    if saved_path.exists():
        try:
            saved_document = json.loads(saved_path.read_text(encoding="utf-8"))
            if not same_study_reference(study.directory, saved_document.get("source_directory")):
                raise DataError(f"{saved_path} относится к другому каталогу данных; сохраните файл отдельно перед сменой набора")
            saved_scenarios = saved_document.get("scenarios", {})
            for item in saved_scenarios.values():
                from model import validate_schools
                validate_schools([tuple(pair) for pair in item["schools"]], set(study.population.index))
        except (ValueError, KeyError, TypeError) as exc:
            raise DataError(f"Некорректный сохранённый сценарий {saved_path}: {exc}") from exc

    saved_ranking = []
    saved_rank_status = "Автоматический поиск ещё не запускался."
    ranking_path = saved_path.parent / "automatic_ranking.csv"
    if not ranking_path.is_file() and (saved_path.parent / "automatic_ranking_incomplete.csv").is_file():
        ranking_path = saved_path.parent / "automatic_ranking_incomplete.csv"
    summary_path = (saved_path.parent / "automatic_search.json" if (saved_path.parent / "automatic_search.json").is_file()
                    else saved_path.parent / "experiment_summary.json")
    if ranking_path.name == "automatic_ranking_incomplete.csv":
        summary_path = saved_path.parent / "automatic_search_incomplete.json"
    if ranking_path.is_file() and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not same_study_reference(study.directory, summary.get("source_directory")):
            raise DataError("Рейтинг принадлежит другому каталогу данных")
        ranked = pd.read_csv(ranking_path, sep=";", encoding="utf-8-sig")
        if len(ranked) != len(study.population) or set(ranked.block_id) != set(study.population.index):
            raise DataError("Сохранённый рейтинг не охватывает все исходные кварталы")
        saved_ranking = [{**row, "id": int(row["block_id"])} for row in
                         ranked.astype(object).where(pd.notna(ranked), None).to_dict("records")]
        saved_rank_status = (f"Полный перебор {summary['blocks_checked']} кварталов. Лучшие ID: "
                             + ", ".join(map(str, summary["winners"])) + "." if summary.get("complete")
                             else f"Неполный перебор: {summary['errors']} ошибок; глобальный оптимум не установлен.")
        later_incomplete = saved_path.parent / "automatic_ranking_incomplete.csv"
        if (ranking_path.name == "automatic_ranking.csv" and later_incomplete.is_file()
                and later_incomplete.stat().st_mtime > ranking_path.stat().st_mtime):
            saved_rank_status += (" После этого выполнялся неполный поиск: его ошибки сохранены "
                                  "в automatic_ranking_incomplete.csv; показан предыдущий полный рейтинг.")

    def persist(scenarios):
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"source_directory": str(study.directory.resolve()), "scenarios":
                   {name: data for name, data in scenarios.items() if name != "Кандидат"}}
        reports = {name: (scenario_result(data["schools"]), data["reason"])
                   for name, data in payload["scenarios"].items()}
        (saved_path.parent / "scenario_comparison.csv").write_bytes(csv_bytes(compare_frame(reports, baseline)))
        for name, (outcome, _) in reports.items():
            (saved_path.parent / f"scenario_{name}.csv").write_bytes(csv_bytes(quarters_frame(outcome)))
        temp = saved_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(saved_path)

    calculated = {(): baseline}
    def scenario_result(schools):
        key = tuple((int(i), int(c)) for i, c in schools)
        if key not in calculated:
            calculated[key] = calculate(study, key, baseline)
        return calculated[key]

    def figure(result, mode, selected):
        polygons = dl.GeoJSON(data=map_data(study, baseline, result, mode, selected),
                              options={"style": STYLE}, hideout={"selected": selected}, id="polygons")
        schools = []
        if mode != "Исходное состояние":
            for block_id, places in result.schools:
                point = study.blocks.loc[block_id].geometry.representative_point()
                schools.append(dl.CircleMarker(center=[point.y, point.x], radius=9,
                               color="#112d41", fillColor="#f5c45f", fillOpacity=1,
                               children=dl.Tooltip(f"Новая школа · квартал {block_id} · {places} мест")))
        return [polygons, *schools]

    def current(name, scenarios):
        if name == "Исходное состояние" or name not in scenarios:
            return baseline
        return scenario_result(scenarios[name]["schools"])

    app.layout = html.Div(className="page", children=[
        dcc.Store(id="draft", data=[]), dcc.Store(id="scenarios", data=saved_scenarios), dcc.Store(id="selected", data=None),
        dcc.Store(id="job", data=None), dcc.Interval(id="timer", interval=800, disabled=True),
        dcc.Download(id="download"),
        html.Header([html.Div("ПРОСТРАНСТВЕННАЯ АНАЛИТИКА  /  ПРАКТИКА №3", className="eyebrow"),
                     html.H1("Школы и доступность"), html.P(study.metadata["territory"]),
                     html.P("Сравнение размещения школ по кварталам · поездки на автомобиле до 15 минут")]),
        html.Div(id="status", className="notice", children=verification),
        html.Div(className="workspace", children=[
            html.Aside(className="panel controls", children=[
                html.H2("Конструктор сценария"),
                html.Label("Сценарий"), dcc.Dropdown(id="scenario-name", options=[{"label": x, "value": x} for x in ("А", "Б", "В", "Свой")], value="А", clearable=False),
                html.Label("Квартал (также выбирается кликом на карте)"), dcc.Dropdown(id="block", options=[{"label": str(i), "value": int(i)} for i in study.population.index], placeholder="Выберите квартал"),
                html.Label("Вместимость новой школы, мест"), dcc.Input(id="places", type="number", min=1, step=1, value=600),
                html.Button("Добавить школу", id="add", className="primary"),
                html.Div(id="draft-list"), html.Label("Обоснование выбора А/Б до автопоиска"),
                dcc.Textarea(id="reason", placeholder="Почему выбраны эти кварталы?", style={"width": "100%", "height": "76px"}),
                html.Button("Рассчитать и сохранить сценарий", id="run", className="primary"),
                html.Hr(), html.H2("Автоматическое размещение"), html.Label("Вместимость, мест"),
                dcc.Input(id="auto-places", type="number", min=1, step=1, value=600),
                html.Button("Разместить автоматически", id="automatic"), html.Div(id="progress"),
                html.Button("Сбросить черновик и просмотр", id="reset", className="quiet"),
            ]),
            html.Section(className="panel map-panel", children=[
                html.Div(className="panel-heading", children=[html.H2("Карта кварталов"), dcc.RadioItems(
                    id="map-mode", options=["Исходное состояние", "Сценарий", "Изменение"], value="Исходное состояние", inline=True)]),
                dl.Map(id="map", center=[(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2],
                       bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]], children=[dl.LayerGroup(id="map-layers")], style={"height": "570px", "width": "100%"}),
                html.Div(id="legend", className="legend"), html.Div(id="quarter-detail", className="detail")
            ])]),
        html.Section(className="panel", children=[html.H2("Показатели территории"), html.Div(id="totals", className="cards")]),
        html.Section(className="panel", children=[html.H2("Сравнение сценариев"), dcc.Dropdown(id="view", value="Исходное состояние", clearable=False),
            dash_table.DataTable(id="comparison", page_size=8, style_table={"overflowX": "auto"}, style_cell={"textAlign": "left"})]),
        html.Section(className="panel", children=[html.H2("По кварталам"), dash_table.DataTable(id="quarters", page_size=15, sort_action="native", filter_action="native", style_table={"overflowX": "auto"})]),
        html.Section(className="panel", children=[html.H2("Рейтинг размещения одной школы"),
            html.P("Нажмите строку рейтинга, чтобы открыть расчёт кандидата на карте."),
            dash_table.DataTable(id="ranking", data=saved_ranking,
                                 columns=[{"name": value, "id": key} for key, value in RANK_LABELS.items()],
                                 page_size=15, sort_action="native", style_table={"overflowX": "auto"}),
            html.Div(id="ranking-status", children=saved_rank_status)]),
        html.Section(className="panel exports", children=[html.H2("Экспорт"),
            *[html.Button(label, id=identifier) for identifier, label in (("export-quarters", "Кварталы · CSV"), ("export-ranking", "Рейтинг · CSV"),
                ("export-comparison", "Сценарии · CSV"), ("export-geojson", "Сценарий · GeoJSON"))]]),
        html.Footer([html.P("Источники: локальные файлы практики №2 · IDUclub/blocksnet-data v0.0.1. Население и вместимость — расчётные оценки. Порог: 15 минут на автомобиле; потребность: 120 мест / 1000 жителей."),
                     html.P("Выбор квартала не подтверждает наличие земельного участка или возможность строительства школы.")])
    ])

    @app.callback(Output("selected", "data"), Output("block", "value"), Input("polygons", "clickData"), prevent_initial_call=True)
    def click_quarter(clicked):
        if clicked and clicked.get("properties"):
            key = int(clicked["properties"]["block_id"])
            return key, key
        return no_update, no_update

    @app.callback(Output("draft", "data"), Output("status", "children"), Input("add", "n_clicks"), Input("reset", "n_clicks"),
                  Input({"type": "remove", "index": dash.ALL}, "n_clicks"), State("draft", "data"), State("block", "value"), State("places", "value"), prevent_initial_call=True)
    def edit_draft(add, reset, removed, draft, block_id, places):
        trigger = dash.ctx.triggered_id
        if trigger == "reset":
            return [], "Черновик очищен; сохранённые сценарии не изменены."
        if isinstance(trigger, dict):
            return [entry for idx, entry in enumerate(draft) if idx != trigger["index"]], "Школа удалена из черновика."
        if block_id is None or isinstance(places, bool) or places is None or not isinstance(places, (int, float)) or int(places) != places or places <= 0:
            return no_update, "Выберите квартал и положительную целочисленную вместимость."
        return draft + [{"id": int(block_id), "places": int(places)}], "Школа добавлена в черновик."

    @app.callback(Output("draft-list", "children"), Input("draft", "data"))
    def draft_list(draft):
        return [html.Div([f"Квартал {e['id']} · {e['places']} мест", html.Button("×", id={"type": "remove", "index": idx}, title="Удалить")], className="draft-item") for idx, e in enumerate(draft)] or "Новые школы пока не добавлены."

    @app.callback(Output("scenarios", "data"), Output("view", "value"), Output("map-mode", "value", allow_duplicate=True),
                  Output("status", "children", allow_duplicate=True),
                  Input("run", "n_clicks"), State("scenario-name", "value"), State("draft", "data"), State("reason", "value"), State("scenarios", "data"), prevent_initial_call=True)
    def save_scenario(_, name, draft, reason, scenarios):
        if not draft:
            return no_update, no_update, no_update, "Добавьте хотя бы одну школу в черновик."
        if name == "В":
            return no_update, no_update, no_update, "Сценарий В заполняется только полным автоматическим поиском."
        if name in ("А", "Б") and (not reason or not reason.strip()):
            return no_update, no_update, no_update, "Перед расчётом А/Б запишите обоснование ручного выбора."
        if name == "А" and (len(draft) != 1 or draft[0]["places"] != 600):
            return no_update, no_update, no_update, "Для А требуется одна школа на 600 мест."
        if name == "Б" and (len(draft) != 2 or any(e["places"] != 300 for e in draft) or draft[0]["id"] == draft[1]["id"]):
            return no_update, no_update, no_update, "Для Б нужны две школы по 300 мест в разных кварталах."
        try:
            schools = [(e["id"], e["places"]) for e in draft]
            scenario_result(schools)
        except Exception as exc:
            return no_update, no_update, no_update, f"Расчёт не выполнен: {exc}"
        updated = dict(scenarios)
        updated[name] = {"schools": schools, "reason": (reason or "").strip()}
        try:
            persist(updated)
        except OSError as exc:
            return no_update, no_update, no_update, f"Не удалось сохранить сценарий: {exc}"
        return updated, name, "Сценарий", f"Сценарий {name} рассчитан и сохранён."

    @app.callback(Output("job", "data"), Output("timer", "disabled"), Output("progress", "children"),
                  Input("automatic", "n_clicks"), State("auto-places", "value"), State("scenarios", "data"), prevent_initial_call=True)
    def start_search(_, places, scenarios):
        if places is None or isinstance(places, bool) or not isinstance(places, (int, float)) or int(places) != places or places <= 0:
            return no_update, True, "Введите положительное целое число мест."
        if "А" not in scenarios or "Б" not in scenarios:
            return no_update, True, "До автопоиска рассчитайте А и Б с обоснованием выбора."
        with jobs_lock:
            if any(not job["future"].done() for job in jobs.values()):
                return no_update, no_update, "Поиск уже выполняется."
            token = uuid4().hex
            state = {"done": 0, "total": len(study.population), "errors": 0}
            def progress(done, total, errors):
                with jobs_lock:
                    state.update(done=done, total=total, errors=errors)
            future = executor.submit(search, study, baseline, int(places), progress)
            jobs[token] = {"future": future, "state": state, "places": int(places)}
        return token, False, "Поиск запущен."

    @app.callback(Output("progress", "children", allow_duplicate=True), Output("timer", "disabled", allow_duplicate=True),
                  Output("ranking", "data"), Output("ranking", "columns"), Output("ranking-status", "children"), Output("scenarios", "data", allow_duplicate=True),
                  Output("view", "value", allow_duplicate=True), Output("map-mode", "value", allow_duplicate=True),
                  Output("selected", "data", allow_duplicate=True),
                  Input("timer", "n_intervals"), State("job", "data"), State("scenarios", "data"), prevent_initial_call=True)
    def poll(_, token, scenarios):
        if not token or token not in jobs:
            return no_update, True, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        job = jobs[token]
        with jobs_lock:
            state = dict(job["state"])
        line = f"Проверено: {state['done']} из {state['total']}; осталось: {state['total'] - state['done']}; ошибок: {state['errors']}."
        if not job["future"].done():
            return line, False, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        try:
            result = job["future"].result()
        except Exception as exc:
            with jobs_lock:
                jobs.pop(token, None)
            return line, True, no_update, no_update, f"Поиск завершился ошибкой: {exc}", no_update, no_update, no_update, no_update
        with jobs_lock:
            jobs.pop(token, None)
        suffix = "" if job["places"] == 600 else f"_{job['places']}"
        if not result.complete:
            suffix += "_incomplete"
        try:
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            (saved_path.parent / f"automatic_ranking{suffix}.csv").write_bytes(csv_bytes(pd.DataFrame(result.candidates)))
            (saved_path.parent / f"automatic_search{suffix}.json").write_text(json.dumps({
                "source_directory": str(study.directory.resolve()), "capacity_places": job["places"],
                "blocks_checked": result.checked, "errors": result.errors, "complete": result.complete,
                "winners": result.winners}, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            line += f" Рейтинг не удалось записать на диск: {exc}."
        status = (f"Полный перебор {result.checked} кварталов. Лучшие ID: {', '.join(map(str, result.winners))}." if result.complete
                  else f"Неполный перебор: {result.errors} ошибок из {result.checked}. Глобальный оптимум НЕ установлен.")
        updated = dict(scenarios)
        if not result.complete and "В" in scenarios:
            status += " Ранее сохранённый В относится к предыдущему успешному поиску."
        auto_name = "В" if job["places"] == 600 else "Автопоиск"
        if result.complete and result.winners:
            updated[auto_name] = {"schools": [[result.winners[0], job["places"]]], "reason": "Полный перебор; равноценные кварталы: " + ", ".join(map(str, result.winners))}
            try:
                persist(updated)
            except OSError as exc:
                line += f" Сценарий {auto_name} не удалось записать на диск: {exc}."
        return (line, True, [{**row, "id": row["block_id"]} for row in result.candidates],
                [{"name": v, "id": k} for k, v in RANK_LABELS.items()],
                status, updated, auto_name if result.complete and result.winners else no_update,
                "Сценарий" if result.complete and result.winners else no_update,
                result.winners[0] if result.complete and result.winners else no_update)

    @app.callback(Output("view", "options"), Input("scenarios", "data"))
    def view_options(scenarios):
        return ["Исходное состояние", *[name for name in ("А", "Б", "В", "Свой", "Автопоиск", "Кандидат") if name in scenarios]]

    @app.callback(Output("view", "value", allow_duplicate=True), Output("draft", "data", allow_duplicate=True),
                  Output("map-mode", "value", allow_duplicate=True),
                  Output("selected", "data", allow_duplicate=True), Output("block", "value", allow_duplicate=True),
                  Output("reason", "value"), Output("places", "value"),
                  Input("reset", "n_clicks"), prevent_initial_call=True)
    def reset_view(_):
        return "Исходное состояние", [], "Исходное состояние", None, None, "", 600

    @app.callback(Output("map-layers", "children"), Output("legend", "children"), Output("quarter-detail", "children"),
                  Output("totals", "children"), Output("quarters", "data"), Output("quarters", "columns"),
                  Output("comparison", "data"), Output("comparison", "columns"),
                  Input("view", "value"), Input("map-mode", "value"), Input("selected", "data"), Input("scenarios", "data"))
    def render(name, mode, selected, scenarios):
        result = current(name, scenarios)
        frame = quarters_frame(result)
        comparison = compare_frame({key: (scenario_result(data["schools"]), data["reason"]) for key, data in scenarios.items()}, baseline)
        t = result.totals
        cards = [html.Div([html.Span(label), html.Strong(_fmt(value))], className="card") for label, value in
                 (("Население · чел.", t["population"]), ("Потребность · мест", t["demand"]),
                  ("Вместимость · мест", t["capacity"]), ("Обслужено · мест", t["served"]),
                  ("Остаточный дефицит · мест", t["deficit"]),
                  ("Сокращение · мест", t.get("reduction", 0)))]
        detail = "Нажмите на квартал на карте."
        if selected in study.population.index:
            row = result.rows.loc[selected]
            land_use = study.blocks.loc[selected].get("land_use", "не определено")
            land_use = LAND_USE_LABELS.get(str(land_use), land_use)
            detail = (f"Квартал {selected} · население {_fmt(row.population)} чел. · вместимость {_fmt(int(row.capacity))} мест "
                      f"· спрос {_fmt(int(row.demand))} мест · дефицит {_fmt(int(row.deficit))} мест "
                      f"· обеспеченность {_fmt(row.provision_strong)} (доля) · использование: {land_use}. "
                      "Возможность строительства требуется проверить отдельно.")
        legend = legend_component(mode)
        return (figure(result, mode, selected), legend, detail, cards,
                frame.astype(object).where(pd.notna(frame), None).to_dict("records"), [{"name": c, "id": c} for c in frame],
                comparison.to_dict("records"), [{"name": c, "id": c} for c in comparison])

    @app.callback(Output("view", "value", allow_duplicate=True), Output("selected", "data", allow_duplicate=True),
                  Output("map-mode", "value", allow_duplicate=True),
                  Output("scenarios", "data", allow_duplicate=True), Input("ranking", "active_cell"),
                   State("ranking", "derived_viewport_data"), State("ranking", "data"), State("scenarios", "data"), prevent_initial_call=True)
    def select_rank(cell, visible, all_rows, scenarios):
        if not cell:
            return no_update, no_update, no_update, no_update
        rows = visible if visible is not None else all_rows
        if cell.get("row_id") is not None:
            row = next((r for r in (all_rows or []) if r["block_id"] == cell["row_id"]), None)
        else:
            row = rows[cell["row"]] if rows and 0 <= cell["row"] < len(rows) else None
        if row is None or row["status"] != "Рассчитан":
            return no_update, no_update, no_update, no_update
        updated = dict(scenarios)
        updated["Кандидат"] = {"schools": [[row["block_id"], row["new_capacity_places"]]], "reason": "Просмотр строки рейтинга"}
        return "Кандидат", row["block_id"], "Сценарий", updated

    @app.callback(Output("download", "data"), Input("export-quarters", "n_clicks"), Input("export-ranking", "n_clicks"),
                  Input("export-comparison", "n_clicks"), Input("export-geojson", "n_clicks"),
                  State("view", "value"), State("scenarios", "data"), State("ranking", "data"), prevent_initial_call=True)
    def download(a, b, c, d, name, scenarios, ranking):
        kind = dash.ctx.triggered_id
        result = current(name, scenarios)
        if kind == "export-geojson":
            return dcc.send_bytes(geojson(study, result), "school_scenario.geojson")
        if kind == "export-quarters":
            return dcc.send_bytes(csv_bytes(quarters_frame(result)), "school_quarters.csv")
        if kind == "export-comparison":
            frame = compare_frame({key: (scenario_result(data["schools"]), data["reason"]) for key, data in scenarios.items()}, baseline)
            return dcc.send_bytes(csv_bytes(frame), "school_scenarios.csv")
        if kind == "export-ranking" and ranking:
            return dcc.send_bytes(csv_bytes(pd.DataFrame(ranking).drop(columns=["id"], errors="ignore")), "school_ranking.csv")
        return no_update
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    try:
        app = make_app(args.data_dir, args.results_dir)
    except (DataError, ImportError) as exc:
        parser.exit(2, f"Не удалось запустить приложение: {exc}\n")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
