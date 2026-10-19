"""Build one self-contained-data Pavlovsk dashboard from saved study outputs.

Usage: python examples/build_pavlovsk_dashboard.py [--data-dir DIR] [--output FILE] [--overwrite]
The HTML embeds data and charts; Leaflet assets and CARTO basemap tiles need internet.
"""

import argparse
import base64
import csv
import html
import io
import json
import math
from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from branca.element import Element, MacroElement, Template


DATA_DIR = Path(__file__).resolve().parent / "pavlovsk_study_data"
OUTPUT = Path.home() / "Downloads" / "pavlovsk_map.html"
METRICS = {
    "fsi": ("FSI · плотность застройки", "#2563eb"),
    "gsi": ("GSI · доля застройки", "#0d9488"),
    "mxi": ("MXI · смешанность функций", "#7c3aed"),
    "area_accessibility": ("Доступность территории · мин", "#d97706"),
    "provision_strong": ("Обеспеченность школами · 0–1", "#e11d48"),
    "population": ("Население · чел", "#0891b2"),
    "travel_time_median_min": ("Медиана поездки · мин", "#ca8a04"),
    "strelka_morphotype": ("Морфотип · Strelka", "#7c3aed"),
    "spacematrix_morphotype": ("Морфотип · Spacematrix", "#7c3aed"),
}
NUMERIC = set(METRICS) - {"strelka_morphotype", "spacematrix_morphotype"}
SCHOOL_COLUMNS = (
    "demand", "capacity", "demand_left", "demand_within", "demand_without",
    "capacity_left", "capacity_within", "capacity_without", "provision_strong", "provision_weak",
)


def safe_json(value):
    """Serialize for an inline HTML script, including hostile CSV/GeoJSON strings."""
    return (json.dumps(value, ensure_ascii=False, allow_nan=False)
            .replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def clean(value):
    """Convert non-finite numbers to JSON null recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def display(value):
    if value is None or value == "" or (isinstance(value, str) and value.lower() in ("nan", "inf", "-inf")):
        return "—"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:,.1f}".replace(",", " ") if isinstance(value, float) else f"{value:,}".replace(",", " ")
    return str(value)


def load_data(data_dir):
    """Read only saved JSON/CSV; reject mismatched or incomplete input before writing."""
    names = ("blocks_indicators.geojson", "blocks_indicators.csv", "provision_schools.csv",
             "study_info.json", "boundary.geojson")
    for name in names:
        if not (data_dir / name).is_file():
            raise ValueError(f"Отсутствует исходный файл: {data_dir / name}")
    try:
        blocks = json.loads((data_dir / names[0]).read_text(encoding="utf-8"))
        boundary = json.loads((data_dir / names[4]).read_text(encoding="utf-8"))
        info = json.loads((data_dir / names[3]).read_text(encoding="utf-8"))
        with (data_dir / names[1]).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or reader.fieldnames[0] != "":
                raise ValueError("В blocks_indicators.csv отсутствует первый столбец ID")
            columns = [column for column in (reader.fieldnames or []) if column and not column.startswith("Unnamed")]
            rows = list(reader)
        with (data_dir / names[2]).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or reader.fieldnames[0] != "":
                raise ValueError("В provision_schools.csv отсутствует первый столбец ID")
            schools = list(reader)
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as exc:
        raise ValueError(f"Не удалось прочитать исходные данные: {exc}") from exc
    if not isinstance(info, dict) or not isinstance(blocks, dict) or blocks.get("type") != "FeatureCollection":
        raise ValueError("Неверный формат метаданных или кварталов GeoJSON")
    if not isinstance(boundary, dict) or boundary.get("type") != "FeatureCollection" or not boundary.get("features"):
        raise ValueError("Неверный формат границы GeoJSON")
    features = blocks.get("features")
    if not isinstance(features, list) or not features or len(rows) != len(features) or len(schools) != len(features):
        raise ValueError("Количество кварталов в GeoJSON и таблицах должно совпадать и быть ненулевым")
    if not {"fsi", "gsi", "mxi", "area_accessibility", "provision_strong", "population"}.issubset(columns):
        raise ValueError("В blocks_indicators.csv отсутствуют обязательные показатели")
    if not set(SCHOOL_COLUMNS).issubset(schools[0]):
        raise ValueError("В provision_schools.csv отсутствуют обязательные показатели")
    ids = []
    for index, (feature, row, school) in enumerate(zip(features, rows, schools)):
        if not isinstance(feature, dict) or not isinstance(feature.get("properties"), dict) or not feature.get("geometry"):
            raise ValueError(f"Неверная геометрия квартала {index}")
        block_id = str(feature["properties"].get("block_id", index))
        if block_id in ids or str(row.get("", index)) != block_id or str(school.get("", index)) != block_id:
            raise ValueError(f"Несовпадение или повторение ID квартала {block_id}")
        ids.append(block_id)
    return features, boundary, info, rows, schools, columns, ids


def chart(values, title, label, categorical=False):
    """Return an inline PNG: image contents never execute HTML or JavaScript."""
    fig, ax = plt.subplots(figsize=(6.8, 3), layout="constrained")
    fig.patch.set_facecolor("#ffffff")
    if categorical:
        from collections import Counter
        counts = Counter(str(value or "Нет данных") for value in values)
        labels, heights = zip(*counts.most_common(8))
        ax.barh(range(len(labels)), heights, color="#2563eb")
        ax.set_yticks(range(len(labels)), [label[:34] for label in labels], fontsize=8)
        ax.invert_yaxis()
    else:
        valid = [value for value in (number(item) for item in values) if value is not None]
        if valid:
            ax.hist(valid, bins=min(16, max(1, len(set(valid)))), color="#2563eb", edgecolor="white")
        ax.set_xlabel(label)
        ax.set_ylabel("Кварталов")
    ax.set_title(title, loc="left", fontsize=11, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=125, facecolor="white")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def table(rows, columns, table_id, ids):
    head = "".join(f"<th scope='col'>{html.escape(column)}</th>" for column in ("block_id", *columns))
    body = "".join("<tr>" + "".join(f"<td>{html.escape(display(value))}</td>" for value in
                   (block_id, *(row.get(column) for column in columns))) + "</tr>"
                   for block_id, row in zip(ids, rows))
    return f"<div class='table-scroll'><table id='{table_id}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def build_map(features, boundary):
    # Drop hundreds of unrelated properties: only the selected indicators enter the map.
    geo = {"type": "FeatureCollection", "features": []}
    for feature in features:
        props = feature["properties"]
        selected = {key: (number(props.get(key)) if key in NUMERIC else props.get(key)) for key in METRICS}
        selected["block_id"] = str(props.get("block_id", "—"))
        geo["features"].append({"type": "Feature", "geometry": clean(feature["geometry"]),
                                "properties": clean(selected)})
    # Folium derives the view from the actual geometry, not a guessed city centre.
    fmap = folium.Map(location=[59.68, 30.45], zoom_start=11, tiles=None, control_scale=True)
    basemap = folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
             '&copy; <a href="https://carto.com/attributions">CARTO</a>',
        name="Светлая подложка CARTO", subdomains="abcd", max_zoom=20, control=False,
    ).add_to(fmap)
    boundary_layer = folium.GeoJson(clean(boundary), name="Граница исследования", style_function=lambda _:
                                    {"color": "#0f172a", "weight": 3, "fillOpacity": 0, "dashArray": "6 4"}).add_to(fmap)
    options = "".join(f"<option value='{key}'>{html.escape(label)}</option>" for key, (label, _) in METRICS.items())
    control = ('<label for="metric">Показатель карты</label><select id="metric">'
               + options + '</select><div id="legend"></div>')
    js = """
const data = __DATA__, metrics = __METRICS__, map = __MAP__;
const escapeHtml = x => String(x ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const numeric = ['fsi','gsi','mxi','area_accessibility','provision_strong','population','travel_time_median_min'];
const extent = L.geoJSON(data).getBounds(); if (extent.isValid()) map.fitBounds(extent, {padding:[14,14]});
const selector = L.control({position:'topright'});
selector.onAdd = function () { const div = L.DomUtil.create('div','metric-control');
  div.innerHTML = __CONTROL__;
  L.DomEvent.disableClickPropagation(div); L.DomEvent.disableScrollPropagation(div); return div; };
selector.addTo(map);
let metric = 'fsi';
const values = Object.fromEntries(numeric.map(k => [k, data.features.map(f => f.properties[k]).filter(v => typeof v === 'number' && Number.isFinite(v))]));
const ranges = Object.fromEntries(numeric.map(k => [k, [Math.min(...values[k]), Math.max(...values[k])] ]));
const categoryColors = {};
function color(p) {
  const v = p[metric]; if (v === null || v === undefined || v === '') return '#cbd5e1';
  if (!numeric.includes(metric)) { const text = String(v); if (!categoryColors[text])
    categoryColors[text] = ['#2563eb','#f97316','#10b981','#a855f7','#e11d48','#eab308','#06b6d4'][Object.keys(categoryColors).length % 7];
    return categoryColors[text]; }
  const [low,high] = ranges[metric]; if (!Number.isFinite(low)) return '#cbd5e1';
  const t = high === low ? 0.5 : Math.max(0,Math.min(1,(v-low)/(high-low)));
  return `hsl(${215-185*t}, 74%, ${55-12*t}%)`;
}
const layer = L.geoJSON(data, {style:f => ({color:'#ffffff', weight:1, fillColor:color(f.properties), fillOpacity:.82}),
  onEachFeature:(f,l) => l.bindPopup(() => '<strong>Квартал '+escapeHtml(f.properties.block_id)+'</strong><br>'+Object.entries(metrics).map(([k,label]) => escapeHtml(label[0])+': '+escapeHtml(f.properties[k])).join('<br>'))});
layer.addTo(map);
const basemap = __BASEMAP__, blank = L.layerGroup();
L.control.layers({'Светлая подложка CARTO':basemap, 'Без подложки':blank},
  {'Кварталы':layer, 'Граница исследования':__BOUNDARY__}, {collapsed:false}).addTo(map);
const tileWarning = L.control({position:'bottomleft'});
tileWarning.onAdd = function () { const div = L.DomUtil.create('div','metric-control');
  div.textContent = 'Подложка недоступна. Выберите «Без подложки»: кварталы и показатели сохранятся.';
  L.DomEvent.disableClickPropagation(div); return div; };
let warned = false;
basemap.on('tileerror', () => { if (!warned) { tileWarning.addTo(map); warned = true; } });
map.on('baselayerchange', () => { if (warned) { tileWarning.remove(); warned = false; } });
function update() { metric = document.getElementById('metric').value; layer.setStyle(f => ({fillColor:color(f.properties)}));
  const [low,high] = ranges[metric] || []; document.getElementById('legend').textContent = numeric.includes(metric)
    ? 'Диапазон: '+(Number.isFinite(low)?low.toFixed(2):'нет данных')+' — '+(Number.isFinite(high)?high.toFixed(2):'нет данных')+'; серый — нет данных'
    : 'Цвет — категория; серый — нет данных'; }
document.getElementById('metric').addEventListener('change',update); update();
"""
    js = js.replace("__DATA__", safe_json(geo)).replace("__METRICS__", safe_json(METRICS))
    js = js.replace("__MAP__", fmap.get_name()).replace("__BOUNDARY__", boundary_layer.get_name())
    js = js.replace("__BASEMAP__", basemap.get_name())
    js = js.replace("__CONTROL__", safe_json(control))
    fmap.get_root().header.add_child(Element("<style>.metric-control{background:#fff;padding:10px;border-radius:10px;box-shadow:0 2px 12px #0003;max-width:210px;font:13px sans-serif}.metric-control label{display:block;font-weight:bold;margin-bottom:5px}.metric-control select{width:100%;padding:5px}.metric-control #legend{margin-top:6px}</style>"))
    # A child macro runs after Folium has created the map and boundary layers.
    # Adding directly to root.script would execute before the map variable exists.
    script = MacroElement()
    script._template = Template("{% macro script(this, kwargs) %}{{ this.javascript | safe }}{% endmacro %}")
    script.javascript = js
    fmap.add_child(script)
    return fmap.get_root().render()


def render(data_dir):
    features, boundary, info, rows, schools, columns, ids = load_data(data_dir)
    figures = [
        ("Плотность застройки", chart((row.get("fsi") for row in rows), "FSI по кварталам", "FSI")),
        ("Доступность территории", chart((row.get("area_accessibility") for row in rows), "Доступность территории", "минуты, автомобиль")),
        ("Обеспеченность школами", chart((row.get("provision_strong") for row in schools), "Школы: сильная обеспеченность", "доля 0–1")),
        ("Морфотипы Strelka", chart((row.get("strelka_morphotype") for row in rows), "Морфотипы Strelka", "", True)),
        ("Морфотипы Spacematrix", chart((row.get("spacematrix_morphotype") for row in rows), "Морфотипы Spacematrix", "", True)),
    ]
    populations = [number(row.get("population")) for row in rows]
    demand = sum(number(row.get("demand")) or 0 for row in schools)
    served = sum(number(row.get("demand_within")) or 0 for row in schools)
    fsi = [number(row.get("fsi")) for row in rows]
    access = [number(row.get("area_accessibility")) for row in rows]
    def median(items):
        valid = sorted(value for value in items if value is not None)
        if not valid:
            return None
        middle = len(valid) // 2
        return valid[middle] if len(valid) % 2 else (valid[middle - 1] + valid[middle]) / 2
    cards = [("Кварталов", len(rows)), ("Население, сумма по кварталам", sum(v or 0 for v in populations)),
             ("FSI, медиана кварталов", median(fsi)), ("Доступность, медиана · мин", median(access)),
             ("Спрос на школы, сумма", demand),
             ("Обеспечено в пределах доступности", f"{100 * served / demand:.1f}%" if demand else None)]
    card_html = "".join(f"<article class='card'><span>{html.escape(label)}</span><strong>{html.escape(display(value))}</strong></article>" for label, value in cards)
    figure_html = "".join(f"<figure><img alt='{html.escape(label, quote=True)}' src='data:image/png;base64,{image}'><figcaption>{html.escape(label)}</figcaption></figure>" for label, image in figures)
    conclusions = [f"Исследование содержит {len(rows)} кварталов; суммарное население по таблице — {display(sum(v or 0 for v in populations))} чел."]
    if median(fsi) is not None:
        conclusions.append(f"Медианный FSI по кварталам с данными — {display(median(fsi))} (без взвешивания по площади).")
    if median(access) is not None:
        conclusions.append(f"Медианная доступность территории по кварталам с данными — {display(median(access))} мин; режим поездки — автомобиль.")
    if demand:
        conclusions.append(f"По таблице школ в пределах принятой доступности обеспечено {display(served)} из {display(demand)} мест спроса ({display(100 * served / demand)}%); это отношение сумм спроса, не среднее долей и не оценка пешеходной доступности.")
    else:
        conclusions.append("Суммарный спрос на школы равен нулю: доля обеспеченности территории не определяется.")
    conclusions_html = "".join(f"<li>{html.escape(item)}</li>" for item in conclusions)
    metadata = "".join(f"<dt>{html.escape(str(key))}</dt><dd>{html.escape(display(value))}</dd>" for key, value in info.items())
    map_html = html.escape(build_map(features, boundary), quote=True)
    block_table = table(rows, columns, "blocks-table", ids)
    school_table = table(schools, SCHOOL_COLUMNS, "schools-table", ids)
    return f"""<!doctype html><html lang='ru'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Павловск · Городские кварталы</title><style>
:root{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#14243e;background:#f3f6fb}}*{{box-sizing:border-box}}
body{{margin:0}}header{{background:linear-gradient(115deg,#102b55,#166e83);color:white;padding:40px max(24px,calc((100vw - 1200px)/2))}}
header h1{{font-size:clamp(2rem,4vw,3.3rem);margin:.2em 0}}header p{{max-width:65ch;line-height:1.6}}
main{{max-width:1250px;margin:auto;padding:22px}}h2{{margin-top:36px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px}}
.card,figure,.panel{{background:white;border:1px solid #e0e9f2;border-radius:15px;box-shadow:0 3px 16px #1232;padding:18px}}
.card span{{display:block;color:#53657f;font-size:.88em}}.card strong{{display:block;margin-top:12px;font-size:1.5rem}}iframe{{width:100%;height:580px;border:0;border-radius:15px;background:white}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:14px}}figure{{margin:0}}figure img{{display:block;max-width:100%;height:auto}}figcaption{{font-size:.85em;color:#53657f}}
.table-scroll{{overflow:auto;max-height:490px}}table{{border-collapse:collapse;white-space:nowrap;font-size:.84em}}th,td{{padding:9px;border-bottom:1px solid #e2e8f0;text-align:left}}th{{position:sticky;top:0;background:#e9f1fa;cursor:pointer}}tr:nth-child(even){{background:#f8fafc}}
input[type=search]{{padding:10px;border:1px solid #aabbd0;border-radius:8px;width:min(100%,380px);margin:8px 0 14px}}details{{margin-top:14px}}summary{{cursor:pointer;font-weight:bold}}dt{{font-weight:bold;margin-top:8px}}dd{{margin-left:0;overflow-wrap:anywhere}}footer{{color:#53657f;padding:20px 0}}
</style></head><body><header><small>ИНТЕРАКТИВНЫЙ ОБЗОР · ПАВЛОВСК</small><h1>Кварталы, плотность и доступность</h1><p>Сводка по сохранённым результатам исследования BlocksNet. Переключайте показатель на карте, ищите кварталы и сортируйте таблицы.</p></header>
<main><section aria-label='Ключевые показатели' class='cards'>{card_html}</section>
<section><h2>Карта кварталов</h2><p>Граница исследования и показатели кварталов. Серый цвет — нет данных. Подложка CARTO использует данные OpenStreetMap. Для загрузки Leaflet и подложки нужен интернет; генерация отчёта сеть не использует. Если фон недоступен, выберите «Без подложки» в переключателе слоёв: кварталы и показатели останутся видны.</p><iframe title='Карта кварталов Павловска' srcdoc='{map_html}' loading='lazy'></iframe></section>
<section><h2>Распределения</h2><div class='charts'>{figure_html}</div></section>
<section class='panel'><h2>Выводы по сохранённым данным</h2><ul>{conclusions_html}</ul><p>Это описательные результаты, рассчитанные из предоставленных таблиц, а не причинные выводы и не текст исходного ноутбука.</p></section>
<section class='panel'><h2>Результаты по кварталам</h2><label for='blocks-search'>Поиск по любому столбцу</label><br><input id='blocks-search' type='search' placeholder='ID, морфотип, показатель…'>{block_table}
<details><summary>Школы: спрос, вместимость и обеспеченность (все кварталы)</summary><label for='schools-search'>Поиск по таблице школ</label><br><input id='schools-search' type='search' placeholder='Поиск…'>{school_table}</details></section>
<section class='panel'><h2>Источники и ограничения</h2><dl>{metadata}</dl><p>Источники: blocks_indicators.geojson/csv, provision_schools.csv, boundary.geojson и study_info.json из локального каталога результатов. Показатели переписаны из сохранённых данных без повторного анализа сети. Для карты используется EPSG:4326; расстояния и минуты относятся к сохранённому автомобильному сценарию, не к пешему маршруту. Нулевой спрос на школу не следует считать дефицитом; отношение суммарного обслуженного спроса к суммарному спросу рассчитано только при ненулевом знаменателе. Пропуски не включаются в медианы. Интерпретация результатов ограничена качеством и датой исходных данных; интерактивные базовые тайлы и библиотеки карты загружаются из интернета.</p></section>
<footer>Локальный HTML-отчёт · данные и графики встроены в файл</footer></main><script>
for(const [inputId,tableId] of [['blocks-search','blocks-table'],['schools-search','schools-table']]){{
 const input=document.getElementById(inputId), table=document.getElementById(tableId);
 input.addEventListener('input',()=>{{ const q=input.value.toLocaleLowerCase(); for(const row of table.tBodies[0].rows) row.hidden=!row.textContent.toLocaleLowerCase().includes(q); }});
 for(const [index,th] of [...table.tHead.rows[0].cells].entries()) th.addEventListener('click',()=>{{
  const asc=th.dataset.asc!=='true'; th.dataset.asc=String(asc);
  const value=row=>row.cells[index].textContent.trim(); const numeric=s=>Number(s.replace(/\\s/g,'').replace(',','.'));
  const rows=[...table.tBodies[0].rows]; rows.sort((a,b)=>{{const x=value(a),y=value(b); const comparison=x!==''&&y!==''&&Number.isFinite(numeric(x))&&Number.isFinite(numeric(y))?numeric(x)-numeric(y):x.localeCompare(y,'ru',{{numeric:true}});return asc?comparison:-comparison}});
  table.tBodies[0].append(...rows);
 }});
}}
</script></body></html>"""


def generate(data_dir=DATA_DIR, output=OUTPUT, overwrite=False):
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Файл уже существует: {output}. Для замены укажите --overwrite")
    document = render(Path(data_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents a race when --overwrite was not requested.
    with output.open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(document)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(f"Отчёт сохранён: {generate(args.data_dir, args.output, args.overwrite)}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
