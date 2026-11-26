"""Evidence-driven animated scenes 07–10 of the decision-invariance film."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
from math import hypot, pi, sin
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SIZE = (1920, 1080)
BG = "#08131F"
PANEL = "#111F2C"
TEXT = "#F2F8F5"
MUTED = "#A7BBCB"
MINT = "#57E6D0"
AMBER = "#FFAA65"
VIOLET = "#B7AAFF"
OUTLINE = "#2A4355"
GRID = "#264050"
FAINT = "#6F8C9E"
NAVY = "#162A39"
DATA = Path(__file__).resolve().parents[2] / "crossing_results"


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    windows = Path(os.getenv("WINDIR", "C:/Windows")) / "Fonts"
    dejavu = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        windows / ("segoeuib.ttf" if bold else "segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu") / dejavu,
        dejavu,
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    raise RuntimeError("Segoe UI or DejaVu Sans is required for the Cyrillic scenes.")


@lru_cache(maxsize=None)
def read_evidence(name: str) -> dict:
    return json.loads((DATA / f"flat_boundary_{name}.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def rgb(color: str) -> tuple[int, int, int]:
    return tuple(bytes.fromhex(color.lstrip("#")))


def mix(low: str, high: str, amount: float) -> tuple[int, int, int]:
    return tuple(round(a + amount * (b - a)) for a, b in zip(rgb(low), rgb(high), strict=True))


def ramp(position: float, start: float, end: float) -> float:
    x = max(0.0, min(1.0, (position - start) / (end - start)))
    return x * x * (3 - 2 * x)


def txt(draw, xy, text, size=30, color=TEXT, bold=False, anchor="lt"):
    draw.text(xy, text, font=font(size, bold), fill=color, anchor=anchor)


def box(draw, bounds, *, fill=PANEL, outline=OUTLINE, radius=28, width=2):
    draw.rounded_rectangle(bounds, radius, fill=fill, outline=outline, width=width)


def dot(draw, x, y, radius, fill, outline=None, width=1):
    draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                 fill=fill, outline=outline, width=width)


def number(value: float, precision=2):
    return f"{value:.{precision}f}".replace(".", ",")


def arrow(draw, start, end, color, width=4, head=13):
    draw.line((start, end), fill=color, width=width)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = hypot(dx, dy)
    if length > head:
        ux, uy = dx / length, dy / length
        back = (end[0] - head * ux, end[1] - head * uy)
        draw.polygon((end, (back[0] + head * .51 * uy, back[1] - head * .51 * ux),
                      (back[0] - head * .51 * uy, back[1] + head * .51 * ux)), fill=color)


def dashed(draw, start, end, color, width=2, on=10, off=10):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    for index in range(0, int(length), on + off):
        stop = min(index + on, length)
        draw.line((start[0] + index * ux, start[1] + index * uy,
                   start[0] + stop * ux, start[1] + stop * uy), fill=color, width=width)


HEADINGS = {
    "07": ("КАК ПРОВЕРИТЬ", "Можно ли проверить заранее?",
           "Смотрим, может ли траектория дойти до границы.",
           "Схема метода · локальный сертификат для плоских границ"),
    "08": ("ПРОВЕРКА НА ДАННЫХ", "Сколько удалось подтвердить?",
           "Покрытие — доля запросов, прошедших проверку на сохранение класса.",
           "Публичные выборки · один прототип на класс · численное применение сертификатов"),
    "09": ("ИНТЕРВАЛЬНАЯ ПРОВЕРКА", "Финиш и маршрут — разные вопросы",
           "17 стартовых точек на трёх горизонтах дают 51 проверку.",
           "Два двумерных примера · точные входы · интервальная арифметика"),
    "10": ("ВЫЧИСЛИТЕЛЬНАЯ ЦЕНА", "Больше гарантий ≠ быстрее",
           "1× — время полного расчёта всех траекторий.",
           "Замеры на одной машине · один поток · подготовка геометрии вне замера"),
}


@lru_cache(maxsize=4)
def scene_base(scene: str) -> Image.Image:
    frame = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(frame)
    for y in range(SIZE[1]):
        draw.line((0, y, SIZE[0], y), fill=mix(BG, "#102435", 0.38 * y / SIZE[1]))
    for x in range(1352, 1850, 24):
        for y in range(33, 238, 24):
            dot(draw, x, y, 1, "#24394B")
    kicker, title, subtitle, footnote = HEADINGS[scene]
    draw.line((96, 66, 140, 66), fill=MINT, width=4)
    txt(draw, (157, 56), kicker, 24, MINT, True)
    txt(draw, (96, 102), title, 74, TEXT, True)
    txt(draw, (98, 204), subtitle, 35, MUTED)
    txt(draw, (96, 973), footnote, 25, MUTED)
    draw.line((96, 1034, 1824, 1034), fill=OUTLINE, width=2)
    return frame


def bezier(anchors, t):
    return tuple((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b +
                 3 * (1 - t) * t ** 2 * c + t ** 3 * d
                 for a, b, c, d in zip(*anchors, strict=True))


def convex_outline(points):
    def half(ordered):
        boundary = []
        for x, y in ordered:
            while len(boundary) > 1:
                ax, ay = boundary[-2]
                bx, by = boundary[-1]
                if (bx - ax) * (y - ay) > (by - ay) * (x - ax):
                    break
                boundary.pop()
            boundary.append((x, y))
        return boundary[:-1]

    ordered = sorted(points)
    return tuple(half(ordered) + half(reversed(ordered)))


@lru_cache(maxsize=1)
def tube_geometry():
    path = tuple(bezier(((1067, 479), (850, 479), (909, 689), (641, 706)), i / 40) for i in range(41))
    upper, lower = [], []
    for i, (x, y) in enumerate(path):
        before, after = path[max(i - 1, 0)], path[min(i + 1, 40)]
        dx, dy = after[0] - before[0], after[1] - before[1]
        width = 27 + 29 * sin(pi * i / 54) ** 2
        nx, ny = -dy / hypot(dx, dy) * width, dx / hypot(dx, dy) * width
        upper.append((x + nx, y + ny))
        lower.append((x - nx, y - ny))
    start, end = path[0], path[-1]
    return path, tuple(upper + list(reversed(lower))), start, end


@lru_cache(maxsize=1)
def guard_base():
    frame = scene_base("07").copy()
    draw = ImageDraw.Draw(frame)
    box(draw, (96, 270, 1200, 909))
    box(draw, (1232, 270, 1824, 909))
    txt(draw, (134, 303), "ПЛОСКАЯ ГРАНИЦА КЛАССОВ", 25, TEXT, True)
    txt(draw, (1270, 303), "ЛОКАЛЬНАЯ ПРОВЕРКА", 25, TEXT, True)
    for x in range(151, 1168, 47):
        draw.line((x, 365, x, 856), fill="#1D3242")
    for y in range(380, 857, 47):
        draw.line((143, y, 1156, y), fill="#1D3242")
    draw.rectangle((142, 360, 429, 853), fill="#192333")
    draw.rectangle((433, 360, 1155, 853), outline="#264553", width=1)
    dashed(draw, (431, 351), (431, 857), VIOLET, width=5, on=15, off=13)
    txt(draw, (173, 817), "Класс A", 28, VIOLET, True)
    txt(draw, (1007, 817), "Класс Б", 28, MINT, True)
    return frame


def guard(t):
    frame = guard_base().copy()
    draw = ImageDraw.Draw(frame)
    path, shape, start, end = tube_geometry()
    path_reveal = ramp(t, .025, .202)
    tube_reveal = ramp(t, .228, .37)
    hull_reveal = ramp(t, .34, .495)
    pass_reveal = ramp(t, .50, .62)
    abstain_reveal = ramp(t, .72, .84)

    if tube_reveal:
        fill = mix(PANEL, "#173F47", tube_reveal)
        line = mix(PANEL, "#50BAAE", tube_reveal)
        draw.polygon(shape, fill=fill)
        draw.line((*shape, shape[0]), fill=line, width=4, joint="curve")
    if abstain_reveal:
        # The same centreline can have a conservative tube that touches a
        # boundary. Contact of an enclosure is only a reason to abstain.
        shifted = tuple((x - 277 * abstain_reveal * max(0.0, 1 - ((x - 630) / 340) ** 2), y)
                        for x, y in shape)
        grown = convex_outline((*shape, *shifted))
        draw.polygon(grown, fill=mix("#193E44", "#352E31", abstain_reveal))
        draw.line((*grown, grown[0]), fill=mix("#193E44", AMBER, abstain_reveal * .82), width=4,
                  joint="curve")
        draw.line((*shape, shape[0]), fill=mix("#50BAAE", "#396264", abstain_reveal), width=3,
                  joint="curve")
        dashed(draw, (431, 351), (431, 857), VIOLET, width=5, on=15, off=13)
    if path_reveal:
        visible_path = path[:max(2, round(41 * path_reveal))]
        draw.line(visible_path, fill=MINT, width=5, joint="curve")
        dot(draw, *start, 11, TEXT, outline=MINT, width=3)
        txt(draw, (1065, 441), "СТАРТ", 23, TEXT, True, "mt")
        dot(draw, *visible_path[-1], 11, MINT, outline=TEXT, width=2)
    if tube_reveal > .7:
        tube_label = ("Более широкая трубка", AMBER) if abstain_reveal > .7 else (
            "Трубка возможных положений", MINT)
        txt(draw, (704, 752), tube_label[0], 25, tube_label[1], True)
    if hull_reveal:
        low, high = min(p[1] for p in shape), max(p[1] for p in shape)
        a0 = 352 + (low - 352) * hull_reveal
        a1 = 857 + (high - 857) * hull_reveal
        for index in (4, 17, 30, 39):
            x, y = path[index]
            target_x = x - (x - 434) * hull_reveal
            if target_x < x - 18:
                arrow(draw, (x - 10, y), (target_x, y), "#54869D", 2, 11)
        draw.line((430, a0, 430, a1), fill=MINT, width=8)
        draw.line((407, a0, 453, a0), fill=MINT, width=4)
        draw.line((407, a1, 453, a1), fill=MINT, width=4)
        txt(draw, (163, 501), "Достижимая", 27, MINT, True)
        txt(draw, (163, 538), "проекция", 27, MINT, True)
    if hull_reveal < 0.7:
        txt(draw, (1270, 380), "Оценка всей грани", 38, TEXT, True)
        txt(draw, (1270, 468), "В неё попадают и дальние", 30, MUTED)
        txt(draw, (1270, 510), "участки линии.", 30, MUTED)
    else:
        txt(draw, (1270, 379), "Берём только область,", 37, TEXT, True)
        txt(draw, (1270, 430), "достижимую отсюда.", 37, TEXT, True)
        txt(draw, (1270, 505), "Проверяем запас до границы.", 29, MUTED)

    if pass_reveal:
        good = mix(MINT, "#61958E", .8 * abstain_reveal)
        box(draw, (1270, 594, 1785, 687), fill=mix("#143534", PANEL, abstain_reveal * .8),
            outline=mix(PANEL, MINT, pass_reveal * (.66 - .42 * abstain_reveal)), radius=20)
        draw.line((1298, 641, 1309, 653, 1333, 626), fill=good, width=5, joint="curve")
        txt(draw, (1357, 608), "Класс сохранится", 33, good, True)
        txt(draw, (1357, 651), "Узкая трубка вне границы", 24, mix(TEXT, MUTED, abstain_reveal))
    if abstain_reveal:
        box(draw, (1270, 716, 1785, 829), fill="#2A292A", outline=mix(PANEL, AMBER, abstain_reveal * .68), radius=20)
        dot(draw, 1317, 770, 19, "#3D332B", AMBER, 2)
        txt(draw, (1317, 754), "?", 27, AMBER, True, "mt")
        txt(draw, (1357, 733), "Оценка воздержалась", 32, AMBER, True)
        txt(draw, (1357, 777), "Широкая трубка задела грань", 24, TEXT)
    return frame


@lru_cache(maxsize=1)
def coverage_evidence():
    data = read_evidence("transfer")
    result = []
    for key, label, unit in (("iris", "Iris", "объектов"),
                             ("letter", "Letter", "образов букв"),
                             ("har", "HAR", "окна движений")):
        row, = [item for item in data["results"][key]["overall"]
                if item["ratio"] == .35 and item["horizon"] == .50]
        result.append((label, unit, row["counts"]["n"], row["covariance_stable_coverage"],
                       row["local_flat_stable_coverage"]))
    return tuple(result)


@lru_cache(maxsize=1)
def coverage_base():
    frame = scene_base("08").copy()
    draw = ImageDraw.Draw(frame)
    box(draw, (96, 270, 1824, 918))
    txt(draw, (134, 307), "СРАВНЕНИЕ СЕРТИФИКАТОВ", 25, TEXT, True)
    txt(draw, (1770, 307), "T = 0,50     σ/D = 0,35", 26, MUTED, True, "rt")
    for x, name in ((655, "Ковариационная оценка"), (1127, "Локальная оценка")):
        swatch_color = FAINT if x == 655 else MINT
        box(draw, (x, 361, x + 30, 380), fill=swatch_color, outline=swatch_color, radius=4, width=1)
        txt(draw, (x + 46, 352), name, 27, TEXT)
    left, right = 469, 1587
    for value in range(0, 101, 25):
        x = left + (right - left) * value / 100
        draw.line((x, 414, x, 829), fill=GRID, width=2)
        txt(draw, (x, 850), f"{value}%", 29, MUTED, anchor="mt")
    for (label, unit, count, _, _), y in zip(coverage_evidence(), (464, 591, 718), strict=True):
        txt(draw, (145, y - 6), label, 42, TEXT, True)
        txt(draw, (145, y + 47), f"{count:,} {unit}".replace(",", " "), 27, MUTED)
        for dy in (0, 47):
            box(draw, (left, y + dy, right, y + dy + 30), fill=NAVY,
                outline=NAVY, radius=7, width=1)
    return frame


def coverage(t):
    frame = coverage_base().copy()
    draw = ImageDraw.Draw(frame)
    left, right = 469, 1587
    for i, ((_, _, _, old, local), y) in enumerate(zip(coverage_evidence(), (464, 591, 718), strict=True)):
        old_reveal = ramp(t, .11 + .018 * i, .21 + .018 * i)
        local_start = (.258, .368, .507)[i]
        local_reveal = ramp(t, local_start, local_start + .09)
        for dy, value, reveal, color in ((0, old, old_reveal, FAINT), (47, local, local_reveal, MINT)):
            if not reveal:
                continue
            tip = left + (right - left) * value * reveal
            if tip - left > 4:
                box(draw, (left, y + dy, tip, y + dy + 30), fill=color, outline=color, radius=7, width=1)
            if reveal > .84:
                txt(draw, (tip + 18, y + dy - 3), number(value * 100) + " %", 30, color, True)
        if i in (1, 2) and t > local_start + .09:
            x0, x1 = left + (right - left) * old, left + (right - left) * local
            draw.line((x0, y + 97, x1, y + 97), fill="#367B77", width=2)
            draw.line((x0, y + 91, x0, y + 103), fill="#367B77", width=2)
            draw.line((x1, y + 91, x1, y + 103), fill="#367B77", width=2)
    if t > .76:
        txt(draw, (1783, 973), "864 окна HAR от 9 участников", 25, MUTED, anchor="rt")
    return frame


@lru_cache(maxsize=1)
def interval_evidence():
    data = read_evidence("interval")
    keys = ("interval_endpoint", "interval_whole_path")
    rows = tuple(tuple(item for name in ("centered_cospherical", "reflective_rectangle")
                       for item in data["observations"][name] if item["horizon"] == horizon)
                 for horizon in data["horizons"])
    counts = tuple(Counter(item[key] for row in rows for item in row) for key in keys)
    if any(len(row) != 17 for row in rows) or any(sum(values.values()) != 51 for values in counts):
        raise ValueError("The interval display requires 17 starts on each of three horizons.")
    return tuple(data["horizons"]), rows, keys, counts


def paint_status(draw, x, y, status, amount):
    palette = {"same": MINT, "stable": MINT, "changed": AMBER, "departure": AMBER, "abstain": VIOLET}
    edge = palette[status]
    color = mix(PANEL, edge, amount)
    fill = mix(PANEL, edge, .18 * amount)
    box(draw, (x, y, x + 33, y + 40), fill=fill, outline=color, radius=8, width=2)
    if status == "abstain":
        draw.line((x + 10, y + 28, x + 23, y + 12), fill=color, width=3)
        draw.line((x + 15, y + 31, x + 25, y + 19), fill=color, width=2)
    elif status in ("departure", "changed"):
        arrow(draw, (x + 8, y + 20), (x + 25, y + 20), color, width=2, head=7)
    else:
        dot(draw, x + 16, y + 20, 5, color)


@lru_cache(maxsize=1)
def interval_base():
    frame = scene_base("09").copy()
    draw = ImageDraw.Draw(frame)
    for x, title, short in ((96, "Финиш", "Какой класс в конечной точке?"),
                            (976, "Весь маршрут", "Покидала ли точка исходный класс?")):
        box(draw, (x, 270, x + 848, 918))
        txt(draw, (x + 38, 310), title, 53, TEXT, True)
        txt(draw, (x + 40, 385), short, 31, MUTED)
        draw.line((x + 39, 446, x + 809, 446), fill=GRID, width=2)
        txt(draw, (x + 129, 474), "Шесть центров", 22, FAINT, True)
        txt(draw, (x + 471, 474), "Прямоугольник", 22, FAINT, True)
        dashed(draw, (x + 444, 510), (x + 444, 724), GRID, width=2, on=6, off=8)
        for horizon, y in zip(interval_evidence()[0], (528, 600, 672), strict=True):
            txt(draw, (x + 39, y + 5), "T " + horizon.replace(".", ","), 24, MUTED)
            for i in range(17):
                xi = x + 128 + 40 * i + 14 * (i >= 8)
                box(draw, (xi, y, xi + 33, y + 40), fill=NAVY, outline=NAVY, radius=8, width=1)
        draw.line((x + 39, 748, x + 809, 748), fill=GRID, width=2)
    return frame


def interval(t):
    frame = interval_base().copy()
    draw = ImageDraw.Draw(frame)
    _, rows, keys, counts = interval_evidence()
    for j, (x, key, count) in enumerate(zip((96, 976), keys, counts, strict=True)):
        tick = .255 if j == 0 else .505
        for r, (y, entries) in enumerate(zip((528, 600, 672), rows, strict=True)):
            for i, item in enumerate(entries):
                onset = tick + .0042 * (r * 17 + i)
                opacity = ramp(t, onset, onset + .055)
                if opacity:
                    xi = x + 128 + 40 * i + 14 * (i >= 8)
                    paint_status(draw, xi, y, item[key], opacity)
        if j == 0:
            legend = (("same", "та же метка", MINT), ("changed", "другая метка", AMBER))
            spacing = 384
        else:
            legend = (("stable", "сохранений", MINT), ("departure", "выходов", AMBER),
                      ("abstain", "не определены", VIOLET))
            spacing = 254
        if t >= tick + .21:
            for i, (status, caption, color) in enumerate(legend):
                location = x + 41 + i * spacing
                txt(draw, (location, 779), str(count[status]), 52, color, True)
                txt(draw, (location, 849), caption, 27, TEXT)
    if t > .79:
        txt(draw, (1814, 973), "Один квадрат — одна пара «старт × горизонт»", 25, MUTED, anchor="rt")
    return frame


@lru_cache(maxsize=1)
def timing_evidence():
    timing = read_evidence("timings")["timings"]
    scaling = read_evidence("scaling")["overall"]
    observed = tuple((name, timing[key]["hybrid_to_rk4_ratio"])
                     for name, key in (("Iris", "iris"), ("Letter", "letter"), ("HAR", "har")))
    ranges = []
    for k in (3, 40):
        group = [row["median_hybrid_to_rk4_ratio"] for row in scaling if row["classes"] == k]
        if len(group) != 4:
            raise ValueError(f"Expected both dimensions and regimes at K={k}.")
        ranges.append((k, min(group), max(group)))
    return observed, tuple(ranges)


def ratio_axis(draw, left, right, y0, y1):
    for ratio in (0, 1, 2):
        x = left + (right - left) * ratio / 2.64
        if ratio == 1:
            dashed(draw, (x, y0), (x, y1), VIOLET, width=3, on=11, off=8)
        else:
            draw.line((x, y0, x, y1), fill=GRID, width=2)
        txt(draw, (x, y1 + 21), f"{ratio}×", 28, VIOLET if ratio == 1 else MUTED, True, "mt")


@lru_cache(maxsize=1)
def timing_base():
    frame = scene_base("10").copy()
    draw = ImageDraw.Draw(frame)
    box(draw, (96, 270, 944, 918))
    box(draw, (976, 270, 1824, 918))
    txt(draw, (134, 309), "Публичные данные", 45, TEXT, True)
    txt(draw, (136, 373), "Первый тестовый блок каждого набора", 27, MUTED)
    txt(draw, (1014, 309), "Число классов", 45, TEXT, True)
    txt(draw, (1016, 373), "Синтетический тест масштабирования", 27, MUTED)
    ratio_axis(draw, 302, 862, 433, 771)
    ratio_axis(draw, 1193, 1753, 433, 771)
    for (name, _), y in zip(timing_evidence()[0], (486, 585, 684), strict=True):
        txt(draw, (137, y - 2), name, 36, TEXT, True)
        box(draw, (302, y + 3, 862, y + 29), fill=NAVY, outline=NAVY, radius=7, width=1)
    for (k, _, _), y in zip(timing_evidence()[1], (527, 677), strict=True):
        txt(draw, (1014, y - 22), str(k), 63, TEXT, True)
        txt(draw, (1016, y + 49), "класса" if k == 3 else "классов", 27, MUTED)
    txt(draw, (136, 856), "Время гибрида / время полного RK4", 27, MUTED)
    txt(draw, (1016, 856), "Диапазон по размерностям и режимам", 27, MUTED)
    return frame


def timings(t):
    frame = timing_base().copy()
    draw = ImageDraw.Draw(frame)
    observed, ranges = timing_evidence()
    for i, ((_, ratio), y) in enumerate(zip(observed, (486, 585, 684), strict=True)):
        start = .259 + .072 * i
        reveal = ramp(t, start, start + .066)
        if reveal:
            edge = MINT if ratio <= 1 else AMBER
            tip = 302 + 560 * ratio / 2.64 * reveal
            if tip > 308:
                box(draw, (302, y + 3, tip, y + 29), fill=edge, outline=edge, radius=7, width=1)
            if reveal > .82:
                txt(draw, (tip + 15, y - 9), number(ratio) + "×", 36, edge, True)
    for i, ((_, lo, hi), y) in enumerate(zip(ranges, (527, 677), strict=True)):
        start = .56 + .096 * i
        reveal = ramp(t, start, start + .086)
        if not reveal:
            continue
        edge = MINT if hi < 1 else AMBER
        lhs = 1193 + 560 * lo / 2.64
        rhs = lhs + 560 * (hi - lo) / 2.64 * reveal
        draw.line((lhs, y, rhs, y), fill=edge, width=13)
        dot(draw, lhs, y, 9, edge, TEXT, 2)
        dot(draw, rhs, y, 9, edge, TEXT, 2)
        if reveal > .82:
            precision = 3 if i == 0 else 2
            txt(draw, (lhs, y - 59), f"{number(lo, precision)}–{number(hi, precision)}×",
                33, edge, True)
    return frame


SCENES = {"07": guard, "08": coverage, "09": interval, "10": timings}


def render_data_scene(scene_id: str, seconds: float, duration: float,
                      size: tuple[int, int] = SIZE) -> Image.Image:
    """Render one scientific panel at a time tied to its narration beats."""
    if duration <= 0:
        raise ValueError("A scene needs a positive duration.")
    position = min(1.0, max(0.0, seconds / duration))
    frame = SCENES[scene_id](position)
    draw = ImageDraw.Draw(frame)
    if position > 0:
        draw.line((96, 1034, 96 + 1728 * position, 1034), fill=MINT, width=3)
    return frame if size == SIZE else frame.resize(size, Image.Resampling.LANCZOS)
