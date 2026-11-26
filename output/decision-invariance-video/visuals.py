"""Animated geometry for the decision-invariance episode.

Coordinates in the numerical panels are reproduced from paper/flat_boundary.tex
and tools/plot_flat_boundary.py. The water and rotating-disk panels are labelled
illustrations. All frames depend only on scene-relative time.
"""

from functools import lru_cache
from math import cos, hypot, pi, sin
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.integrate import solve_ivp


WIDTH, HEIGHT = 1920, 1080
BG = "#08131F"
PANEL = "#111F2C"
TEXT = "#F2F8F5"
MUTED = "#A7BBCB"
MINT = "#57E6D0"
AMBER = "#FFAA65"
VIOLET = "#B7AAFF"
EDGE = "#2A4357"
GRID = "#203749"
INK = "#0C1927"
DARK_MINT = "#103937"
DARK_AMBER = "#3B2929"

COSPHERICAL = np.array([(1., 0.), (.6, .8), (0., 1.), (-1., 0.), (-.6, -.8), (0., -1.)])
RECTANGLE = np.array([(-1., -.75), (1., -.75), (1., .75), (-1., .75)])
AXIAL = np.array([2., 1.]) / np.sqrt(5)
NORMAL = np.array([-1., 2.]) / np.sqrt(5)
START = 1.45 * AXIAL - .10 * NORMAL
HORIZON = 2.25


@lru_cache
def font(size, bold=False):
    windows = Path(os.getenv("WINDIR", "C:/Windows")) / "Fonts"
    names = ("segoeuib.ttf", "DejaVuSans-Bold.ttf") if bold else ("segoeui.ttf", "DejaVuSans.ttf")
    for path in (windows / names[0], Path("/usr/share/fonts/truetype/dejavu") / names[1], names[1]):
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            pass
    raise RuntimeError("Install Segoe UI or DejaVu Sans to render Cyrillic text")


def txt(draw, xy, text, size, color=TEXT, bold=False, anchor=None):
    draw.text(xy, text, fill=color, font=font(size, bold), anchor=anchor)


def panel(draw, box, fill=PANEL, outline=EDGE, r=28):
    draw.rounded_rectangle(box, r, fill=fill, outline=outline, width=2)


def dot(draw, xy, r, color, outline=None, width=1):
    x, y = xy
    draw.ellipse((x-r, y-r, x+r, y+r), fill=color, outline=outline, width=width)


def arrow(draw, start, end, color, width=4, head=14):
    x, y = end
    dx, dy = x-start[0], y-start[1]
    length = hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx/length, dy/length
    draw.line((start, end), fill=color, width=width)
    draw.polygon((end, (x-head*ux+head*.48*uy, y-head*uy-head*.48*ux),
                  (x-head*ux-head*.48*uy, y-head*uy+head*.48*ux)), fill=color)


def dashed(draw, start, end, color, width=2, dash=13, gap=10):
    a, b = np.array(start, dtype=np.float64), np.array(end, dtype=np.float64)
    d = b-a
    n = np.linalg.norm(d)
    if n == 0:
        return
    for i in range(0, int(n)+1, dash+gap):
        draw.line((tuple(a+i*d/n), tuple(a+min(i+dash, n)*d/n)), fill=color, width=width)


def soft(a, b, t):
    x = min(1., max(0., (t-a)/(b-a)))
    return x*x*(3.-2.*x)


def mix(a, b, alpha):
    channels = lambda s: [int(s[j:j+2], 16) for j in (1, 3, 5)]
    return "#{:02x}{:02x}{:02x}".format(*[round((1-alpha)*x+alpha*y) for x, y in zip(channels(a), channels(b), strict=True)])


def tag(draw, xy, text, color=MINT, background=DARK_MINT, size=24):
    x, y = xy
    w = draw.textlength(text, font=font(size, True))
    draw.rounded_rectangle((x, y, x+w+33, y+43), radius=14, fill=background)
    txt(draw, (x+16, y+20), text, size, color, True, "lm")


@lru_cache(maxsize=1)
def background():
    xx, yy = np.meshgrid(np.linspace(0, 1, WIDTH), np.linspace(0, 1, HEIGHT))
    haze = np.exp(-((xx-.79)**2/.12+(yy-.44)**2/.45))
    cool = np.exp(-((xx-.08)**2/.13+(yy-.15)**2/.15))
    pixels = np.stack((8+4*haze+3*cool, 19+13*haze+5*cool, 31+16*haze+8*cool), axis=-1)
    image = Image.fromarray(pixels.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for x in range(18, WIDTH, 42):
        for y in range(20, HEIGHT, 42):
            dot(draw, (x, y), 1, "#1C3447")
    return image


def begin(scene, label, title=None, line=None):
    image = background().copy()
    draw = ImageDraw.Draw(image)
    draw.line((96, 72, 137, 72), fill=MINT, width=4)
    txt(draw, (151, 69), f"{scene}   /   {label}", 24, MINT, True, "lm")
    if title:
        txt(draw, (95, 107), title, 72, TEXT, True)
    if line:
        txt(draw, (100, 195), line, 32, MUTED)
    return image, draw


def footer(draw, caption, p):
    txt(draw, (98, 985), caption, 24, MUTED, anchor="lm")
    draw.line((98, 1034, 1823, 1034), fill=EDGE, width=3)
    draw.line((98, 1034, 98+1725*p, 1034), fill=MINT, width=3)


@lru_cache(maxsize=4)
def solution(name):
    sites, beta, start = (
        (COSPHERICAL, 1.2, START) if name == "circle"
        else (RECTANGLE, 2., np.array([1.39, .49]))
    )
    offsets = np.sum(sites*sites, axis=1) / 2

    def field(_t, x):
        logits = (sites @ x - offsets) / beta
        responsibilities = np.exp(logits - logits.max())
        return responsibilities @ sites / responsibilities.sum() - x

    def boundary(_t, x):
        return NORMAL @ x

    out = solve_ivp(field, (0., HORIZON), start, rtol=2e-12, atol=2e-14,
                    dense_output=True, method="DOP853",
                    events=boundary if name == "circle" else None)
    if not out.success:
        raise RuntimeError(f"Cannot render {name} trajectory")
    cross = (float(out.t_events[0][0]), tuple(out.y_events[0][0])) if name == "circle" else None
    values = out.sol(np.linspace(0, HORIZON, 351)).T
    if name == "circle" and (COSPHERICAL @ values[0]).argmax() == (COSPHERICAL @ values[-1]).argmax():
        raise RuntimeError("The published trajectory must cross between strict cells")
    return values, cross


def clip(poly, vector, offset):
    """Cut a convex polygon by vector·x <= offset."""
    out = []
    if not poly:
        return out
    for previous, current in zip(poly[-1:]+poly[:-1], poly, strict=True):
        z = float(vector @ previous - offset)
        z_next = float(vector @ current - offset)
        if (z <= 0) != (z_next <= 0):
            out.append(previous+(current-previous)*z/(z-z_next))
        if z_next <= 0:
            out.append(current)
    return out


def coordinates(box, extent):
    xmin, ymin, xmax, ymax = extent
    x0, y0, x1, y1 = box
    scale = min((x1-x0)/(xmax-xmin), (y1-y0)/(ymax-ymin))
    cx = .5*(x0+x1) - .5*scale*(xmax+xmin)
    cy = .5*(y0+y1) + .5*scale*(ymax+ymin)
    return lambda xy: (round(cx+xy[0]*scale), round(cy-xy[1]*scale)), scale


@lru_cache(maxsize=16)
def map_image(name, w, h, extent, field_arrows=False):
    sites = COSPHERICAL if name == "circle" else RECTANGLE
    beta = 1.2 if name == "circle" else 2.
    paint = ("#39292D", "#163B3F", "#222B44", "#172A3D", "#1B2D44", "#212A3C") if name == "circle" else (
        "#212941", "#39292D", "#15393C", "#1C3144")
    image = Image.new("RGB", (w, h), PANEL)
    draw = ImageDraw.Draw(image)
    project, scale = coordinates((0, 0, w-1, h-1), extent)
    xmin, ymin, xmax, ymax = extent
    outer = [np.array(p) for p in [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]]
    polygons = []
    for i, a in enumerate(sites):
        poly = outer.copy()
        for j, b in enumerate(sites):
            if i != j:
                poly = clip(poly, b-a, (b @ b - a @ a)/2)
        pixels = [project(p) for p in poly]
        draw.polygon(pixels, fill=paint[i])
        polygons.append(pixels)
    for x in np.arange(np.ceil(2*xmin)/2, xmax+.01, .5):
        draw.line((project((x, ymin)), project((x, ymax))), fill="#2C4252", width=1)
    for y in np.arange(np.ceil(2*ymin)/2, ymax+.01, .5):
        draw.line((project((xmin, y)), project((xmax, y))), fill="#2C4252", width=1)
    for pixels in polygons:
        draw.line([*pixels, pixels[0]], fill="#436173", width=2)
    if name == "circle":
        draw.line((project((0, 0)), project((3, 1.5))), fill="#A1D2D4", width=3)
        x, y = project((0, 0))
        r = scale
        draw.ellipse((x-r, y-r, x+r, y+r), outline="#476678", width=2)
    if field_arrows:
        offsets = np.sum(sites*sites, axis=1) / 2
        for x in np.arange(xmin+.17, xmax-.13, .28):
            for y in np.arange(ymin+.18, ymax-.13, .28):
                z = np.array((x, y))
                responsibilities = np.exp((sites @ z - offsets) / beta)
                drift = responsibilities @ sites / responsibilities.sum() - z
                if np.linalg.norm(drift) > .045:
                    delta = .10*drift / max(.42, np.linalg.norm(drift))
                    arrow(draw, project(z-delta/4), project(z+delta), "#59788E", 2, 7)
    for i, site in enumerate(sites):
        color = AMBER if (name == "circle" and i == 0) or (name == "rectangle" and i == 1) else (
            MINT if (name == "circle" and i == 1) or (name == "rectangle" and i == 2) else "#A1AFCB"
        )
        dot(draw, project(site), 11, color, INK, 3)
    x, y = project((0, 0))
    draw.line((x-9, y, x+9, y), fill=TEXT, width=3)
    draw.line((x, y-9, x, y+9), fill=TEXT, width=3)
    return image


def trajectory_panel(image, box, extent, name, t, full=False, field=False):
    x0, y0, x1, y1 = box
    image.paste(map_image(name, x1-x0+1, y1-y0+1, extent, field), (x0, y0))
    draw = ImageDraw.Draw(image)
    project, _ = coordinates(box, extent)
    values, cross = solution(name)
    index = min(len(values)-1, int(round(t/HORIZON*(len(values)-1))))
    highlight = MINT if cross is None or t >= cross[0] else AMBER
    pixels = [project(value) for value in values[:index+1]]
    if full:
        draw.line([project(v) for v in values], fill="#79675F", width=4, joint="curve")
    if len(pixels) > 1:
        draw.line(pixels, fill="#08131F", width=11, joint="curve")
        draw.line(pixels, fill="#FFE1B4", width=4, joint="curve")
    dot(draw, pixels[0], 8, INK, "#FFE1B4", 3)
    if cross and t >= cross[0]:
        x, y = project(cross[1])
        radius = 12 + max(0, 1-abs(t-cross[0])/.45)*12
        dot(draw, (x, y), radius, None, MINT, 2)
    dot(draw, pixels[-1], 12, highlight, TEXT, 3)
    draw.rectangle(box, outline=EDGE, width=2)
    return draw, project, highlight


def scene01(p):
    image, draw = begin("01", "ГАУССОВЫ ПОТОКИ")
    t = HORIZON * soft(.08, .60, p)
    t_cross, hit = solution("circle")[1]
    box = (871, 101, 1818, 866)
    extent = (-1.38, -1.24, 1.69, 1.24)
    draw, project, highlight = trajectory_panel(image, box, extent, "circle", t, True, False)
    txt(draw, (91, 173), "ОДНА", 116, TEXT, True)
    txt(draw, (91, 292), "ТОЧКА.", 116, TEXT, True)
    txt(draw, (91, 440), "ДВА", 116, highlight, True)
    txt(draw, (92, 559), "РЕШЕНИЯ.", 107, highlight, True)
    tag(draw, (103, 752), "Метка А" if t < t_cross else "Метка Б", highlight,
        DARK_AMBER if t < t_cross else DARK_MINT, size=29)
    tag(draw, (895, 124), "ТРАЕКТОРИЯ ИЗ СТАТЬИ", TEXT, INK)
    txt(draw, (1795, 838), f"t = {t:.2f}".replace(".", ","), 30, TEXT, True, "rm")
    if t >= t_cross:
        x, y = project(hit)
        label = (x-257, y-107)
        arrow(draw, (x-51, y-76), (x-13, y-13), MINT, 3, 11)
        tag(draw, label, "Смена метки", MINT, INK, size=24)
    footer(draw, "Численная траектория · шесть равных прототипов · β = 1,2", p)
    return image


def analogy_map(draw, box, traveling, p):
    x0, y0, x1, y1 = box
    split = (x0+x1)//2
    draw.rectangle((x0, y0, split, y1), fill="#372A33")
    draw.rectangle((split, y0, x1, y1), fill="#173943")
    for x in range(x0+20, x1-20, 58):
        draw.line((x, y0, x, y1), fill=GRID, width=1)
    for y in range(y0+20, y1-20, 50):
        draw.line((x0, y, x1, y), fill=GRID, width=1)
    dashed(draw, (split, y0), (split, y1), "#ACCFD6", width=3, dash=15, gap=9)
    for cx, icon, tone in ((x0+112, "А", AMBER), (x1-112, "Б", MINT)):
        dot(draw, (cx, y0+120), 43, INK, tone, 3)
        txt(draw, (cx, y0+120), icon, 42, tone, True, "mm")
    start = (x0+152, y0+312)
    def at(k):
        return start[0]+k*(x1-x0-282), start[1]-58*sin(.83*pi*k)-46*k
    if traveling:
        for j in range(3):
            path = [(x0+38+i*(x1-x0-76)/100,
                     y0+246+54*j+22*sin(i*.086+p*3.1+j)) for i in range(101)]
            draw.line(path, fill="#3F7484", width=2)
            arrow(draw, path[-13], path[-1], "#3F7484", 2, 10)
        progress = soft(.21, .79, p)
        if progress > 0:
            points = [at(i*progress/60) for i in range(61)]
            draw.line(points, fill="#F1D2AE", width=4, joint="curve")
        point = at(progress)
    else:
        point = start
    color = MINT if point[0] > split else AMBER
    dot(draw, start, 11, INK, AMBER, 2)
    dot(draw, point, 16, color, TEXT, 3)
    return color


def scene02(p):
    image, draw = begin("02", "АНАЛОГИЯ", "Карта и течение",
                        "Одна карта районов. Два способа поставить штамп.")
    for x, heading, travel in ((97, "Проверить на месте", False), (981, "Сначала дать течению унести", True)):
        card = (x, 273, x+842, 910)
        panel(draw, card)
        txt(draw, (x+27, 297), heading, 38, TEXT, True)
        color = analogy_map(draw, (x+28, 364, x+814, 781), travel, p)
        tag(draw, (x+27, 823), "Штамп Б" if color == MINT else "Штамп А",
            color, DARK_MINT if color == MINT else DARK_AMBER, size=27)
        txt(draw, (x+810, 844), "Границы те же", 27, MUTED, anchor="rm")
    footer(draw, "Схема-аналогия · А и Б — ближайшие пункты · течение показано условно", p)
    return image


def scene03(p):
    image, draw = begin("03", "КОНТРПРИМЕР", "Общий центр не спасает",
                        "В обеих картах траектории тянутся к общему равновесию.")
    extent = (-1.6, -1.12, 1.7, 1.12)
    for x, name, heading, subtitle in (
        (96, "rectangle", "Симметричный прямоугольник", "Равные центры · β = 2"),
        (981, "circle", "Шесть центров на окружности", "Равные центры · β = 1,2"),
    ):
        t = HORIZON * (soft(.10, .43, p) if name == "rectangle" else soft(.49, .90, p))
        panel(draw, (x, 273, x+842, 925))
        txt(draw, (x+23, 295), heading, 36, TEXT, True)
        txt(draw, (x+25, 348), subtitle, 24, MUTED)
        draw, project, _ = trajectory_panel(image, (x+25, 392, x+817, 875), extent, name, t, False, True)
        proof = name == "rectangle"
        crossed = not proof and t >= solution("circle")[1][0]
        ribbon = "Грани сохраняются" if proof else "Траектория пересекает грань" if crossed else "Один общий центр"
        tone = MINT if proof else AMBER
        tag(draw, (x+43, 815), ribbon, tone, INK, size=24)
        cx, cy = project((0, 0))
        txt(draw, (cx+18, cy+13), "равновесие", 24, TEXT, True)
    footer(draw, "Сравнение численных траекторий · стрелки показывают направления поля", p)
    return image


def scene04(p):
    image, draw = begin("04", "МЕХАНИЗМ", "Что держит границу",
                        "Складываем вклады по нормали внутри каждой группы проекций.")
    panel(draw, (97, 281, 1258, 915))
    panel(draw, (1283, 281, 1823, 915))
    y0 = 590
    for x in range(130, 1250, 41):
        draw.line((x, 350, x, 856), fill=GRID, width=1)
    for y in range(353, 861, 40):
        draw.line((130, y, 1225, y), fill=GRID, width=1)
    draw.line((131, y0, 1215, y0), fill="#D4E8E8", width=5)
    tag(draw, (145, 610), "Грань", TEXT, INK)
    arrow(draw, (168, y0-18), (168, y0-127), MUTED, 4, 14)
    txt(draw, (190, 489), "нормаль", 25, MUTED)
    phases = [
        (395, 127, 127, .02, "u₁"),
        (704, 94, 94, .24, "u₂"),
        (1007, 146, 0, .58, "u₃"),
    ]
    for x, up, down, reveal, label in phases:
        fade = soft(reveal, reveal+.08, p)
        if fade == 0:
            continue
        dashed(draw, (x, y0-up), (x, y0+down), mix(PANEL, "#9BB8BA", fade), 2)
        dot(draw, (x, y0), 8, TEXT, INK, 2)
        txt(draw, (x, 797), label, 30, TEXT, True, "mm")
        for sign, distance, tone in ((-1, up, MINT), (1, down, AMBER)):
            if distance:
                point = x, y0+sign*distance
                dot(draw, point, 23, mix(PANEL, tone, fade), TEXT, 2)
                arrow(draw, (x, y0+sign*11), (x, y0+sign*.72*distance*fade), tone, 5, 14)
        if down:
            tag(draw, (x-69, 842), "Сумма 0", MINT, DARK_MINT)
        else:
            tag(draw, (x-86, 842), "Сумма ≠ 0", AMBER, DARK_AMBER)
    txt(draw, (1312, 312), "Вес вклада", 41, TEXT, True)
    txt(draw, (1314, 388), "Знак", 39, MINT, True)
    txt(draw, (1314, 439), "сторона границы", 30, MUTED)
    txt(draw, (1314, 512), "Масса", 39, MINT, True)
    txt(draw, (1314, 563), "вес прототипа", 30, MUTED)
    txt(draw, (1314, 638), "Удалённость", 39, MINT, True)
    txt(draw, (1314, 689), "ослабление по Гауссу", 30, MUTED)
    if p >= .61:
        draw.line((1314, 772, 1787, 772), fill=EDGE, width=2)
        txt(draw, (1314, 802), "Где-то появляется", 34, AMBER, True)
        txt(draw, (1314, 848), "поперечный ход", 34, AMBER, True)
    footer(draw, "Схема проекций · баланс проверяется на каждой опорной плоскости", p)
    return image


def scene05(p):
    image, draw = begin("05", "ТЕОРЕМА", "Один горизонт, все горизонты",
                        "Проверка всех решений в одно время даёт глобальный результат.")
    panel(draw, (96, 281, 820, 927))
    panel(draw, (843, 281, 1823, 927))
    txt(draw, (123, 311), "Обычное вращение", 42, TEXT, True)
    tag(draw, (123, 372), "СХЕМА", AMBER, DARK_AMBER)
    center, radius = (459, 626), 181
    bounds = (center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius)
    draw.pieslice(bounds, 90, 270, fill=DARK_AMBER)
    draw.pieslice(bounds, -90, 90, fill=DARK_MINT)
    draw.ellipse(bounds, outline=EDGE, width=3)
    draw.line((459, 446, 459, 806), fill="#ACCFD6", width=3)
    dot(draw, center, 5, MUTED)
    phase = soft(.06, .35, p)
    angles = np.linspace(.45*pi, .45*pi+2*pi*phase, 100)
    points = [(459-141*cos(q), 626-141*sin(q)) for q in angles]
    draw.line(points, fill="#EDCAAA", width=4, joint="curve")
    dot(draw, points[0], 9, INK, TEXT, 2)
    dot(draw, points[-1], 13, AMBER if points[-1][0]<459 else MINT, TEXT, 3)
    txt(draw, (459, 855), "Промежуточные пересечения" if .15 < phase < .93 else "После оборота метка прежняя",
        28, MUTED, anchor="mm")

    txt(draw, (881, 311), "Гауссов градиентный поток", 45, TEXT, True)
    tag(draw, (881, 373), "ПЛОСКИЕ ГРАНИЦЫ", MINT, DARK_MINT)
    cards = [
        (449, .29, "Все строгие решения", "Сохранились при одном и том же T > 0"),
        (604, .49, "Баланс по нормали", "Во всех группах на каждой грани"),
        (759, .66, "Любой горизонт", "Решения сохраняются при всех t ≥ 0"),
    ]
    for y, start, heading, sub in cards:
        revealed = soft(start, start+.08, p)
        filled = mix(PANEL, "#183944", revealed)
        outline = mix(EDGE, MINT, revealed*.61)
        panel(draw, (879, y, 1784, y+121), filled, outline, 19)
        txt(draw, (904, y+13), heading, 38, mix(EDGE, TEXT, revealed), True)
        txt(draw, (906, y+70), sub, 29, mix(EDGE, MUTED, revealed))
    for y, t0 in ((586, .52), (741, .69)):
        if p >= t0:
            arrow(draw, (1320, y+10), (1320, y-10), MINT, 2, 7)
            arrow(draw, (1332, y-10), (1332, y+10), MINT, 2, 7)
    footer(draw, "Теорема относится ко всем строгим ячейкам фиксированного аффинного классификатора", p)
    return image


def anisotropy_map(draw, box, compensated, p):
    x0, y0, x1, y1 = box
    extent = (-1.65, -2.39, 1.65, 1.45)
    poly0 = [np.array(x) for x in [(extent[0], extent[1]), (extent[2], extent[1]), (extent[2], extent[3]), (extent[0], extent[3])]]
    project, _ = coordinates(box, extent)
    for sign, tone in ((1, DARK_MINT), (-1, DARK_AMBER)):
        shape = clip(poly0, sign*np.array((2, 1)), 0)
        screen = [project(xy) for xy in shape]
        draw.polygon(screen, fill=tone)
        draw.line(screen+[screen[0]], fill=EDGE, width=2)
    for value in (-1., 0., 1.):
        draw.line((project((value, extent[1])), project((value, extent[3]))), fill=GRID, width=2)
    for value in (-2., -1., 0., 1.):
        draw.line((project((extent[0], value)), project((extent[2], value))), fill=GRID, width=2)
    draw.line((project((-.725, 1.45)), project(((2.39/2), -2.39))), fill="#C4DEDE", width=4)
    draw.line((project((-.09, 0)), project((.09, 0))), fill=TEXT, width=2)
    draw.line((project((0, -.09)), project((0, .09))), fill=TEXT, width=2)
    for site, tone in ((np.array((-1, -1)), MINT), (np.array((1, 1)), AMBER)):
        dot(draw, project(site), 13, tone, TEXT, 2)
    point = np.array((1., -2.))
    velocity = np.array((-1., 2.)) if compensated else np.array((-2., 2.))
    color = MINT if compensated else AMBER
    end = point+.61*soft(.49 if compensated else .17, .78 if compensated else .50, p)*velocity
    arrow(draw, project(point), project(end), INK, 10, 27)
    arrow(draw, project(point), project(end), color, 6, 23)
    dot(draw, project(point), 13, TEXT, INK, 3)
    x, y = project(point)
    txt(draw, (x+22, y+8), "(1, −2)", 25, TEXT, True)
    txt(draw, (x0+28, y0+21), "2x + y = 0", 25, TEXT, True)


def scene06(p):
    image, draw = begin("06", "СМЕНА МАСШТАБА", "Одна граница, два движения",
                        "Общая анизотропия · вес оси x в метрике вдвое выше.")
    for x, heading, compensated, end in (
        (96, "Евклидов градиент", False, "(−2, 2)"),
        (981, "Предобусловленный поток", True, "(−1, 2)"),
    ):
        panel(draw, (x, 271, x+842, 927))
        txt(draw, (x+29, 296), heading, 40, TEXT, True)
        anisotropy_map(draw, (x+42, 367, x+798, 821), compensated, p)
        tone = MINT if compensated else AMBER
        tag(draw, (x+39, 857), "Вдоль границы" if compensated else "Поперёк границы",
            tone, DARK_MINT if compensated else DARK_AMBER)
        txt(draw, (x+804, 880), end, 30, tone, True, "rm")
    footer(draw, "Пример из статьи · равные прототипы ±(1, 1) · векторы показаны в одном масштабе", p)
    return image


def scene11(p):
    image, draw = begin("11", "ВЫВОД")
    t = HORIZON * soft(.10, .65, p)
    extent = (-1.36, -1.24, 1.66, 1.24)
    draw, _, _ = trajectory_panel(image, (1060, 128, 1819, 769), extent, "circle", t, True, True)
    txt(draw, (94, 159), "СМОТРИТЕ", 113, TEXT, True)
    txt(draw, (94, 286), "НА ГРАНИЦУ.", 108, MINT, True)
    stages = ((.12, 483, "01", "Карта задаёт метку."),
              (.36, 577, "02", "Поле двигает точку."),
              (.59, 671, "03", "Баланс граней сохраняет метки."))
    for a, y, k, line in stages:
        fade = soft(a, a+.10, p)
        if fade:
            txt(draw, (100, y), k, 31, mix(BG, MINT, fade), True)
            txt(draw, (178, y-7), line, 44, mix(BG, TEXT, fade), True)
    if p >= .62:
        panel(draw, (101, 826, 1818, 933), INK, EDGE, 21)
        txt(draw, (126, 850), "Gravitational Decision Maps", 38, TEXT, True)
        txt(draw, (1789, 878), "Доказательства и данные в описании", 34, MUTED, anchor="rm")
    footer(draw, "Теорема, интервальная проверка и открытые эксперименты", p)
    return image


SCENES = {"01": scene01, "02": scene02, "03": scene03, "04": scene04,
          "05": scene05, "06": scene06, "11": scene11}


def render_scene(scene_id, seconds, duration, size=(WIDTH, HEIGHT)):
    p = min(1., max(0., seconds/max(.001, duration)))
    frame = SCENES[str(scene_id).zfill(2)](p)
    return frame if size == (WIDTH, HEIGHT) else frame.resize(size, Image.Resampling.LANCZOS)


def render_thumbnail(size=(1280, 720)):
    """Build a high-contrast cover from the actual orbit and decision map."""
    frame = background().copy()
    box = (886, 81, 1841, 990)
    extent = (-1.03, -1.12, 1.72, 1.36)
    draw, project, _ = trajectory_panel(frame, box, extent, "circle", HORIZON)
    values, event = solution("circle")
    hit = project(event[1])
    start, finish = map(project, (values[0], values[-1]))
    cutoff = int(np.ceil(event[0]/HORIZON * (len(values)-1)))
    draw.line([project(y) for y in values[:cutoff]] + [hit], fill=AMBER, width=12, joint="curve")
    draw.line([hit] + [project(y) for y in values[cutoff:]], fill=MINT, width=12, joint="curve")
    dot(draw, hit, 22, None, TEXT, 4)
    dot(draw, start, 19, AMBER, TEXT, 4)
    dot(draw, finish, 19, MINT, TEXT, 4)
    txt(draw, (start[0]-8, start[1]+36), "А", 48, AMBER, True, "rm")
    txt(draw, (finish[0]-35, finish[1]-29), "Б", 48, MINT, True, "rm")
    draw.rounded_rectangle((91, 159, 148, 168), 4, fill=MINT)
    txt(draw, (169, 164), "ГАУССОВЫ ПОТОКИ", 33, MUTED, True, "lm")
    txt(draw, (88, 228), "ОДНА", 136, TEXT, True)
    txt(draw, (88, 366), "ТОЧКА.", 136, TEXT, True)
    txt(draw, (90, 535), "ДВА РЕШЕНИЯ.", 96, MINT, True)
    dot(draw, (160, 798), 50, DARK_AMBER, AMBER, 4)
    dot(draw, (456, 798), 50, DARK_MINT, MINT, 4)
    txt(draw, (160, 798), "А", 56, AMBER, True, "mm")
    txt(draw, (456, 798), "Б", 56, MINT, True, "mm")
    arrow(draw, (238, 798), (375, 798), TEXT, 6, 19)
    txt(draw, (1789, 952), "Смена метки на границе", 36, TEXT, True, "rm")
    draw.line((92, 945, 822, 945), fill=EDGE, width=3)
    txt(draw, (92, 977), "Почему движение меняет класс", 32, MUTED)
    return frame.resize(size, Image.Resampling.LANCZOS)
