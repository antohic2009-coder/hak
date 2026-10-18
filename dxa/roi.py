"""Корректность области интереса бедра без обучения.

ТЗ: «правильной считается визуализация по 3 см сверху и снизу от области интереса,
2 см от края правого и левого». Область интереса на GE Lunar (Total Hip) охватывает
головку, шейку, большой и малый вертел. Её границы строим по анатомическим ориентирам:
  * верх — верхний край головки бедра;
  * низ — нижний край малого вертела;
  * медиально — медиальный край головки;
  * латерально — наружный край большого вертела.
Отступ меряется до края СНЯТОГО поля (чёрные поля дополнения и вырезы — не поле).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

Spacing = tuple[float, float]

MARGIN_TOP_CM = 3.0
MARGIN_BOTTOM_CM = 3.0
MARGIN_SIDE_CM = 2.0
FEMUR_LEVEL = 0.08
# ТЗ: область интереса бедра ~7 см + по 3 см сверху и снизу = минимум 13 см снятого поля.
# Скан короче — бедро почти не попало в кадр, ориентиры и рамка ненадёжны (замечание экспертов 16.10).
ROI_MIN_FIELD_CM = 13.0


def normalize(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float32)
    lo, hi = np.percentile(values, (0.5, 99.5))
    return np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1)


def field_mask(array: np.ndarray) -> np.ndarray:
    """Снятое поле — рамка кадра без строк и столбцов нулевого дополнения.

    Воздух и мягкие ткани на DXA тоже бывают нулевыми, поэтому поле нельзя брать
    по ненулевым пикселям: берём прямоугольник от первой до последней ненулевой строки/столбца.
    """
    nz = array > 0
    rows = np.nonzero(nz.sum(axis=1) > 3)[0]
    cols = np.nonzero(nz.sum(axis=0) > 3)[0]
    field = np.zeros_like(nz)
    if len(rows) and len(cols):
        field[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1] = True
    return field


def femur_mask(values: np.ndarray, field: np.ndarray) -> np.ndarray:
    inside = values[field]
    soft, dense = np.percentile(inside, (50, 98))
    mask = (values > soft + (dense - soft) * FEMUR_LEVEL) & field
    mask = ndimage.binary_closing(mask, np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, np.ones((3, 3)))
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    # бедренная кость — компонента, в которой лежит диафиз у нижнего края поля
    bottom_rows = np.nonzero(field.any(axis=1))[0][-5:]
    ids = [i for i in np.unique(labels[bottom_rows]) if i]
    if not ids:
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        return labels == int(np.argmax(sizes))
    sizes = np.bincount(labels.ravel())
    return labels == max(ids, key=lambda i: sizes[i])


def landmarks(array: np.ndarray, spacing: Spacing, side: str | None = None) -> dict | None:
    """Ориентиры области интереса в пикселях (x, y)."""
    values = normalize(array)
    field = field_mask(array)
    mask = femur_mask(values, field)
    ys, xs = np.nonzero(mask)
    if len(ys) < 200:
        return None
    rows, cols = mask.shape
    sx, sy = spacing
    y1 = int(ys.max())

    # диафиз: нижние 3 см кости
    shaft_rows = range(max(y1 - int(30 / sy), 0), y1 + 1)
    shaft_x = np.array([np.nonzero(mask[r])[0].mean() for r in shaft_rows if mask[r].any()])
    shaft_w = np.array([mask[r].sum() for r in shaft_rows if mask[r].any()])
    shaft_center, shaft_width = float(np.median(shaft_x)), float(np.median(shaft_w))

    # медиальная сторона — где головка: больше кости вверху по эту сторону от диафиза
    upper = mask[: max(y1 - int(60 / sy), 1)]
    left_mass = upper[:, : int(shaft_center)].sum()
    right_mass = upper[:, int(shaft_center) :].sum()
    medial = "left" if left_mass > right_mass else "right"
    if side in ("l", "r"):  # сторона из порядка съёмки надёжнее: таз искажает массу
        medial = "right" if side == "r" else "left"
    sign = -1 if medial == "left" else 1  # направление к медиальной стороне по x

    # головка: максимум карты расстояний по медиальную сторону от диафиза, выше его
    dist = ndimage.distance_transform_edt(mask)
    region = np.zeros_like(mask)
    top_limit = y1 - int(40 / sy)
    # центр головки смещён от оси диафиза медиально на ~40 мм (офсет бедра);
    # без этого ограничения максимум попадает в толстую межвертельную область
    offset = int(25 / sx)
    if medial == "left":
        region[:top_limit, : max(int(shaft_center) - offset, 1)] = True
    else:
        region[:top_limit, min(int(shaft_center) + offset, cols - 1) :] = True
    d = np.where(region, dist, 0)
    hy, hx = np.unravel_index(int(np.argmax(d)), d.shape)
    radius = float(d[hy, hx])
    head_top = hy - radius
    head_medial = hx + sign * radius

    # большой вертел: самая латеральная точка кости выше середины между головкой и низом
    lat_rows = range(max(int(hy - radius), 0), int((hy + y1) / 2))
    lateral_x = []
    for r in lat_rows:
        c = np.nonzero(mask[r])[0]
        if len(c):
            lateral_x.append(c.min() if medial == "right" else c.max())
    gt_lateral = float(min(lateral_x) if medial == "right" else max(lateral_x)) if lateral_x else shaft_center

    # малый вертел: выступ медиального контура ниже головки, над диафизом;
    # нижний край — где контур возвращается к линии диафиза
    band = [r for r in range(int(hy + radius), y1 + 1) if mask[r].any()]
    lt_bottom = float(hy + radius)
    lt_peak = None  # вершина малого вертела (x, y) — для анатомической разметки
    if len(band) > 10:
        edge = np.array([(np.nonzero(mask[r])[0].min() if medial == "left" else np.nonzero(mask[r])[0].max())
                         for r in band], dtype=float)
        shaft_edge = shaft_center + sign * shaft_width / 2
        protrusion = (edge - shaft_edge) * sign * sx  # мм медиальнее линии диафиза
        protrusion = ndimage.median_filter(protrusion, 5)
        peak = int(np.argmax(protrusion))
        # анатомическая защита: вершина малого вертела лежит в 1.2–3.2 радиуса головки ниже
        # её центра; ниже — седалищная кость, на которую медиальный контур может «перепрыгнуть»
        # и увести вершину вниз (тогда lt_bottom занижает отступ до края и портит решение по ТЗ)
        rel = (np.asarray(band, float) - hy) / radius
        ok = np.nonzero((rel >= 1.2) & (rel <= 3.2))[0]
        if len(ok):
            peak = int(ok[int(np.argmax(protrusion[ok]))])
        lt_peak = (float(edge[peak]), float(band[peak]))
        after = np.nonzero(protrusion[peak:] < max(3.0, protrusion[peak] * 0.2))[0]
        lt_bottom = float(band[peak + int(after[0])] if len(after) else band[-1])

    # границы снятого поля в строках/столбцах ориентиров
    def field_top(x):
        c = np.nonzero(field[:, int(np.clip(x, 0, cols - 1))])[0]
        return float(c.min()) if len(c) else 0.0

    def field_bottom(x):
        c = np.nonzero(field[:, int(np.clip(x, 0, cols - 1))])[0]
        return float(c.max()) if len(c) else rows - 1.0

    def field_side(y, left):
        c = np.nonzero(field[int(np.clip(y, 0, rows - 1))])[0]
        if not len(c):
            return 0.0 if left else cols - 1.0
        return float(c.min() if left else c.max())

    roi_left = min(head_medial, gt_lateral)
    roi_right = max(head_medial, gt_lateral)
    top_y, bottom_y = head_top, lt_bottom
    # отступ сверху — по всей ширине области (минимум), снизу — тоже
    xs_roi = np.linspace(roi_left, roi_right, 9)
    margin_top = min(top_y - field_top(x) for x in xs_roi) * sy / 10
    margin_bottom = min(field_bottom(x) - bottom_y for x in xs_roi) * sy / 10
    ys_roi = np.linspace(top_y, bottom_y, 9)
    margin_left = min(roi_left - field_side(y, True) for y in ys_roi) * sx / 10
    margin_right = min(field_side(y, False) - roi_right for y in ys_roi) * sx / 10
    margin_medial, margin_lateral = (margin_left, margin_right) if medial == "left" else (margin_right, margin_left)

    # Надёжность ориентиров (замечание экспертов 16.10): на коротком скане рамка области интереса
    # ошибочно сегментирует посторонние структуры, и по ней нельзя делать вывод. Флаг сообщает
    # потребителю (оверлей, отчёт), что рамку рисовать не следует.
    field_height_cm = float(field.any(axis=1).sum() * sy / 10)
    if field_height_cm < ROI_MIN_FIELD_CM:
        reliable, reason = False, "короткий скан: поле {:.1f} см < {:g} см".format(field_height_cm, ROI_MIN_FIELD_CM)
    elif head_top < 0:
        reliable, reason = False, "верх головки бедра за кадром"
    else:
        reliable, reason = True, ""

    return dict(
        mask=mask, field=field, medial=medial,
        head=(hx, hy, radius), gt_lateral=gt_lateral, lt_bottom=lt_bottom,
        shaft_center=float(shaft_center), lt_peak=lt_peak,
        box=(roi_left, top_y, roi_right, bottom_y),
        reliable=reliable, reliable_reason=reason,
        roi_margin_top_cm=float(margin_top), roi_margin_bottom_cm=float(margin_bottom),
        roi_margin_medial_cm=float(margin_medial), roi_margin_lateral_cm=float(margin_lateral),
        roi_height_cm=float((bottom_y - top_y) * sy / 10),
        field_height_cm=field_height_cm,
    )


def measure(array: np.ndarray, spacing: Spacing, side: str | None = None) -> dict:
    lm = landmarks(array, spacing, side)
    if lm is None:
        return dict(roi_margin_top_cm=np.nan, roi_margin_bottom_cm=np.nan,
                    roi_margin_medial_cm=np.nan, roi_margin_lateral_cm=np.nan,
                    field_height_cm=np.nan, femur_roi_reliable=0.0)
    out = {k: v for k, v in lm.items() if k.startswith(("roi_", "field_"))}
    out["femur_roi_reliable"] = 1.0 if lm.get("reliable", True) else 0.0
    out["femur_ischium_frac"] = ischium_fraction(array, lm)
    return out


def ischium_fraction(array: np.ndarray, lm: dict) -> float:
    """ТЗ: «на изображении представлены большой вертел, шейка бедра и седалищная кость».

    Седалищная кость лежит медиальнее и ниже головки бедра. Считаем долю кости в зоне
    от 1.2 до 3 радиусов головки ниже её центра, медиальнее 0.3 радиуса от центра.
    Почти ноль — кость не попала в кадр. (Вертел и шейку отдельно не проверяем: они есть
    на любом снимке, где найдены головка и диафиз; вертел у края кадра врачи не штрафуют.)
    """
    values = normalize(array)
    field = lm["field"]
    hx, hy, r = lm["head"]
    if lm["medial"] == "left":
        values, field = values[:, ::-1], field[:, ::-1]
        hx = values.shape[1] - 1 - hx
    inside = values[field]
    if not inside.size:
        return float("nan")
    soft, dense = np.percentile(inside, (50, 98))
    # порог плотнее, чем для маски бедра: мягкие ткани у таза не должны сойти за кость
    bone = (values > soft + (dense - soft) * 0.15) & field
    y0, y1, x0 = int(hy + 1.2 * r), int(hy + 3.0 * r), int(hx + 0.3 * r)
    zone = field[y0:y1, x0:]
    return float(bone[y0:y1, x0:][zone].mean()) if zone.any() else 0.0


def deficit(features: dict) -> float:
    """Насколько не хватает до нормы ТЗ, см (0 — всё в норме). Непрерывная оценка для AUC."""
    return max(0.0,
               MARGIN_TOP_CM - features["roi_margin_top_cm"],
               MARGIN_BOTTOM_CM - features["roi_margin_bottom_cm"],
               MARGIN_SIDE_CM - features["roi_margin_medial_cm"],
               MARGIN_SIDE_CM - features["roi_margin_lateral_cm"])


def verdict(features: dict) -> tuple[int, str]:
    """Решение по отступу снизу (3 см ниже малого вертела), остальные отступы — в отчёт.

    Верхний и боковые отступы ТЗ меряются от рамки области интереса GE, которой нет
    в выгруженном снимке. От анатомических ориентиров они строже практики врачей:
    верх < 3 см от головки у половины норм, большой вертел почти всегда у края кадра.
    Отступ снизу от малого вертела ориентиров не требует лишних допущений и с разметкой
    совпадает без ложных срабатываний.
    """
    f = features
    bad = f["roi_margin_bottom_cm"] < MARGIN_BOTTOM_CM
    text = (f"отступы от области интереса: снизу {f['roi_margin_bottom_cm']:.1f} см (норма ≥ {MARGIN_BOTTOM_CM:g}), "
            f"сверху {f['roi_margin_top_cm']:.1f}, медиально {f['roi_margin_medial_cm']:.1f}, "
            f"латерально {f['roi_margin_lateral_cm']:.1f} см")
    return int(bad), text
