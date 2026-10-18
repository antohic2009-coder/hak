"""Модель контроля качества: решение по каждому критерию ТЗ.

| Область | Критерий (метка организатора) | Как решается |
|---|---|---|
| позвоночник | Не выравнена ось позвоночника | правило ТЗ: наклон > 5° (линия через крайние позвонки) |
| позвоночник | Некорректная укладка | правило ТЗ: не видны гребни подвздошных костей |
| позвоночник | Присутствуют посторонние предметы | сила тонких ярких линий вне столба, порог по обучению |
| бедро | Некорректная область интереса | правило ТЗ: < 3 см ниже малого вертела до края кадра |
| бедро | Некорректная укладка (ротация) | среднее двух оценок: малый вертел (U-образно), офсет головки |

Обучаемые параметры (fit): медиана выступа малого вертела у нормы, эталонные распределения
для перевода оценок в процентили, пороги для двух оценок, калибровка вероятностей.
Нейросетевых признаков нет: замороженный SigLIP в честной CV не улучшал ротацию бедра
(AUC 0.852 с ним против 0.863 без него), а тянул за собой torch и 1.6 ГБ весов.
Пороги подбираются по F1 на out-of-fold предсказаниях ОБУЧАЮЩЕЙ части.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .data import LABEL_AXIS, LABEL_FOREIGN, LABEL_POSITIONING, LABEL_ROI

AXIS_LIMIT_DEG = 5.0          # ТЗ
# Порог оси проверяли обучаемым (коридор 4.0-5.0°): честная CV дала F1 0.370 против 0.364 при
# пороге ТЗ, но общий F1 просел 0.745 -> 0.739 из-за ложных тревог. Оставлен порог ТЗ:
# ось ограничена шумом разметки (6/10 нарушений имеют наклон 2-4.6°, 8 норм наклонены > 5°),
# а не выбором порога. См. REPORT.md, раздел 4.1.
ILIAC_MIN_FRAC = 0.005        # «гребни не видны»: кости по бокам у нижнего края практически нет
ROI_BOTTOM_CM = 3.0           # ТЗ
# ТЗ: по 3 см сверху и снизу от области интереса; стандартная область интереса бедра ~7 см —
# скан короче 7 + 3 + 3 = 13 см не может её вместить (бедро почти не попало в кадр)
FIELD_MIN_CM = 13.0
ISCHIUM_MIN_FRAC = 0.02       # ТЗ: седалищная кость должна быть в кадре; «не видна» — кости почти нет

CRITERIA = {
    "spine": [LABEL_POSITIONING, LABEL_AXIS, LABEL_FOREIGN],
    "femur": [LABEL_POSITIONING, LABEL_ROI],
}


def _nan(x, fill):
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x), x, fill)


def _cdf(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    ref = np.sort(ref)
    return np.searchsorted(ref, x, side="right") / max(len(ref), 1)


def _best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Порог с максимальным F1; кандидаты — квантили оценки (без подгонки под отдельные точки)."""
    cands = np.unique(np.quantile(score, np.linspace(0.5, 0.98, 49)))
    f1 = [f1_score(y, score >= t, zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1))])


def _axis_threshold(y: np.ndarray, angle: np.ndarray) -> float:
    """Порог наклона оси — значение ТЗ.

    Обучаемый порог проверяли (коридор 4.0-5.0°): выигрыша в честной CV нет, а ложных тревог
    больше. Разметка врачей расходится с порогом ТЗ в обе стороны, поэтому следуем ТЗ.
    """
    return AXIS_LIMIT_DEG


class QualityModel:
    def fit(self, feats: list[dict], regions: list[str], labels: list[dict]):
        regions = np.array(regions)
        # --- позвоночник: порог оси (в коридоре вокруг ТЗ) и посторонние предметы
        sp = np.nonzero(regions == "spine")[0]
        y_axis = np.array([labels[i].get(LABEL_AXIS, 0) for i in sp])
        angle = _nan([feats[i].get("axis_chord15_deg", np.nan) for i in sp], 0.0)
        self.axis_threshold = _axis_threshold(y_axis, angle)
        y_fo = np.array([labels[i].get(LABEL_FOREIGN, 0) for i in sp])
        ridge = _nan([feats[i].get("fo_ridge_p99", np.nan) for i in sp], 0.0)
        self.fo_threshold = _best_threshold(y_fo, ridge)
        # Второй детектор — объекты ПОВЕРХ столба (замечание экспертов 16.10). Порог консервативный:
        # максимум по нормам обучающей части (ложных тревог на обучении нет), срабатывает только на
        # сильном сигнале — чтобы не потерять предмет, наложенный на позвоночник, не вредя специфичности.
        over = _nan([feats[i].get("fo_over_spine_p99", np.nan) for i in sp], 0.0)
        neg_in = over[y_fo == 0]
        self.fo_over_spine_threshold = float(neg_in.max() * 1.02) if len(neg_in) else float("inf")

        # --- бедро: укладка/ротация — две оценки
        fe = np.nonzero(regions == "femur")[0]
        y = np.array([labels[i].get(LABEL_POSITIONING, 0) for i in fe])
        area = np.array([feats[i].get("rot_lt_area_mm2", np.nan) for i in fe], float)
        off = np.array([feats[i].get("rot_offset_mm", np.nan) for i in fe], float)
        self.area_fill, self.off_fill = float(np.nanmedian(area)), float(np.nanmedian(off))
        area, off = _nan(area, self.area_fill), _nan(off, self.off_fill)
        # площадь выступа — величина «во сколько раз»: гладкий контур 1 мм² и норма 6 мм² отличаются
        # так же сильно, как крупный вертел 36 мм²; в линейной шкале переротация недооценивалась
        area = np.log1p(np.clip(area, 0, None))
        self.area_norm = float(np.median(area[y == 0]))
        self.ref_u = np.abs(area - self.area_norm)
        self.ref_off = -off
        # Третий признак (наклон диафиза rot_shaft_deg) проверяли честной CV + перестановочным тестом:
        # при пороге max-F1 он поднимал F1 0.669 -> 0.722, но за счёт recall (0.789 -> 0.739),
        # а при одинаковом recall>=0.85 был ХУЖЕ (F1 0.609 против 0.669). Прирост AUC +0.004 незначим
        # (p = 0.17). Отклонён: обмен чувствительности на точность не нужен — см. REPORT.md 4.5.
        score = self._femur_score(area, off)
        self.femur_threshold = _best_threshold(y, score)

        # --- калибровка: оценка каждого критерия -> вероятность нарушения (для quality_prob)
        self.calib = {}
        spine_out = [self.predict(feats[i], "spine") for i in sp]
        for L in (LABEL_AXIS, LABEL_POSITIONING, LABEL_FOREIGN):
            idx = [k for k, i in enumerate(sp) if L in labels[i]]
            self._fit_calib(("spine", L), [spine_out[k][L][1] for k in idx], [labels[sp[k]][L] for k in idx])
        roi_idx = [i for i in fe if LABEL_ROI in labels[i]]
        self._fit_calib(("femur", LABEL_ROI), [self.predict(feats[i], "femur")[LABEL_ROI][1] for i in roi_idx],
                        [labels[i][LABEL_ROI] for i in roi_idx])
        self._fit_calib(("femur", LABEL_POSITIONING), score, y)
        return self

    def _fit_calib(self, key, score, y):
        s = np.asarray(score, float)
        y = np.asarray(y, int)
        mu, sd = float(s.mean()), float(s.std() + 1e-9)
        lr = LogisticRegression(C=1.0).fit(((s - mu) / sd)[:, None], y) if 0 < y.sum() < len(y) else None
        self.calib[key] = (mu, sd, lr, float(y.mean()))

    def probabilities(self, out: dict, region: str) -> tuple[dict, float]:
        """Вероятность нарушения по каждому критерию и итоговая quality_prob = P(хотя бы одно нарушение)."""
        probs = {}
        for L, (_, score, _) in out.items():
            mu, sd, lr, base = self.calib.get((region, L), (0.0, 1.0, None, 0.0))
            probs[L] = float(lr.predict_proba([[(score - mu) / sd]])[0, 1]) if lr is not None else base
        return probs, float(1 - np.prod([1 - p for p in probs.values()]))

    def _femur_score(self, area, off):
        u = np.abs(area - self.area_norm)
        return (_cdf(self.ref_u, u) + _cdf(self.ref_off, -off)) / 2

    def predict(self, feat: dict, region: str) -> dict:
        """{метка: (флаг 0/1, оценка, объяснение)} по критериям области."""
        out = {}
        if region == "spine":
            ang = feat.get("axis_chord15_deg", np.nan)
            thr_axis = float(getattr(self, "axis_threshold", AXIS_LIMIT_DEG))
            flag = int(np.isfinite(ang) and ang > thr_axis)
            out[LABEL_AXIS] = (flag, float(_nan([ang], 0)[0]),
                               f"наклон оси {ang:.1f}° (порог {thr_axis:.1f}°, допуск ТЗ {AXIS_LIMIT_DEG:g}°)")
            fr = feat.get("sp_iliac_frac", np.nan)
            out[LABEL_POSITIONING] = (int(np.isfinite(fr) and fr < ILIAC_MIN_FRAC), float(-_nan([fr], 0)[0]),
                                      "гребни подвздошных костей не видны у нижнего края" if np.isfinite(fr) and fr < ILIAC_MIN_FRAC
                                      else "гребни подвздошных костей видны")
            r = float(_nan([feat.get("fo_ridge_p99", np.nan)], 0)[0])
            r_in = float(_nan([feat.get("fo_over_spine_p99", np.nan)], 0)[0])
            thr_in = float(getattr(self, "fo_over_spine_threshold", float("inf")))
            hit_lat = r >= self.fo_threshold
            hit_in = r_in >= thr_in
            hit = hit_lat or hit_in
            where = "вне столба" if hit_lat else ("поверх позвоночного столба" if hit_in else "")
            out[LABEL_FOREIGN] = (int(hit), r,
                                  f"тонкие яркие линии вне позвоночника: {r:.3f} (порог {self.fo_threshold:.3f}); "
                                  f"поверх столба: {r_in:.3f} (порог {thr_in:.3f})"
                                  + (f" — обнаружено {where}" if hit else ""))
        else:
            b = feat.get("roi_margin_bottom_cm", np.nan)
            h = feat.get("field_height_cm", np.nan)
            rel = feat.get("femur_roi_reliable", 1.0)
            short_field = bool(np.isfinite(h) and h < FIELD_MIN_CM)
            # при ненадёжных ориентирах (короткий скан) решение принимаем ТОЛЬКО по длине поля:
            # отступ до малого вертела не используем (замечание экспертов 16.10)
            short_bottom = bool(rel and np.isfinite(b) and b < ROI_BOTTOM_CM)
            # оценка — насколько не хватает до нормы ТЗ (больше — хуже)
            deficit = max(ROI_BOTTOM_CM - float(_nan([b], 0)[0]), FIELD_MIN_CM - float(_nan([h], FIELD_MIN_CM)[0]))
            out[LABEL_ROI] = (int(short_bottom or short_field), deficit,
                              f"ниже малого вертела {b:.1f} см (норма ТЗ ≥ {ROI_BOTTOM_CM:g}); длина скана {h:.1f} см "
                              f"(минимум {FIELD_MIN_CM:g}); сверху {feat.get('roi_margin_top_cm', np.nan):.1f}, "
                              f"медиально {feat.get('roi_margin_medial_cm', np.nan):.1f} см"
                              + ("; скан слишком короткий — бедро не помещается" if short_field else "")
                              + ("; ориентиры ненадёжны — решение только по длине скана" if not rel else ""))
            area_mm2 = float(_nan([feat.get("rot_lt_area_mm2", np.nan)], self.area_fill)[0])
            area = float(np.log1p(max(area_mm2, 0.0)))
            off = float(_nan([feat.get("rot_offset_mm", np.nan)], self.off_fill)[0])
            s = float(self._femur_score(np.array([area]), np.array([off]))[0])
            kind = "переротация (контур гладкий)" if area < self.area_norm else "недоротация (малый вертел крупный)"
            isch = feat.get("femur_ischium_frac", np.nan)
            no_ischium = bool(np.isfinite(isch) and isch < ISCHIUM_MIN_FRAC)
            rotated = s >= self.femur_threshold
            out[LABEL_POSITIONING] = (int(rotated or no_ischium), s,
                                      f"малый вертел {area_mm2:.0f} мм² (норма ~{np.expm1(self.area_norm):.0f}), офсет головки {off:.0f} мм; "
                                      f"оценка ротации {s:.2f} (порог {self.femur_threshold:.2f})"
                                      + (f"; вероятно {kind}" if rotated else "")
                                      + ("; седалищная кость не попала в кадр" if no_ischium else ""))
        return out
