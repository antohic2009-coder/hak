"""Пакетная обработка: каталог / zip с DICOM → таблица результатов (+ доп. серии и DICOM SR).

Устойчивость (ТЗ п. 2.7): любая ошибка на файле или снимке фиксируется в строке результата
(processing_status = Failure, error = текст), обработка остальных продолжается.
"""

from __future__ import annotations

import io
import re
import time
import traceback
import warnings
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pydicom

from . import features
from .data import anatomy_region, pixel_spacing

warnings.filterwarnings("ignore")

REGION_NAMES = {"spine": "Поясничный отдел позвоночника", "femur": "Проксимальный отдел бедра"}
SIDE_NAMES = {"l": "левое", "r": "правое"}

# Область без тегов: у аппарата организатора снимок позвоночника 300 пикселей в ширину, бедра — 248–280
# (252 из 252 снимков обучения). Порог — середина промежутка.
SPINE_MIN_WIDTH_PX = 290


# подсказки из тегов и имён файлов (в наборе организатора: ПОП, ЛПОБ, ППОБ)
_SPINE_HINT = re.compile(r"(ПОП|позвон|spine|l1-?l4|lumbar)", re.I)
_FEMUR_HINT = re.compile(r"(ПОБ|бедр|femur|hip|neck)", re.I)
_LEFT_HINT = re.compile(r"(ЛПОБ|лев|left|\bL\b)", re.I)
_RIGHT_HINT = re.compile(r"(ППОБ|прав|right|\bR\b)", re.I)


@dataclass
class Item:
    path: str
    ds: object = None
    array: np.ndarray | None = None
    error: str | None = None
    study_uid: str = ""
    image_uid: str = ""
    instance: int = 0
    duplicate_of: str | None = None
    region: str | None = None
    side: str | None = None
    result: dict = field(default_factory=dict)


def _iter_sources(src: str | Path):
    """(имя, байты) для всех файлов каталога или zip-архива (включая вложенные zip)."""
    src = Path(src)
    if src.is_file() and zipfile.is_zipfile(src):
        yield from _iter_zip(zipfile.ZipFile(src), src.name)
        return
    for p in sorted(src.rglob("*") if src.is_dir() else [src]):
        if p.is_file():
            if p.suffix.lower() == ".zip" and zipfile.is_zipfile(p):
                yield from _iter_zip(zipfile.ZipFile(p), str(p))
            elif p.suffix.lower() not in (".xlsx", ".csv", ".txt", ".pdf", ".json", ".md"):
                yield str(p), p.read_bytes()


def _iter_zip(z: zipfile.ZipFile, prefix: str):
    for info in z.infolist():
        if info.is_dir():
            continue
        name = info.filename
        if not info.flag_bits & 0x800:  # имена без флага UTF-8 в архивах Windows — в cp866
            try:
                name = name.encode("cp437").decode("cp866")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        data = z.read(info)
        if name.lower().endswith(".zip"):
            yield from _iter_zip(zipfile.ZipFile(io.BytesIO(data)), f"{prefix}/{name}")
        elif not name.lower().endswith((".xlsx", ".csv", ".txt", ".pdf", ".json", ".md")):
            yield f"{prefix}/{name}", data


def read_items(src) -> list[Item]:
    items = []
    for name, data in _iter_sources(src):
        it = Item(path=name)
        try:
            ds = pydicom.dcmread(io.BytesIO(data), force=True)
            if "PixelData" not in ds:
                if ds.get("SOPClassUID") or ds.get("DirectoryRecordSequence"):
                    continue  # настоящий DICOM без изображения (SR, DICOMDIR) — не наш вход
                raise ValueError("файл не является DICOM-изображением")
            arr = np.asarray(ds.pixel_array)
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                arr = arr[..., :3].mean(axis=-1)
            elif arr.ndim == 3:
                arr = arr[0]
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                arr = arr.max() - arr
            it.ds, it.array = ds, arr
            it.study_uid = str(ds.get("StudyInstanceUID", "") or Path(name).parent.name)
            it.image_uid = str(ds.get("SOPInstanceUID", "") or Path(name).name)
            it.instance = int(ds.get("InstanceNumber", 0) or 0)
        except Exception as e:  # битый или не-DICOM файл
            it.error = f"не удалось прочитать DICOM: {e}"
            it.study_uid = Path(name).parent.name
            it.image_uid = Path(name).name
        items.append(it)
    return items


def _hint(it: Item, pattern: re.Pattern) -> bool:
    ds = it.ds
    text = " ".join(str(ds.get(t, "") or "") for t in ("BodyPartExamined", "SeriesDescription", "ProtocolName",
                                                          "ImageLaterality", "Laterality", "StudyDescription")) if ds is not None else ""
    return bool(pattern.search(text) or pattern.search(Path(it.path).name))


class Service:
    """Загружает модели один раз; process() — обработка каталога/zip."""

    def __init__(self, model_path="models/quality_model.joblib"):
        self.model = joblib.load(model_path)["quality"]

    # --- область и сторона
    @staticmethod
    def _region_with_source(it: Item) -> tuple[str, str]:
        spine_hint, femur_hint = _hint(it, _SPINE_HINT), _hint(it, _FEMUR_HINT)
        if spine_hint != femur_hint:
            return ("spine" if spine_hint else "femur"), "теги/имя файла"
        # анатомия важнее ширины кадра: правило по ширине настроено под конкретный аппарат,
        # а форма столба/диафиза — нет (защита от смены аппарата на закрытом тесте)
        spacing = it.result.get("spacing") or pixel_spacing(it.ds, it.array)
        anat = anatomy_region(it.array, spacing)
        if anat:
            return anat, "анатомия (форма структуры)"
        return ("spine" if it.array.shape[1] >= SPINE_MIN_WIDTH_PX else "femur"), \
            f"ширина кадра {it.array.shape[1]} px"

    @staticmethod
    def _region(it: Item) -> str:
        return Service._region_with_source(it)[0]

    @staticmethod
    def _region_source(it: Item) -> str:
        """Откуда взята область — для прозрачности в details."""
        return Service._region_with_source(it)[1]

    @staticmethod
    def _anatomical_side(it: Item) -> str:
        """Сторона бедра по анатомии: головка и таз — медиально, там больше кости вверху."""
        from . import roi

        lm = roi.landmarks(it.array, pixel_spacing(it.ds, it.array), None)
        # medial="right" на снимке — правое бедро (пациент лицом к наблюдателю)
        return "r" if lm and lm["medial"] == "right" else "l"

    def _side(self, it: Item) -> tuple[str, str]:
        left, right = _hint(it, _LEFT_HINT), _hint(it, _RIGHT_HINT)
        anat = self._anatomical_side(it)
        if left != right:
            tag = "l" if left else "r"
            return tag, ("теги/имя файла" + ("" if tag == anat else f"; анатомия указывает на {SIDE_NAMES[anat]}"))
        return anat, "по анатомии (головка медиально)"

    def process(self, src, series_dir: str | Path | None = None, progress=None) -> list[dict]:
        t_read = time.time()
        items = read_items(src)
        by_study = defaultdict(list)
        for it in items:
            by_study[it.study_uid].append(it)
        rows = []
        for n, (study, group) in enumerate(by_study.items()):
            t0 = time.time()
            seen = {}
            for it in sorted(group, key=lambda i: i.instance):
                if it.error:
                    continue
                key = it.array.tobytes()
                if key in seen:  # копия уже обработанного изображения
                    it.duplicate_of = seen[key].image_uid
                    continue
                seen[key] = it
                try:
                    self._process_image(it)
                except Exception as e:
                    it.error = f"{type(e).__name__}: {e}"
                    it.result["trace"] = traceback.format_exc(limit=3)
            for it in group:
                src_it = next((s for s in group if s.image_uid == it.duplicate_of), None) if it.duplicate_of else it
                if src_it is not it and src_it is not None:
                    it.result, it.region, it.side, it.error = src_it.result, src_it.region, src_it.side, src_it.error
            elapsed = (time.time() - t0) / max(len(group), 1)
            for it in group:
                rows.append(self._row(it, elapsed))
                if series_dir and not it.error and not it.duplicate_of:
                    try:
                        from . import report
                        png = report.write_series(it, Path(series_dir))
                        rows[-1]["overlay"] = str(Path(png).relative_to(Path(series_dir)))
                    except Exception as e:
                        rows[-1]["error"] = (rows[-1].get("error") or "") + f" доп. серия: {e}"
            if progress:
                progress(n + 1, len(by_study))
        return rows

    def _process_image(self, it: Item):
        spacing = pixel_spacing(it.ds, it.array)
        it.result["spacing"] = spacing
        it.region, it.result["region_source"] = self._region_with_source(it)
        if it.region == "femur":
            it.side, it.result["side_source"] = self._side(it)
        feats = features.geometry(it.array, spacing, it.region, it.side)
        it.result["features"] = feats
        it.result["criteria"] = self.model.predict(feats, it.region)
        it.result["probs"], it.result["quality_prob"] = self.model.probabilities(it.result["criteria"], it.region)

    @staticmethod
    def _row(it: Item, elapsed: float) -> dict:
        crit = it.result.get("criteria", {})
        violations = [name for name, (flag, _, _) in crit.items() if flag]
        region = REGION_NAMES.get(it.region, "")  # «Разъяснения V2», вопрос 15: сторона не указывается
        return {
            "path_to_study": it.path,
            "study_uid": it.study_uid,
            "image_uid": it.image_uid,
            "anatomical_region": region if not it.error else "",
            # ТЗ 2.2: проекция. DXA позвоночника и бедра по протоколу — прямая (AP); берём тег, если он есть
            "projection": (str(it.ds.get("ViewPosition", "") or "AP") if it.ds is not None else "") if not it.error else "",
            "side": SIDE_NAMES.get(it.side, "") if not it.error else "",
            "quality_class": (1 if violations else 0) if not it.error else "",
            # «Разъяснения V2», вопрос 8: вероятность нарушения в [0;1]
            "quality_prob": round(it.result.get("quality_prob", 0.0), 4) if not it.error else "",
            "violation_type": "; ".join(violations) if not it.error else "",
            "processing_status": "Failure" if it.error else "Success",
            "time_of_processing": round(elapsed, 3),
            "details": " | ".join(
                ([f"область определена: {it.result['region_source']}"] if it.result.get("region_source") else [])
                + [f"{name}: {text} (вероятность {it.result.get('probs', {}).get(name, 0):.2f})"
                   for name, (_, _, text) in crit.items()]),
            "duplicate_of": it.duplicate_of or "",
            "error": it.error or "",
        }
