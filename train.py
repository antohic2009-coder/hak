"""Обучение итоговой модели на наборе организатора (НД_для_обучения).

    python train.py --data C:\dxa --out models/quality_model.joblib

Шаги: чтение DICOM и меток → измерения по критериям ТЗ → модель качества (пороги и калибровка
по обучающим данным). Для честных метрик см. evaluate_final.py (тот же код модели, CV по исследованиям).
"""

import argparse
import datetime as dt
import json
import pickle
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).parent))
from dxa import features
from dxa.data import load_dataset
from dxa.model import QualityModel


def main():
    try:  # cp1251-консоль на русской Windows не должна ронять печать
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="распакованный НД_для_обучения (с xlsx разметкой)")
    ap.add_argument("--out", default="models/quality_model.joblib")
    ap.add_argument("--cache", default="out/train_geo.pkl", help="кэш измерений (ускоряет повторное обучение)")
    ap.add_argument("--refresh-geo", action="store_true", help="пересчитать измерения, не брать из кэша")
    args = ap.parse_args()

    images = load_dataset(args.data)
    print(f"снимков {len(images)}", flush=True)
    cache = {}
    if Path(args.cache).exists() and not args.refresh_geo:
        cache = pickle.load(open(args.cache, "rb"))
    geo = []
    for n, im in enumerate(images):
        key = f"{im.study_uid}:{im.image_uid}"
        if key not in cache:
            cache[key] = features.geometry(im.array, im.spacing, im.region, im.side)
            if n % 20 == 0:
                print(f"  {n}/{len(images)}", flush=True)
        geo.append(cache[key])
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(cache, open(args.cache, "wb"))

    lab = [i for i, im in enumerate(images) if im.quality is not None]
    model = QualityModel().fit([geo[i] for i in lab], [images[i].region for i in lab], [images[i].labels for i in lab],
                               groups=[images[i].study_uid for i in lab])
    meta = dict(trained=dt.datetime.now().isoformat(timespec="seconds"), n_images=len(images), n_labeled=len(lab),
                fo_threshold=model.fo_threshold, fo_over_spine_threshold=model.fo_over_spine_threshold,
                axis_threshold=model.axis_threshold,
                femur_threshold=model.femur_threshold, area_norm=model.area_norm)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(quality=model, meta=meta), args.out)
    json.dump(meta, open(Path(args.out).with_suffix(".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("сохранено", args.out, meta)


if __name__ == "__main__":
    main()
