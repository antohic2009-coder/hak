"""A/B-сравнение улучшений метрик на том же честном CV, что в evaluate_final.py.

Внешняя CV — StratifiedGroupKFold по исследованию, 5 фолдов x 5 повторов; модель обучается
только на обучающей части фолда. Измерения бедра пересчитываются текущим кодом (с анатомическим
фильтром вершины малого вертела в dxa/roi.py); старые измерения берутся из кэша обучения
(измерения позвоночника фильтр не затрагивает и берутся из того же кэша всегда).

Варианты:
  V0  старые измерения + CDF-скор + калибровка LR (in-sample)  — должен воспроизвести baseline
      из evaluate_final.py (out/metrics_final.json);
  V1  новые измерения + CDF-скор + LR in-sample               — эффект фильтра lt_peak;
  V2  новые измерения + LR-скор + LR in-sample                — эффект скоринга ротации;
  V3  новые измерения + LR-скор + изотоническая калибровка на OOF-оценках;
  V4  новые измерения + LR-скор + LR-калибровка на OOF-оценках (изоляция вклада изотоники).

Дополнительно: парный перестановочный тест для AUC скоринга ротации бедра (CDF против LR):
метки перемешиваются, для каждой перестановки обе версии скорора оцениваются на тех же
внутренних фолдах; p-значение — доля перестановок с dAUC >= реального.

Запуск из корня проекта:  python experiments/ab_improve.py --data <НД_для_обучения> [--perms 200]
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dxa import features  # noqa: E402
from dxa.data import load_dataset  # noqa: E402
from dxa.model import CRITERIA, FemurScorer, QualityModel  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--perms", type=int, default=200)
ap.add_argument("--cache", default=str(ROOT / "out" / "train_geo.pkl"))
args = ap.parse_args()

SEEDS, FOLDS = 5, 5
images = [i for i in load_dataset(args.data) if i.quality is not None]
regions = np.array([i.region for i in images])
labels = [i.labels for i in images]
groups = np.array([i.study_uid for i in images])
y_any = np.array([int(i.quality) for i in images])
keys = [f"{i.study_uid}:{i.image_uid}" for i in images]

old_cache = pickle.load(open(args.cache, "rb"))
old_feats = [old_cache[k] for k in keys]
print("пересчёт измерений бедра текущим кодом...", flush=True)
geo_cache = ROOT / "out" / "ab_new_geo.pkl"
if geo_cache.exists():
    new_feats = pickle.load(open(geo_cache, "rb"))
    print("взято из кэша", geo_cache, flush=True)
else:
    new_feats = []
    for n, im in enumerate(images):
        if im.region == "femur":
            new_feats.append(features.geometry(im.array, im.spacing, im.region, im.side))
        else:  # измерения позвоночника фильтр не затрагивает
            new_feats.append(old_cache[keys[n]])
        if n % 50 == 0:
            print(f"  {n}/{len(images)}", flush=True)
    pickle.dump(new_feats, open(geo_cache, "wb"))
print("готово", flush=True)

# диагностика: насколько пересчёт бедра отличается от кэша (эффект фильтра lt_peak)
for k in ("rot_lt_area_mm2", "roi_margin_bottom_cm", "roi_margin_medial_cm"):
    d = np.abs(np.array([f.get(k, np.nan) for f in new_feats]) -
               np.array([old_cache[keys[n]].get(k, np.nan) for n in range(len(images))]))
    ch = int((np.isfinite(d) & (d > 1e-9)).sum())
    if ch:
        dd = d[np.isfinite(d) & (d > 1e-9)]
        print(f"измерение {k}: изменилось у {ch} снимков, медиана разницы {np.median(dd):.3g}, "
              f"макс {dd.max():.3g}", flush=True)
    else:
        print(f"измерение {k}: изменений нет", flush=True)

fe_area = np.array([f.get("rot_lt_area_mm2", np.nan) for f in new_feats], float)


def operating_threshold(y, score, target_recall=None):
    y, score = np.asarray(y, int), np.asarray(score, float)
    candidates = np.unique(score)
    if target_recall is not None:
        eligible = [t for t in candidates if recall_score(y, score >= t, zero_division=0) >= target_recall]
        return float(max(eligible)) if eligible else float(np.min(score))
    values = [(f1_score(y, score >= t, zero_division=0), t) for t in candidates]
    best_f1 = max(v[0] for v in values)
    return float(max(t for f1, t in values if f1 == best_f1))


def run_cv(feats, femur_mode, calib, calib_kind="lr"):
    """CV-петля evaluate_final.py; возвращает словарь метрик сравнения."""
    old_mode = {c: getattr(QualityModel, c) for c in ("femur_score_mode", "calib_mode", "calib_kind")}
    QualityModel.femur_score_mode, QualityModel.calib_mode, QualityModel.calib_kind = femur_mode, calib, calib_kind
    try:
        flags = {L: np.zeros(len(images)) for L in CRITERIA["femur"]}
        scores = {L: np.zeros(len(images)) for L in CRITERIA["femur"]}
        qprob = np.zeros(len(images))
        op_flags = {n: np.zeros(len(images)) for n in ("t0.50", "max_f1", "sens0.90")}
        for seed in range(SEEDS):
            for tr, te in StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed).split(
                    np.zeros(len(images)), y_any, groups):
                m = QualityModel().fit([feats[i] for i in tr], [regions[i] for i in tr],
                                       [labels[i] for i in tr], groups=groups[tr])
                train_qprob = np.array([m.probabilities(m.predict(feats[i], regions[i]), regions[i])[1] for i in tr])
                thresholds = {"t0.50": 0.5, "max_f1": operating_threshold(y_any[tr], train_qprob),
                              "sens0.90": operating_threshold(y_any[tr], train_qprob, target_recall=0.90)}
                test_qprob = np.zeros(len(te))
                for pos, i in enumerate(te):
                    out = m.predict(feats[i], regions[i])
                    for L, (flag, score, _) in out.items():
                        if L in flags:
                            flags[L][i] += flag / SEEDS
                            scores[L][i] += score / SEEDS
                    test_qprob[pos] = m.probabilities(out, regions[i])[1]
                    qprob[i] += test_qprob[pos] / SEEDS
                for name, threshold in thresholds.items():
                    op_flags[name][te] += (test_qprob >= threshold) / SEEDS
    finally:
        for c, v in old_mode.items():
            setattr(QualityModel, c, v)

    def m_femur(pos_label):
        sub = np.array([i for i in range(len(images)) if regions[i] == "femur" and pos_label in labels[i]])
        y = np.array([labels[i][pos_label] for i in sub])
        pred = (flags[pos_label][sub] >= 0.5).astype(int)
        r = dict(auc=float(roc_auc_score(y, scores[pos_label][sub])), f1=float(f1_score(y, pred, zero_division=0)),
                 sens=float(recall_score(y, pred, zero_division=0)),
                 spec=float(recall_score(1 - y, 1 - pred, zero_division=0)), pos=int(y.sum()))
        yq = y_any[sub]
        pred_any = (np.maximum(flags[CRITERIA["femur"][0]][sub], flags[CRITERIA["femur"][1]][sub]) >= 0.5).astype(int)
        r["quality_f1"] = float(f1_score(yq, pred_any, zero_division=0))
        r["quality_auc"] = float(roc_auc_score(yq, np.maximum(scores[CRITERIA["femur"][0]][sub],
                                                              scores[CRITERIA["femur"][1]][sub])))
        return r

    sub_f = np.array([r == "femur" for r in regions])
    sub_s = np.array([r == "spine" for r in regions])
    result = {"pos": m_femur("Некорректная укладка"), "roi": m_femur("Некорректная область интереса")}
    result["qprob_auc_femur"] = float(roc_auc_score(y_any[sub_f], qprob[sub_f]))
    result["qprob_mean_femur"] = float(qprob[sub_f].mean())
    result["qprob_rate_femur"] = float(y_any[sub_f].mean())
    result["qprob_auc_all"] = float(roc_auc_score(y_any, qprob))
    result["qprob_mean_all"] = float(qprob.mean())
    result["qprob_rate_all"] = float(y_any.mean())
    for name in op_flags:
        pred = (op_flags[name] >= 0.5).astype(int)
        result[f"op_{name}"] = dict(f1=float(f1_score(y_any, pred, zero_division=0)),
                                    precision=float(precision_score(y_any, pred, zero_division=0)),
                                    sens=float(recall_score(y_any, pred, zero_division=0)))
    overall = np.maximum(flags[CRITERIA["femur"][0]], flags[CRITERIA["femur"][1]])
    all_pred = (overall[sub_f] >= 0.5).astype(int)
    result["quality_f1_femur"] = float(f1_score(y_any[sub_f], all_pred, zero_division=0))
    return result


VARIANTS = [
    ("V0 baseline (старые изм., CDF, LR)", old_feats, "cdf", "in_sample", "lr"),
    ("V1 фильтр lt_peak", new_feats, "cdf", "in_sample", "lr"),
    ("V2 = V1 + LR-скор", new_feats, "lr", "in_sample", "lr"),
    ("V3 = V2 + OOF-калибровка (isotonic)", new_feats, "lr", "oof", "isotonic"),
    ("V4 = V2 + OOF-калибровка (LR)", new_feats, "lr", "oof", "lr"),
    ("V5 = CDF-скор + равномерная OOF-изотоника", new_feats, "cdf", "oof", "isotonic"),
    ("V6 = CDF-скор + равномерная OOF-калибровка (LR)", new_feats, "cdf", "oof", "lr"),
]

print("\n=== сравнение вариантов (честная CV 5x5 по исследованиям) ===")
rows = {}
for name, feats, femur_mode, calib, kind in VARIANTS:
    r = run_cv(feats, femur_mode, calib, kind)
    rows[name] = r
    print(f"\n{name}")
    print(f"  укладка бедра:     AUC {r['pos']['auc']:.3f}  F1 {r['pos']['f1']:.3f}  "
          f"чувств {r['pos']['sens']:.2f}  специф {r['pos']['spec']:.2f}  (плюсов {r['pos']['pos']})")
    print(f"  ROI бедра:         AUC {r['roi']['auc']:.3f}  F1 {r['roi']['f1']:.3f}  "
          f"чувств {r['roi']['sens']:.2f}  специф {r['roi']['spec']:.2f}  (плюсов {r['roi']['pos']})")
    print(f"  бедро: качество:   F1 {r['quality_f1_femur']:.3f}")
    print(f"  quality_prob:      AUC(бедро) {r['qprob_auc_femur']:.3f} (средняя {r['qprob_mean_femur']:.2f}/брак {r['qprob_rate_femur']:.2f}); "
          f"AUC(все) {r['qprob_auc_all']:.3f} (средняя {r['qprob_mean_all']:.2f}/брак {r['qprob_rate_all']:.2f})")
    for n in ("t0.50", "max_f1", "sens0.90"):
        o = r[f"op_{n}"]
        print(f"    РТ {n:7s}: F1 {o['f1']:.3f}  precision {o['precision']:.2f}  recall {o['sens']:.2f}")


def paired_perm_auc(area, off, y, groups, perms):
    """Парный перестановочный тест: AUC LR-скора против CDF-скора ротации бедра."""
    def oof(mode, rng_seed):
        pred = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rng_seed).split(
                np.zeros(len(y)), y, groups):
            sc = FemurScorer(mode)
            sc.fit(area[tr], off[tr], y[tr])
            pred[te] = sc.score(area[te], off[te])
        return roc_auc_score(y, pred)

    real_cdf = oof("cdf", 0)
    real_lr = oof("lr", 0)
    real_d = real_lr - real_cdf
    rng = np.random.default_rng(0)
    null = []
    for k in range(perms):
        yp = rng.permutation(y)
        pred_c = np.zeros(len(yp))
        pred_l = np.zeros(len(yp))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=k).split(
                np.zeros(len(yp)), yp, groups):
            c = FemurScorer("cdf")
            c.fit(area[tr], off[tr], yp[tr])
            pred_c[te] = c.score(area[te], off[te])
            l = FemurScorer("lr")
            l.fit(area[tr], off[tr], yp[tr])
            pred_l[te] = l.score(area[te], off[te])
        null.append(roc_auc_score(yp, pred_l) - roc_auc_score(yp, pred_c))
    null = np.array(null)
    p = float((1 + (null >= real_d).sum()) / (1 + perms))
    print(f"\n=== перестановочный тест (скоринг ротации) ===\n"
          f"реальный dAUC {real_d:+.4f} (LR {real_lr:.3f} против CDF {real_cdf:.3f}); "
          f"перестановки: среднее {null.mean():+.4f}, 95% {np.percentile(null, 95):+.4f}, max {null.max():+.4f}; p = {p:.3f}")
    return p


fe_f = np.array([r == "femur" for r in regions])
off_f = np.array([f.get("rot_offset_mm", np.nan) for f in new_feats], float)
paired_perm_auc(fe_area[fe_f], off_f[fe_f], y_any[fe_f], groups[fe_f], args.perms)

import json  # noqa: E402
out_path = ROOT / "out" / "ab_improve.json"
json.dump({k: v for k, v in rows.items()}, open(out_path, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print(f"\nсохранено {out_path}")
