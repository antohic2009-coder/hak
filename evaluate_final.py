"""Честная оценка итоговой системы тем же кодом, что в программе (dxa/model.py).

Внешняя CV — StratifiedGroupKFold по исследованию, 5 фолдов × 5 повторов; модель (пороги,
калибровка, эталоны) обучается только на обучающей части фолда. Метрики — по каждому
критерию, по области («качественное / есть нарушение») и в целом, с 95% ДИ бутстрэпом
по исследованиям. Результат: out/metrics_final.json и таблица в консоли.

    python evaluate_final.py --data /path/to/НД_для_обучения

Измерения считаются заново из DICOM (тот же код, что в сервисе), кэшей нет.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from dxa import features
from dxa.data import LABELS, load_dataset
from dxa.model import CRITERIA, QualityModel

try:  # cp1251-консоль на русской Windows не должна ронять печать
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="распакованный НД_для_обучения (с xlsx разметкой)")
args = ap.parse_args()

SEEDS, FOLDS = 5, 5
images = [i for i in load_dataset(args.data) if i.quality is not None]
keys = [f"{i.study_uid}:{i.image_uid}" for i in images]
feats = [features.geometry(i.array, i.spacing, i.region, i.side) for i in images]
Path("out").mkdir(exist_ok=True)
regions = [i.region for i in images]
labels = [i.labels for i in images]
groups = np.array([i.study_uid for i in images])
y_any = np.array([int(i.quality) for i in images])

flags = {L: np.zeros(len(images)) for L in LABELS}
scores = {L: np.zeros(len(images)) for L in LABELS}
qprob = np.zeros(len(images))
operating_flags = {
    name: np.zeros(len(images))
    for name in ("threshold_0.50", "max_f1", "recall_target_0.90")
}
operating_thresholds = {name: [] for name in operating_flags}


def operating_threshold(y, score, target_recall=None):
    """Выбирает рабочую точку только по обучающей части текущего фолда."""
    y, score = np.asarray(y, int), np.asarray(score, float)
    candidates = np.unique(score)
    if target_recall is not None:
        eligible = [
            t for t in candidates
            if recall_score(y, score >= t, zero_division=0) >= target_recall
        ]
        return float(max(eligible)) if eligible else float(np.min(score))
    values = [(f1_score(y, score >= t, zero_division=0), t) for t in candidates]
    best_f1 = max(v[0] for v in values)
    return float(max(t for f1, t in values if f1 == best_f1))


for seed in range(SEEDS):
    for tr, te in StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(images)), y_any, groups):
        m = QualityModel().fit([feats[i] for i in tr], [regions[i] for i in tr], [labels[i] for i in tr],
                               groups=groups[tr])
        train_qprob = np.array([m.probabilities(m.predict(feats[i], regions[i]), regions[i])[1] for i in tr])
        thresholds = {
            "threshold_0.50": 0.5,
            "max_f1": operating_threshold(y_any[tr], train_qprob),
            "recall_target_0.90": operating_threshold(y_any[tr], train_qprob, target_recall=0.90),
        }
        for name, threshold in thresholds.items():
            operating_thresholds[name].append(threshold)
        test_qprob = np.zeros(len(te))
        for pos, i in enumerate(te):
            out = m.predict(feats[i], regions[i])
            for L, (flag, score, _) in out.items():
                flags[L][i] += flag / SEEDS
                scores[L][i] += score / SEEDS
            test_qprob[pos] = m.probabilities(out, regions[i])[1]
            qprob[i] += test_qprob[pos] / SEEDS
        for name, threshold in thresholds.items():
            operating_flags[name][te] += (test_qprob >= threshold) / SEEDS


def boot(y, pred, score, g, n=1000):
    rng = np.random.default_rng(0)
    u = np.unique(g)
    ix = {s: np.nonzero(g == s)[0] for s in u}
    vals = {"auc": [], "f1": [], "sens": [], "spec": [], "bacc": []}
    for _ in range(n):
        i = np.concatenate([ix[s] for s in rng.choice(u, len(u))])
        if 0 < y[i].sum() < len(i):
            for k, v in metrics(y[i], pred[i], score[i]).items():
                vals[k].append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


def metrics(y, pred, score):
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return dict(auc=float(roc_auc_score(y, score)), f1=float(f1_score(y, pred, zero_division=0)),
                sens=sens, spec=spec, bacc=(sens + spec) / 2)


def operating_metrics(y, pred, score, g, threshold, thresholds):
    result = metrics(y, pred, score)
    result["precision"] = float(precision_score(y, pred, zero_division=0))
    result["threshold"] = float(threshold)
    result["threshold_min"] = float(min(thresholds))
    result["threshold_max"] = float(max(thresholds))
    result["ci"] = boot(y, pred, score, g)
    result["predicted_positive"] = int(pred.sum())
    return result


def rank01(v):
    return (np.argsort(np.argsort(v)) + 0.5) / len(v)


report = {}
rows = []
for region, crits in CRITERIA.items():
    idx = np.array([i for i, r in enumerate(regions) if r == region])
    any_pred = np.zeros(len(idx))
    any_score = np.zeros(len(idx))
    for L in crits:
        sub = np.array([i for i in idx if L in labels[i]])
        y = np.array([labels[i][L] for i in sub])
        pred = (flags[L][sub] >= 0.5).astype(int)
        m = metrics(y, pred, scores[L][sub])
        m["ci"] = boot(y, pred, scores[L][sub], groups[sub])
        m.update(n=len(y), positives=int(y.sum()))
        report[f"{region}: {L}"] = m
        pos = np.searchsorted(idx, sub)
        any_pred[pos] = np.maximum(any_pred[pos], pred)
        any_score[pos] = np.maximum(any_score[pos], rank01(scores[L][sub]))
    y = y_any[idx]
    m = metrics(y, any_pred.astype(int), any_score)
    m["ci"] = boot(y, any_pred.astype(int), any_score, groups[idx])
    m.update(n=len(y), positives=int(y.sum()))
    report[f"{region}: качество (есть нарушение)"] = m

# в целом по всем снимкам и macro-F1 по типам нарушений
all_pred = np.zeros(len(images), int)
all_score = np.zeros(len(images))
for region, crits in CRITERIA.items():
    for L in crits:
        sub = np.array([i for i, r in enumerate(regions) if r == region and L in labels[i]])
        all_pred[sub] = np.maximum(all_pred[sub], (flags[L][sub] >= 0.5).astype(int))
        all_score[sub] = np.maximum(all_score[sub], rank01(scores[L][sub]))
m = metrics(y_any, all_pred, all_score)
m["ci"] = boot(y_any, all_pred, all_score, groups)
m.update(n=len(y_any), positives=int(y_any.sum()))
report["все снимки: качество (есть нарушение)"] = m
# quality_prob — то, по чему организатор посчитает ROC-AUC (Разъяснения V2, вопрос 8)
for name, mask in (("позвоночник", np.array([r == "spine" for r in regions])),
                   ("бедро", np.array([r == "femur" for r in regions])), ("все снимки", np.ones(len(regions), bool))):
    auc = float(roc_auc_score(y_any[mask], qprob[mask]))
    lo, hi = boot(y_any[mask], (qprob[mask] >= 0.5).astype(int), qprob[mask], groups[mask])["auc"]
    report[f"quality_prob ROC-AUC: {name}"] = {"auc": auc, "ci": [lo, hi], "mean_prob": float(qprob[mask].mean()),
                                               "rate": float(y_any[mask].mean())}
report["quality_prob operating points"] = {}
for name, values in operating_flags.items():
    report["quality_prob operating points"][name] = operating_metrics(
        y_any, (values >= 0.5).astype(int), qprob, groups,
        float(np.mean(operating_thresholds[name])), operating_thresholds[name])
report["macro-F1 по типам нарушений"] = float(np.mean([v["f1"] for k, v in report.items()
                                                       if isinstance(v, dict) and "качество" not in k and not k.startswith("quality_prob")]))

print(f"{'критерий':62s} {'n':>4s} {'поз':>4s}  AUC [95% ДИ]         F1 [95% ДИ]         чувств  специф  сбал.т")
for k, v in report.items():
    if not isinstance(v, dict):
        print(f"{k}: {v:.3f}")
        continue
    if k == "quality_prob operating points":
        for name, point in v.items():
            print(f"quality_prob {name:16s} порог {point['threshold']:.3f} "
                  f"precision {point['precision']:.2f} recall {point['sens']:.2f} "
                  f"F1 {point['f1']:.2f} specificity {point['spec']:.2f} "
                  f"(порог по фолдам {point['threshold_min']:.3f}–{point['threshold_max']:.3f})")
        continue
    if k.startswith("quality_prob"):
        print(f"{k:62s} AUC {v['auc']:.3f} [{v['ci'][0]:.2f}; {v['ci'][1]:.2f}]  средняя вероятность {v['mean_prob']:.2f} при доле брака {v['rate']:.2f}")
        continue
    c = v["ci"]
    print(f"{k:62s} {v['n']:4d} {v['positives']:4d}  {v['auc']:.3f} [{c['auc'][0]:.2f}; {c['auc'][1]:.2f}]  "
          f"{v['f1']:.3f} [{c['f1'][0]:.2f}; {c['f1'][1]:.2f}]  {v['sens']:.2f}    {v['spec']:.2f}    {v['bacc']:.2f}")
json.dump(report, open("out/metrics_final.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pickle.dump(dict(flags=flags, scores=scores, qprob=qprob, keys=keys), open("out/oof_final.pkl", "wb"))
