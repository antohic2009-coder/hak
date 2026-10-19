"""Дымовые тесты без pytest: python tests/run_smoke.py

Повторяет проверки tests/test_smoke.py средствами стандартной библиотеки.
Полный прогон пропускается, если нет модели или каталога с тестовым фрагментом (DXA_TEST_DIR).
"""

import io
import os
import sys
import tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dxa.pipeline import read_items  # noqa: E402

TEST_DIR = Path(os.environ.get("DXA_TEST_DIR", "C:/dxa_test"))
FAILED = []


def check(name, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {name}" + (f" {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


# 1) чтение устойчиво к мусору: плохие файлы попадают в отчёт как ошибки
with tempfile.TemporaryDirectory() as td:
    (Path(td) / "garbage.dcm").write_text("not a dicom", encoding="utf-8")
    (Path(td) / "empty.dcm").write_bytes(b"")
    items = read_items(Path(td))
    check("reader: мусорные файлы попадают в отчёт как ошибки",
          len(items) == 2 and all(it.error for it in items), f"(n={len(items)})")

# 2) полный прогон на фрагменте организатора (если есть модель и фрагмент)
model = ROOT / "models" / "quality_model.joblib"
if model.exists() and TEST_DIR.exists():
    os.chdir(ROOT)
    from dxa.predict import main  # noqa: E402
    import pandas as pd

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        rc = main(["--input", str(TEST_DIR), "--output", str(out / "r.xlsx"),
                   "--series", str(out / "s")])
        df = pd.read_csv(out / "r.csv")
        check("predict: код возврата 0", rc == 0, f"({rc})")
        check("predict: все файлы Success", bool((df["processing_status"] == "Success").all()))
        check("predict: quality_class ∈ {0, 1}", set(df["quality_class"]) <= {0, 1})
        cols = ("path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
                "violation_type", "processing_status", "time_of_processing")
        check("predict: колонки ТЗ на месте", all(c in df.columns for c in cols))
        check("predict: время обработки < 180 с", bool((df["time_of_processing"] < 180).all()))
        check("predict: выгружается SR-серия", any((out / "s").rglob("*_SR.dcm")))
else:
    print("[SKIP] полный прогон: нет модели или каталога DXA_TEST_DIR")

print("итог:", "FAIL" if FAILED else "OK")
sys.exit(1 if FAILED else 0)
