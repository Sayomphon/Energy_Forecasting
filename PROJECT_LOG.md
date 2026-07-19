# PROJECT_LOG — บันทึกการขึ้นโปรเจค Energy Forecasting

> บันทึกว่าทำอะไรไปบ้าง ตัดสินใจอะไร เพราะอะไร และจะไปต่ออย่างไร
> อ้างอิงแผนจาก `02_energy_forecasting_ai_engineering_plan.docx` (v1.0)

## 2026-07-19 (ต่อ) — Feature set v2 (lean): controlled experiment → promote (stretch #2)

ทำตามแผน `docs/V2_FEATURE_SET_PLAN.md` — ตั้งสมมติฐานจาก ablation v1 ("ตัด
weather/indoor sensors ทิ้งทั้งกลุ่มแล้ว backtest MAE ดีขึ้น 3.27 Wh") แล้ว
**พิสูจน์ด้วย controlled experiment ที่ตั้งเกณฑ์ตัดสินก่อนรัน** (กัน hindsight bias)

### สิ่งที่ทำ
1. **Toggle `feature_set` ใน `ForecastConfig`** (`v1`=full default, `v2`=lean) +
   property `include_sensor_features` + `__post_init__` validation (รับแค่ `v1`/`v2`)
   → serialize ลง `feature_config.json` อัตโนมัติผ่าน `to_dict()`
2. **`add_features` เคารพ toggle**: v2 ตัด indoor/outdoor aggregates + raw sensor
   loop ทั้งหมด **แต่คง negative controls `rv1`/`rv2` ไว้** (constant
   `NEGATIVE_CONTROL_COLS`) เพื่อให้ negative-control check ยัง valid ใน backtest →
   v2 = **25 คอลัมน์** (จาก 54): `load_now` + lag×4 + `seasonal_ref` + roll×12 +
   calendar×5 + rv1/rv2. v1 code path เดิม **bit-for-bit** (เงื่อนไขใหม่ไม่ trigger)
3. **`train.py --feature-set {v1,v2}`** + derive `model_version` (v2 →
   `energy-1h-v2`) + guard ให้ ablation ข้าม group ว่าง (v2 ไม่มี weather เหลือให้ drop)
4. **Tests +9** (`tests/test_feature_set_v2.py`): v2 ตัด sensor จริง / คง core+calendar /
   คง rv1/rv2 / เป็น strict subset ของ v1 / v1 ไม่ regress (assert exact ordered list) /
   anti-leakage (mutate-the-future) ซ้ำบน v2 / config roundtrip / reject feature_set
   ไม่ถูกต้อง → **รวม 64 tests ผ่านหมด**, ruff check + format สะอาด
5. **Controlled experiment** (`scripts/compare_feature_sets.py`): รัน v1/v2 บน
   rows/split/folds/selection **เดียวกันเป๊ะ** (ต่างแค่ feature columns) บน train+val
   region — **ไม่แตะ test**; เกณฑ์ตัดสิน hardcode ไว้ในสคริปต์ก่อนเห็นผล

### เกณฑ์ตัดสิน (ตั้งก่อนรัน, §3.2)
v2 promote ก็ต่อเมื่อ **ทั้งสอง**: (1) `mae_mean(v2)` ดีขึ้น ≥ 1.0 Wh **และ**
(2) `peak_mae_mean(v2)` แย่ลงไม่เกิน 5% (peak เป็นจุดอ่อนอยู่แล้ว ห้ามซ้ำเติม)

### ผล — v2 ชนะทั้ง validation และ test
**Backtest (3 expanding folds, hgb ชนะ baselines ทั้งคู่):**

| | features | MAE mean±std | WAPE | peak MAE mean±std | fit |
|---|---|---|---|---|---|
| v1 (full) | 54 | 43.96 ± 8.09 | 0.453 | 239.0 ± 11.16 | 1.10 s |
| **v2 (lean)** ✅ | 25 | **40.68 ± 4.18** | 0.419 | 237.9 ± 4.32 | 0.73 s |

- Criterion 1: MAE ดีขึ้น **+3.27 Wh** → **PASS**
- Criterion 2: peak MAE **−0.45%** (ดีขึ้นเล็กน้อยด้วย) → **PASS**
- **v2 std ลดลงเกือบครึ่ง** (mae 8.09→4.18, peak 11.16→4.32) — ตรงสมมติฐาน
  "ตัด sensor noise แล้วโมเดลนิ่งขึ้น" ชัดกว่าที่ตัวเลข ablation บอก และ fit เร็วขึ้น

**Final test (one-shot, n=2,938):** MAE **33.62** (v1: 34.04) · WAPE 0.347 ·
bias −15.73 · peak MAE **220.04** (v1: 220.44) — v2 ดีกว่า v1 บน test ที่ไม่เคยเห็นด้วย

→ **PROMOTE v2**: `train.py --feature-set v2` เต็ม (เปิด test ครั้งเดียว),
`model_version=energy-1h-v2`, regenerate `artifacts/` เป็น v2 canonical

### ข้อค้นพบเชิง methodology (สำคัญ)
- **ablation number generalize เป๊ะ**: v2 retrain MAE = 40.685 = ablation v1
  `drop_weather_sensors` (40.685) **ตรงทุกหลัก** เพราะ `ablation_delta` ในโปรเจคนี้
  *refit* บน column subset จริง (`backtest.py`: `model.fit(X[cols]...)`) ไม่ใช่ zero-out
  → concern ใน §3 ("ablation ≠ retrain") จึงไม่เกิดที่นี่ แต่ controlled experiment
  ยังมีค่า: ยืนยันซ้ำแบบอิสระ + เพิ่มการเช็ค **peak/std** ที่ ablation ไม่ได้ทำ
- **v2 ablation ชี้ทางต่อ**: `drop_rolling_stats` ทำ MAE ดีขึ้นอีก −0.63 Wh (ยังไม่ถึง
  เกณฑ์ 1.0) — rolling stats อาจเพิ่ม variance เล็กน้อยที่ horizon 1h → เก็บเป็น
  candidate v3 (คนละ experiment); `drop_negative_controls` +0.28 Wh (noise level, ผ่าน)

### ไฟล์ที่แตะ
- `src/.../config.py`, `features.py`, `train.py` (ตามข้อ 1–3)
- **สร้าง** `tests/test_feature_set_v2.py` (9), `scripts/compare_feature_sets.py`
- **regenerate** `artifacts/` เป็น v2 canonical + เพิ่ม `feature_set_comparison.csv`
  (หลักฐาน v1 vs v2 ทุกโมเดล); v1 baseline เก็บใน git history + comparison csv
- docs: `README.md` / `docs/model_card.md` (→ v2) / `V2_FEATURE_SET_PLAN.md` (status→done)

---

## 2026-07-19 — GitHub Actions CI: Phase 0 + workflow (stretch goal #1)

ทำตามแผน `docs/CI_PLAN.md` — quality gate อัตโนมัติที่ทำให้คำใน README
("temporal correctness is CI-enforced") เป็นจริง งานทั้งหมดเสร็จในเครื่องแล้ว
**ยังไม่ push** (รอสั่ง)

### สิ่งที่ทำ
1. **Phase 0 — รัน ruff จริงครั้งแรก** (ก่อนหน้านี้เขียนโค้ดตามกฎ ruff ในใจ แต่ยังไม่เคยรัน):
   - `ruff check .` เจอ **27 findings** auto-fixable ทั้งหมด → `--fix`:
     - **UP037** (เยอะสุด): ลบ quotes ที่เกินจำเป็นออกจาก type annotation
       (`folds: "list"` → `folds: list`, `-> "dict"` → `-> dict`) — ปลอดภัยเพราะทุกไฟล์
       มี `from __future__ import annotations` อยู่แล้ว (annotation เป็น lazy string
       ทั้งหมด ไม่ evaluate runtime แม้ `pd.Timestamp | str` บน py3.9)
     - **F401**: `import json` (inference.py) + `Path`/`numpy` (notebook) ที่ไม่ถูกใช้ → ลบ
       (verify แล้วว่าไม่มี usage จริงในทุก cell ของ notebook)
     - **E401/I001**: แยก multi-import + จัด import order (notebook bootstrap cell)
   - `ruff format .` จัด 10 ไฟล์ (line-wrap ของ function args เป็นหลัก) — review diff แล้ว
     ไม่แตะ logic; ยืนยัน dict comprehension ใน `load_bundle` + compileall ผ่าน
   - **Exit criteria ผ่านครบ**: `ruff check` = All checks passed! · `ruff format --check`
     = 15 files already formatted · `pytest` = **55 passed**

### 🔍 ข้อค้นพบ: pytest ค้าง 2 นาทีบน macOS — OpenMP thread oversubscription
pytest บนเครื่อง (macOS หลาย core) กิน CPU 559% (~5.6 cores) เพราะ
`HistGradientBoosting` เปิด OpenMP threads เต็มจำนวน core ทุกครั้งที่ train
(fixture `trained_bundle` function-scope × ~9 tests + backtest 4 models × 3 folds)
→ dataset เล็ก (2016 แถว) แต่ overhead ของ thread sync มากกว่าตัว compute

- **ไม่ใช่ bug / ไม่ใช่ network** — grep ยืนยัน tests ไม่แตะ network เลย (offline จริง)
- **แก้ที่การรัน local**: `OMP_NUM_THREADS=1` (+OPENBLAS/VECLIB/MKL) → **55 tests / ~10 วินาที**
- **CI (Linux ubuntu-latest, 2 core) ไม่เจอปัญหานี้** — จึงไม่ต้องใส่ env var ใน workflow;
  acceptance "< 3 นาที" ทำได้สบายบน CI

### ไฟล์ที่สร้าง / แก้
- **สร้าง** `.github/workflows/ci.yml` — 2 jobs: `lint` (ruff check + format, py3.11)
  → `test` (`needs: lint`, matrix py3.9–3.12, `fail-fast: false`); trigger push/PR (main)
  + `workflow_dispatch`; `concurrency.cancel-in-progress`; `permissions: contents: read`;
  cache pip; pin actions `@v4`/`@v5` (YAML validate ผ่าน)
- **README.md** — เพิ่ม CI badge ใต้ heading หลัก
- **docs/CI_PLAN.md** — status → implemented (local) + tick acceptance criteria
- **แก้จาก ruff**: src 8 ไฟล์ + tests 3 ไฟล์ + notebook (แค่ imports/format ไม่แตะ logic)

### ผลลัพธ์ — push + merge เสร็จ (CI เขียวครบ)
- [x] Push branch + เปิด [PR #1](https://github.com/Sayomphon/Energy_Forecasting/pull/1) →
  **CI เขียวครบ**: lint 33s · tests py3.9/3.10/3.11/3.12 = 52s/49s/43s/46s
- [x] Merge เข้า main (merge commit `9a16363`) + ลบ branch → badge passing → CI_PLAN criteria ครบ `[x]`
- **ยืนยันจริง:** OpenMP oversubscription เป็นปัญหา macOS local เท่านั้น —
  Linux runner แต่ละ job < 1 นาที (acceptance "< 3 นาที" ผ่านสบาย)

---

## 2026-07-18 (ต่อ) — Push GitHub + รัน notebook จบ (Definition of Done ครบ)

### สิ่งที่ทำ
1. **Push ขึ้น GitHub** [Sayomphon/Energy_Forecasting](https://github.com/Sayomphon/Energy_Forecasting)
   (public) — merge กับ initial commit เดิมแบบ `--allow-unrelated-histories`
   เก็บ LICENSE (Apache 2.0) ของ repo และใช้ README ฉบับเต็ม; แก้ license
   references ในโค้ดจาก MIT → Apache 2.0 ให้ตรงกัน; history สะอาดไม่ force push
2. **รัน notebook แบบ Restart & Run All** ด้วย `nbconvert --execute` ในเครื่อง
   (MacBook, CPU) — 19 cells รันครบตามลำดับ 1→19, **0 error**, ฝังกราฟ 4 รูป

### 🐛 บั๊กที่เจอจากการรัน notebook (CLI ไม่เจอ — นี่คือคุณค่าของ Run-all gate)
`predict_one()` ใช้ `pd.Timestamp(prediction_time)` ตรงๆ ซึ่ง parse timestamp
รูปแบบ UCI ที่ไม่มีช่องว่าง (`2016-05-2718:00:00`) ไม่ได้ → section 14 (robustness)
ที่ส่ง `raw["date"].iloc[-1]` (raw string) เข้าไป crash

**แก้ที่ source (ไม่ใช่แค่ notebook)**: เปลี่ยนไปใช้ `parse_timestamps()` ตัวเดียว
กับ pipeline ที่เหลือ → `predict_one` รับได้ทั้ง Timestamp, ISO string และ
UCI no-space string; เพิ่ม regression tests 3 ข้อ (ISO / UCI-format / unparseable)
→ รวมเป็น **55 tests ผ่านหมด** หลักการ "parse timestamp ที่เดียว" ตอนนี้ครอบคลุม
inference boundary ด้วย

### ✅ Definition of Done (บทที่ 13) + Final run checklist — ครบทุกข้อ
- Restart & Run all สำเร็จ ✅
- split/feature schema deterministic ตาม seed/version ✅
- model artifact reload → prediction สอดคล้อง (มี identity check ใน notebook) ✅
- ไม่มี secret/PII/dataset ที่ redistribute ไม่ได้ ✅
- **README / notebook / model card ใช้ตัวเลข metric จาก run เดียวกัน** ✅
  (notebook test_metrics ตรง bit-for-bit กับ `artifacts/test_metrics.json`:
  MAE 34.038… / peak MAE 220.44… / n=2938)

### เครื่องที่ใช้รัน
รันในเครื่อง (CPU) จบใน ~30 วินาที ไม่ต้องใช้ Colab — dataset 19,735 แถว/~12MB
+ โมเดล Ridge/HGB เบามาก RAM ใช้ไม่ถึง 1GB

### ขั้นถัดไป (optional / stretch — บทที่ 12)
- [ ] GitHub Actions CI (pytest + ruff อัตโนมัติ)
- [ ] Feature set v2: ตัด weather/sensor block ตามผล ablation แล้ว backtest ซ้ำ
- [ ] Quantile/conformal intervals, peak-event classifier, FastAPI endpoint

---

## 2026-07-18 — ขึ้นโปรเจคครั้งแรก (v1.0.0)

### สรุปผลลัพธ์

| รายการ | สถานะ |
|---|---|
| โครงสร้างโปรเจค + tooling (pyproject, ruff config, .gitignore, git) | ✅ |
| Library code 9 โมดูล (`src/energy_forecasting/`) | ✅ |
| Unit tests **55 ข้อ ผ่านทั้งหมด** (เน้น temporal correctness) | ✅ |
| Notebook 19 sections + รัน Restart & Run All จบ 0 error (มีกราฟฝัง) | ✅ |
| เอกสาร README / model card / บันทึกนี้ | ✅ |
| รัน pipeline จริงกับ UCI dataset จบ end-to-end + ตรวจ inference จาก bundle | ✅ |

### 1. โครงสร้างที่สร้าง

```
Energy_Forecasting/
├── src/energy_forecasting/    # package ติดตั้งได้ — temporal logic ทั้งหมดอยู่ที่นี่
│   ├── config.py              # ForecastConfig (frozen dataclass) — single source of truth
│   ├── data.py                # fetch/cache + lineage + data contract validation
│   ├── features.py            # past-only features + availability audit
│   ├── splits.py              # chronological split + expanding-window folds
│   ├── models.py              # baselines เป็น estimator จริง + Ridge + HGB
│   ├── metrics.py             # MAE/RMSE/WAPE/bias/peak-MAE + slice analysis
│   ├── backtest.py            # fold runner + selection rule + ablation
│   ├── inference.py           # bundle save/load (sha256) + forecast contract + fallback
│   └── train.py               # CLI: python -m energy_forecasting.train
├── notebooks/energy_1h_forecast.ipynb   # 39 cells / 19 sections
├── tests/                     # 47 tests
├── artifacts/                 # ผลลัพธ์ generated (bundle ไม่เข้า git)
├── docs/model_card.md
├── README.md                  # ภาษาอังกฤษ สำหรับ recruiter/reviewer
├── pyproject.toml             # deps + ruff (รวม bandit security rules) + pytest config
└── requirements.txt           # pinned versions ตาม environment จริง
```

**เหตุผลหลักที่แยก logic ออกจาก notebook**: docx เน้นว่า feature builder
ต้องเป็นชุดเดียวกันทั้ง train/inference — การมี implementation เดียวใน package
แล้วให้ notebook เรียกใช้ คือวิธีกำจัด training-serving skew ที่ตรวจสอบได้จริง

### 2. การตัดสินใจเชิงเทคนิคที่สำคัญ (design decisions)

1. **`load_now` เป็น feature แยกจาก `lag_1`** — ค่าที่ stamp เวลา t ถือว่ารู้แล้ว ณ t
   จึงใช้ได้ (คือ persistence baseline ในตัว) ส่วน lag_k = ค่าที่ t−k step ตาม docx
2. **Seasonal naive align กับเวลา target ไม่ใช่เวลา t** — "same time yesterday"
   ของค่าที่จะพยากรณ์ (t+1h) คือ t+1h−24h = shift(138) ไม่ใช่ shift(144)
   จุดนี้ docx เขียนกำกวม เราเลือกตีความที่ถูกต้องตามความหมายจริง และมี unit test ยืนยัน
3. **Rolling ทุกตัว `shift(1)` ก่อน aggregate** — ตาม docx ทุก window เป็นอดีตล้วน
4. **Peak threshold มาจาก train quantile (q90) เท่านั้น** — evaluation ห้ามแอบดู label ตัวเอง
5. **Selection rule ตายตัวก่อนเปิด test set**: candidate ต้องชนะ baseline
   ทุกตัวเสียงข้างมากของ folds → เลือก mean MAE ต่ำสุด → ถ้าสูสี (<2%)
   เลือกตัวที่ peak MAE ต่ำกว่า และถ้าไม่มีใครชนะ baseline → เลือก baseline (honest failure)
6. **HGB ใช้ `loss="absolute_error"`** ให้สอดคล้องกับการประเมินด้วย MAE/WAPE
7. **ไม่ใช้ LSTM/Transformer** ตาม scope guardrail ของ docx
8. **Splits deterministic จาก timestamp ล้วน** — seed มีผลเฉพาะการ fit โมเดล

### 3. มาตรการด้านความปลอดภัย (security)

- **Artifact integrity**: bundle เป็น joblib pickle (deserialize = execute code ได้)
  → `save_bundle` เขียน sha256 sidecar และ `load_bundle` **ปฏิเสธ** ไฟล์ที่
  checksum ไม่ตรงหรือไม่มี sidecar (มี test ยืนยันการ reject ไฟล์ที่ถูกแก้)
- **Data integrity**: dataset cache มี lineage (source, retrieval date, sha256)
  และ `load_raw` verify hash ทุกครั้ง; hash ของข้อมูล train ฝังใน bundle
- **Input validation ที่ inference**: ตรวจ schema, ตัดข้อมูลหลัง prediction_time ทิ้ง
  (hard guard), history เก่าเกิน 30 นาที / สั้นเกิน / คอลัมน์หาย → fallback
  พร้อม quality_flag — ไม่มีทางส่ง NaN เข้าโมเดลเงียบๆ
- **Ruff เปิดกฎ `S` (bandit)** ใน pyproject.toml สำหรับ security lint
- **.gitignore**: กัน data/, bundle, .env, key/pem — ไม่มี secret/PII เข้า repo
- **ไม่ redistribute dataset** — fetch จากแหล่งทางการตอน build (CC BY 4.0 + attribution ครบ)

### 4. Unit tests (52 ข้อ — ผ่านทั้งหมด)

| กลุ่ม | ตัวอย่างที่สำคัญ |
|---|---|
| Target shift | target ตรง 6 steps เป๊ะ (เทียบ manual lookup), ตัดแถวท้ายที่ไม่มี label |
| **Anti-leakage** | **mutate ข้อมูลอนาคตทั้งหมดแล้ว feature ณ t ต้องไม่เปลี่ยน** (test สำคัญที่สุด), rolling ใช้ window อดีตล้วน, ไม่มี feature ไหน correlate 1.0 กับ label |
| Data contract | ค่าติดลบ→fail, คอลัมน์หาย→fail, gap→report แต่ tolerate |
| Splits | train<val<test เชิงเวลา (และจับ shuffle ได้), folds ขยายแบบ expanding ไม่ overlap |
| Metrics | ค่าที่คำนวณมือ, bias sign convention, peak ใช้ threshold ภายนอก, WAPE denominator=0 |
| Inference | save→load แล้ว prediction เหมือนเดิมทุก bit, ไฟล์ถูก tamper→reject, ป้อนข้อมูลอนาคตปน→ผลไม่เปลี่ยน, stale/short history→fallback พร้อม flag |

รัน: `PYTHONPATH=src python3 -m pytest tests/` (หรือ `pytest` หลัง `pip install -e .`)

หมายเหตุ: มี RuntimeWarning จาก sklearn/Accelerate BLAS บน macOS ระหว่าง matmul
ของ Ridge — เป็น warning ภายในไลบรารีที่ทราบกันบน macOS ไม่กระทบความถูกต้องของผลลัพธ์

### 5. Notebook (`notebooks/energy_1h_forecast.ipynb`)

19 sections ตาม blueprint: overview → environment → ingestion → data contract →
EDA → **leakage audit** → split → preprocessing → baselines → candidates →
comparison/selection → one-shot test eval → error slices → robustness (จำลอง
stale/short history จริง) → explainability + negative controls → packaging →
inference demo (cold start จาก disk) → monitoring plan → limitations

ทุก section มี markdown อธิบายเหตุผลก่อน code และข้อสรุปตาม design rule ของ docx

### 6. ผลการรัน pipeline จริง (2026-07-18)

**Incident ที่เจอและวิธีแก้** — ตอนรันครั้งแรก data contract **fail ทันที**:
timestamp จาก ucimlrepo มาในรูป `2016-01-1117:00:00` (วันที่ติดกับเวลา ไม่มีช่องว่าง)
ทั้ง 19,735 แถว → เพิ่ม `parse_timestamps()` ใน `data.py` ที่ลอง explicit format
ทีละแบบทั้งคอลัมน์ (all-or-nothing ไม่มี half-parse) ใช้ร่วมกันทั้ง contract/
prepare/inference พร้อม test 5 ข้อ **หลักการ: raw cache คงข้อมูลตามต้นฉบับ
การ clean อยู่ในโค้ดที่ version และ test ได้** — และนี่คือตัวอย่างจริงว่า
data contract จับปัญหาก่อนถึง training

**ผล backtest (3 expanding folds, MAE เป็น Wh):**

| Model | MAE mean±std | WAPE | Peak MAE | ชนะ baselines |
|---|---|---|---|---|
| **hgb** ✅ | **43.96 ± 8.09** | 0.453 | 239.0 | 3/3 folds |
| ridge | 54.25 ± 9.67 | 0.559 | 227.0 | 2/3 folds |
| last_value | 55.38 ± 2.98 | 0.570 | 252.8 | — |
| seasonal_naive | 61.39 ± 4.43 | 0.632 | 261.8 | — |

**Final test (one-shot, ~3 สัปดาห์สุดท้าย, n=2,938):**
MAE 34.04 · RMSE 79.68 · WAPE 0.351 · bias −14.36 · peak MAE 220.44 (threshold 200 = train q90)

**ข้อค้นพบที่ต้องพูดตรงๆ:**
1. **Peak คือจุดอ่อน** — peak MAE 220 เทียบ MAE รวม 34 และ bias ติดลบ
   แปลว่า under-forecast ช่วง spike อย่างเป็นระบบ → ต่อยอดด้วย quantile regression
2. **Ablation หักมุม**: ตัด weather/indoor sensors ทิ้งทั้งกลุ่ม MAE *ดีขึ้น* 3.27 Wh
   → ที่ horizon 1 ชั่วโมง สัญญาณอยู่ที่ lag/rolling/calendar เป็นหลัก ควรทำ v2 ที่ feature ผอมลง
3. **Negative controls ผ่าน**: ตัด rv1/rv2 แล้ว MAE ขยับแค่ ~0.5 Wh (ระดับ noise)
4. **Residual ACF lag-1 = 0.64** — ยังมี short-term structure เหลือให้เก็บ

**Inference ตรวจแล้วครบทุก path**: load bundle จาก disk (ผ่าน sha256) → forecast
ปกติได้ contract ครบ / feed stale → `stale_history_fallback` / history สั้น →
`insufficient_history_fallback` — ไม่มี path ไหนส่ง NaN เข้าโมเดล

### 7. สิ่งที่ยังไม่ได้ทำ / ขั้นถัดไป

- [ ] รัน notebook จบ (Restart & Run all gate ตาม Definition of Done)
- [ ] Commit แรกเข้า git (init branch `main` ไว้แล้ว ยังไม่ commit — รอสั่ง)
- [ ] Feature set v2: ตัด weather/sensor block ตามผล ablation แล้ว backtest ซ้ำ
- [ ] Stretch (docx บทที่ 12): quantile/conformal intervals, peak-event classifier,
      MLflow tracking, CI (GitHub Actions รัน pytest + ruff)

### 8. วิธีใช้งานย่อ

```bash
pip install -e ".[dev]"                      # ติดตั้ง package + dev tools
pytest                                       # 47 tests
python -m energy_forecasting.train --fetch   # end-to-end: fetch→contract→backtest→select→package
```

Artifacts ที่ได้ใน `artifacts/`: `forecast_bundle.joblib` (+`.sha256`),
`feature_config.json`, `backtest_metrics.csv`, `backtest_summary.csv`,
`model_selection.json`, `test_metrics.json`, `test_slice_errors.csv`,
`residual_summary.json`, `data_contract_report.json`, `backtest_calendar.csv`
