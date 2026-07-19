# แผนงาน: Feature Set v2 (lean) — ขับด้วย ablation

> Stretch goal #2 — ปรับ feature set ตามหลักฐาน ablation แล้วพิสูจน์ด้วย controlled experiment
> สถานะ: **done** — v2 ผ่านเกณฑ์ §3.2 ทั้งสองข้อ → promoted เป็น `energy-1h-v2` (2026-07-19) · อ้างอิง: `artifacts/feature_set_comparison.csv`, `PROJECT_LOG.md` · ดูผลจริงท้ายเอกสาร

---

## 1. เป้าหมาย (Why)

Ablation ตอน v1 ให้ผลที่ "หักมุม" — **ตัด weather/indoor sensor features ทิ้งทั้งกลุ่ม แล้ว backtest MAE *ดีขึ้น***:

| Ablation (drop group) | mean MAE | Δ vs full |
|---|---|---|
| full_feature_set (v1) | 43.956 | 0.000 |
| drop_rolling_stats | 44.326 | +0.370 (แย่ลง) |
| drop_calendar | 46.852 | +2.896 (แย่ลงมาก — calendar สำคัญ) |
| **drop_weather_sensors** | **40.685** | **−3.271 (ดีขึ้น!)** |
| drop_negative_controls | 44.456 | +0.500 (noise level — ผ่าน) |

**สมมติฐาน:** ที่ horizon 1 ชั่วโมง สัญญาณพยากรณ์อยู่ที่ **recent-load dynamics (lag/rolling) + calendar** เป็นหลัก ส่วน raw sensor readings (T1–T9, RH_*, weather, indoor aggregates) เพิ่ม **noise/variance** มากกว่า signal → ตัดออกแล้วโมเดลนิ่งขึ้น

**คุณค่าเชิง engineering (จุดขาย portfolio):**
- ไม่ใช่แค่ "train ครั้งเดียวจบ" แต่ **วิเคราะห์ → พบ insight → ออกแบบ experiment → พิสูจน์/หักล้าง** = iteration loop ที่ขับด้วยข้อมูล
- โมเดล v2 **เบากว่า** (ฟีเจอร์น้อยลง ~30 คอลัมน์ → ~15), inference เร็วขึ้น, พึ่ง sensor น้อยลง = robust ต่อ sensor outage ในโลกจริง

> ⚠️ **ประเด็นสำคัญที่สุด (methodology):** ablation number (−3.27) มาจากการ *drop group ออกจากโมเดลที่ fit ด้วยฟีเจอร์เต็ม* ซึ่ง**ไม่เท่ากับ**การ retrain โมเดลด้วย lean feature set ตั้งแต่ต้น (HGB หา split/binning ต่างกันเมื่อ feature เปลี่ยน) → **ต้อง backtest v2 ใหม่เต็ม ไม่เชื่อ ablation number ตรงๆ**

---

## 2. Scope

**In scope**
- เพิ่ม toggle ใน `ForecastConfig` เพื่อเลือก feature set (v1 = full, v2 = lean) — reproducible + serialize ลง `feature_config.json`
- `add_features` เคารพ toggle (ตัด sensor block เมื่อ v2)
- **Controlled experiment**: backtest v1 vs v2 บน folds เดียวกัน, selection rule เดียวกัน
- เปิด test **ครั้งเดียว** กับผู้ชนะบน validation → รายงานเทียบ
- Tests สำหรับ v2 (feature set ถูกตัดจริง + ยังผ่าน anti-leakage + v1 ไม่ regress)
- อัปเดต docs (README/model card/PROJECT_LOG) ตาม**ผลจริง** (ไม่ว่าจะชนะหรือไม่)

**Out of scope**
- HGB hyperparameter tuning (คนละ experiment — คุม variable ให้เหลือแค่ feature set)
- Feature ใหม่ (weather forecast feed, holiday) — อยู่ใน stretch #3
- เปลี่ยน model family

---

## 3. Hypothesis & methodology (หัวใจของงานนี้)

### 3.1 สิ่งที่ต้องระวังไม่ให้ leak
- Ablation ทำบน **train+val folds** (validation) → การ *ตัดสินใจลองv2* มาจาก validation ✅ ไม่แตะ test
- แต่การเลือก v1/v2 = model selection → **test เปิดครั้งเดียวหลัง lock ผู้ชนะจาก validation** (หลักการเดียวกับ v1 เป๊ะ)
- **ห้าม** ลอง feature set หลายแบบแล้วเลือกอันที่ดีสุด*บน test* — นั่นคือ test leakage; decision ต้องจบที่ validation ก่อน

### 3.2 เกณฑ์ตัดสิน (ตั้งก่อนรัน — กัน hindsight bias)
v2 จะถูก **promote เป็น default** ก็ต่อเมื่อ **ทั้งสอง**เงื่อนไขจริงบน backtest folds:
1. `mae_mean(v2) < mae_mean(v1)` อย่างน้อย **1.0 Wh** (มากกว่า noise; ablation บอก ~3.3)
2. `peak_mae_mean(v2)` ไม่แย่ลงเกิน **5%** จาก v1 (peak คือจุดอ่อนอยู่แล้ว ห้ามซ้ำเติม)

- ถ้าผ่าน → v2 เป็น default, `model_version = "energy-1h-v2"`, retrain + update artifacts/docs
- ถ้า **ไม่ผ่าน** (ablation เป็น false signal เมื่อ retrain จริง) → **รายงานตามตรงใน PROJECT_LOG**, เก็บ v1 เป็น default, บันทึกว่า "ablation ไม่ generalize สู่ full retrain" ← นี่คือ honest science และก็เป็นผลลัพธ์ที่มีค่าพอกัน

> จุดขายคือ**กระบวนการ** ไม่ใช่การบังคับให้ v2 ชนะ — reviewer ที่เก่งจะเห็นค่าของการตั้งเกณฑ์ก่อนแล้วรายงานตามจริง

---

## 4. Design การเปลี่ยนโค้ด

### 4.1 `config.py` — เพิ่ม toggle
```python
feature_set: str = "v1"   # "v1" = full (default, backward-compatible), "v2" = lean
```
- คง v1 เป็น default → ไม่ทำลาย reproducibility ของผลเดิม / tests เดิม
- serialize อยู่แล้วผ่าน `to_dict()` → ติดไปกับ `feature_config.json` และ bundle อัตโนมัติ
- (ทางเลือก) property `include_sensor_features -> bool` = `self.feature_set == "v1"` เพื่ออ่านง่าย

### 4.2 `features.py` — เคารพ toggle
ครอบ 2 บล็อกท้ายของ `add_features` (บรรทัด ~89–113: indoor/outdoor aggregates + raw contemporaneous sensor loop) ด้วยเงื่อนไข:
```python
if cfg.include_sensor_features:   # v1 เท่านั้น
    # ... indoor_T_mean / diff_* / raw sensor loop ...
```
**ผลลัพธ์ v2 feature set** = `load_now`, `lag_{1,6,12,144}`, `seasonal_ref`, `roll_*` (12 คอลัมน์), calendar (5), + negative controls `rv1/rv2`
- **หมายเหตุ rv1/rv2**: ยังอยู่ใน v2 (เข้าทาง raw loop) — ถ้าอยากได้ "production-clean" set ให้เพิ่ม toggle `include_negative_controls` แยก แต่**แนะนำเก็บ rv ไว้ใน backtest** เพื่อคง negative-control check; จะตัดเฉพาะตอน package production

### 4.3 `train.py` — รองรับการเทียบ
- เพิ่ม arg `--feature-set {v1,v2}` (default v1) → ส่งเข้า `ForecastConfig(feature_set=...)`
- ไม่ต้องเขียน experiment loop ใน train.py; ใช้สคริปต์เทียบแยก (ดูข้อ 5) เพื่อไม่ให้ entry point หลักรก

### 4.4 model_version
- เปลี่ยนเป็น `energy-1h-v2` **เฉพาะเมื่อ** v2 ผ่านเกณฑ์และถูก promote

---

## 5. Experiment protocol (step-by-step)

1. **Prereq**: push+merge CI ก่อน (ดูข้อ 8) แล้วแตก branch `feat/feature-set-v2` จาก main
2. Implement §4.1–4.3 + tests §6 → `ruff check/format` + `pytest` เขียวในเครื่อง
3. รันเทียบ (สคริปต์ scratch หรือ cell — ใช้ dataset cache เดิม, **ไม่แตะ test**):
   ```
   for fs in [v1, v2]:
       features(fs) → chronological_split (เดิม) → expanding folds (เดิม)
       → run_backtest(ทุกโมเดล) → summarize
   เทียบ mae_mean / peak_mae_mean ตามเกณฑ์ §3.2
   ```
4. **ตัดสินตามเกณฑ์ที่ตั้งไว้:**
   - v2 ชนะ → `train.py --feature-set v2` เต็ม (เปิด test ครั้งเดียว) → artifacts v2 + model_version v2
   - v2 ไม่ชนะ → เก็บ v1, รายงานผลลบตามตรง
5. อัปเดต docs ด้วยตัวเลขจริงจากทั้งสอง run (ตาราง v1 vs v2)
6. Commit + push branch → PR (ได้ CI gate) → merge

---

## 6. Test plan (เพิ่มใน `tests/`)
- `test_v2_drops_sensor_features`: `feature_set="v2"` → `feature_cols` ไม่มี `indoor_*`, `diff_*`, `T_out`, `RH_out`, `T1..`, `lights` ฯลฯ
- `test_v2_keeps_core_features`: v2 ยังมี `load_now`, `lag_*`, `seasonal_ref`, `roll_*`, calendar ครบ
- `test_v1_unchanged`: `feature_set="v1"` (default) → feature_cols เท่าเดิมทุกคอลัมน์ (กัน regression)
- `test_v2_still_leakage_safe`: รัน anti-leakage test (mutate future) ซ้ำบน v2 → ต้องผ่าน
- `test_config_feature_set_roundtrip`: v2 config serialize→load แล้วเท่าเดิม

> anti-leakage บน v2 สำคัญ: ยืนยันว่าการตัด feature ไม่ทำให้ availability guarantee เสีย

---

## 7. Acceptance criteria (Definition of Done)
- [x] `config.feature_set` toggle + serialize ทำงาน; v1 เป็น default ที่ backward-compatible
- [x] `add_features` เคารพ toggle; v2 ตัด sensor block จริง (พิสูจน์ด้วย test)
- [x] Tests ใหม่ผ่าน + ชุดเดิม 55 ข้อไม่ regress (v1 เหมือนเดิม bit-for-bit) — รวม **64 tests** ผ่าน
- [x] Backtest v1 vs v2 เทียบ fair (folds/selection เดียวกัน) — ตารางใน PROJECT_LOG + `feature_set_comparison.csv`
- [x] ตัดสินตามเกณฑ์ §3.2 **ที่ตั้งไว้ก่อนรัน**; test เปิดครั้งเดียว → **v2 ชนะ, promoted**
- [x] PROJECT_LOG + README + model card อัปเดตด้วยผลจริง (v2 ชนะ → promote v2)
- [x] ผ่าน CI บน PR — [PR #2](https://github.com/Sayomphon/Energy_Forecasting/pull/2) เขียวครบ (lint + tests py3.9–3.12)

---

## 8. Dependencies & ข้อควรระวัง

| ประเด็น | การจัดการ |
|---|---|
| **ควรทำหลัง CI merge** | v2 branch ควรแตกจาก main *หลัง* push+merge `ci/github-actions` เพื่อให้ v2 PR ได้ CI gate คุ้มครอง (ตอนนี้ CI ยัง local) |
| **ablation ≠ retrain** | §3 ย้ำ — ต้อง backtest v2 เต็ม ไม่เชื่อ −3.27 ตรงๆ |
| **test leakage** | decision จบที่ validation; test เปิดครั้งเดียวกับผู้ชนะ |
| **peak MAE regress** | เกณฑ์ §3.2 ข้อ 2 กันไว้ — v2 ที่ MAE ดีขึ้นแต่ peak แย่ลงมากไม่ผ่าน |
| **reproducibility เดิม** | v1 เป็น default → artifacts/tests/notebook เดิมไม่เปลี่ยน; v2 เป็น opt-in |
| **negative controls** | เก็บ rv1/rv2 ใน backtest เพื่อคง control; ตัดเฉพาะตอน package prod (ถ้าเลือก) |
| **notebook** | ถ้า promote v2 → เพิ่ม cell เทียบ v1/v2 ใน section 11 (comparison) ให้ story ครบ |

---

## 9. ไฟล์ที่จะแตะ
- `src/energy_forecasting/config.py` — เพิ่ม `feature_set` + property
- `src/energy_forecasting/features.py` — ครอบ sensor block ด้วย toggle
- `src/energy_forecasting/train.py` — เพิ่ม `--feature-set`
- `tests/test_temporal_features.py` (หรือไฟล์ใหม่ `test_feature_set_v2.py`) — 5 tests §6
- `artifacts/` — regenerate ถ้า promote v2 (v1 canonical ถ้าไม่)
- `docs/model_card.md`, `README.md`, `PROJECT_LOG.md`, เอกสารนี้ (status→done)

---

## สรุป
v2 ไม่ใช่แค่ "ตัดฟีเจอร์ให้ MAE ดีขึ้น" แต่เป็น **controlled experiment ที่ตั้งเกณฑ์ก่อน แล้วตัดสินด้วยหลักฐานจาก validation โดยไม่แตะ test** — ถ้า v2 ชนะก็ได้โมเดลที่ดีขึ้นจริงและเบาลง; ถ้าไม่ชนะก็ได้บทเรียนว่า ablation signal ไม่ generalize ซึ่งมีค่าพอกันในเชิง engineering ทั้งสองผลลัพธ์เอาไปเล่าในสัมภาษณ์ได้

---

## ผลลัพธ์จริง (2026-07-19) — v2 ชนะ, promoted

**ผ่านทั้งสองเกณฑ์ §3.2 → `energy-1h-v2`:**

| | v1 (full, 54) | v2 (lean, 25) | Δ |
|---|---|---|---|
| Backtest MAE mean±std | 43.96 ± 8.09 | **40.68 ± 4.18** | **−3.27** (เกณฑ์ ≥1.0 ✅) |
| Backtest peak MAE mean | 239.0 | 237.9 | −0.45% (เกณฑ์ ≤+5% ✅) |
| Test MAE (one-shot, n=2,938) | 34.04 | **33.62** | −0.42 (ดีกว่าบน test ด้วย) |
| Test peak MAE | 220.44 | 220.04 | −0.40 |

- **std ลดเกือบครึ่ง** (8.09→4.18; peak 11.16→4.32) และ fit เร็วขึ้น — ตัด sensor noise → โมเดลนิ่งขึ้น ตรงสมมติฐาน §1
- **Methodology note**: v2 retrain MAE (40.685) = ablation v1 `drop_weather_sensors` (40.685) **ตรงเป๊ะ** เพราะ `ablation_delta` ในโปรเจคนี้ *refit* บน column subset จริง (ไม่ใช่ zero-out) → ablation signal generalize สู่ full retrain; controlled experiment ยืนยันซ้ำแบบอิสระ + เพิ่มการเช็ค peak/std ที่ ablation ไม่ได้ทำ
- **ทางต่อ (v3 candidate)**: v2 ablation ชี้ว่า `drop_rolling_stats` ทำ MAE ดีขึ้นอีก −0.63 Wh (ยังไม่ถึงเกณฑ์ 1.0) — คนละ experiment

รายละเอียด: `PROJECT_LOG.md` (entry 2026-07-19), `artifacts/feature_set_comparison.csv`, `scripts/compare_feature_sets.py`
