# แผนงาน: GitHub Actions CI

> Stretch goal #1 — automated quality gate สำหรับ [Sayomphon/Energy_Forecasting](https://github.com/Sayomphon/Energy_Forecasting)
> สถานะ: **done ✅ — CI เขียวครบ py3.9–3.12 บน GitHub** ([PR #1](https://github.com/Sayomphon/Energy_Forecasting/pull/1) merged, commit `9a16363`) · Phase 0 ผ่านครบ (ruff เขียว + 55 tests) · เสร็จ 2026-07-19

---

## 1. เป้าหมาย (Why)

| เป้าหมาย | รายละเอียด |
|---|---|
| **ป้องกัน regression** | รัน 55 tests ทุก push/PR โดยเฉพาะ anti-leakage tests ที่เป็นหัวใจ — ถ้าใครแก้โค้ดแล้วเผลอทำ future leakage กลับมา CI จับได้ทันที |
| **ทำให้คำโฆษณาเป็นจริง** | README เขียนว่า *"temporal correctness is CI-enforced"* แต่ตอนนี้ยังไม่มี CI จริง — แผนนี้ปิด gap นั้น |
| **Lint อัตโนมัติ** | บังคับ `ruff` (รวม bandit security rules `S`) ทุก commit ให้ code quality สม่ำเสมอ |
| **Portfolio signal** | badge **CI passing** สีเขียวใน README ที่ reviewer เห็นทันที = โปรเจคมี engineering discipline จริง |

---

## 2. Scope

**In scope**
- Lint: `ruff check` (E/F/I/B/UP/S rules ที่ตั้งไว้ใน `pyproject.toml`)
- Format check: `ruff format --check`
- Unit tests: `pytest` (55 ข้อ) บน Python matrix
- รันแบบ **offline** — tests ใช้ synthetic fixtures (`tests/conftest.py`) ไม่แตะ network

**Out of scope (ตอนนี้)**
- รัน `train.py` / integration กับ UCI dataset จริง (ต้อง network → ช้า/ไม่เสถียร ไม่เหมาะกับ CI)
- Notebook execution (ใช้เวลานาน + ต้อง dataset) — พิจารณาเป็น smoke test แยกภายหลัง
- Coverage upload, deployment, release automation

> **หลักคิด:** CI ต้อง **เร็วและ deterministic** — เพราะ tests แยกจาก network อยู่แล้ว (fetch_raw ถูก import แบบ lazy ภายใน function) CI จึงรันได้ในไม่กี่วินาทีโดยไม่ต้องโหลด dataset

---

## 3. Phase 0 — Pre-requisite (⚠️ สำคัญ ต้องทำก่อน)

**เหตุผล:** เราเขียนโค้ดตาม ruff rules ในใจ แต่**ยังไม่เคยรัน `ruff` จริง** ถ้าเปิด CI เลยโดยไม่เช็คก่อน workflow จะ**แดงทันทีในรอบแรก** ซึ่งดูไม่ดี ต้อง resolve findings ในเครื่องให้เขียวก่อน

```bash
# 1. ติดตั้ง dev deps (รวม ruff)
pip install -e ".[dev]"

# 2. ดู lint findings ทั้งหมด
ruff check .

# 3. auto-fix ที่แก้อัตโนมัติได้ (import order, ฯลฯ)
ruff check --fix .

# 4. findings ที่เหลือ (เช่น S-rules) แก้มือหรือใส่ # noqa: <CODE> พร้อมเหตุผล
#    - S301 (pickle/joblib.load) ที่ inference.py มี # noqa อยู่แล้ว
#    - ตรวจว่าไม่มี finding ค้าง

# 5. format ทั้ง repo ให้ผ่าน format --check
ruff format .

# 6. ยืนยันทั้งสองผ่าน + tests ยังเขียว
ruff check . && ruff format --check . && pytest -q
```

**Exit criteria ของ Phase 0:** ทั้ง 3 คำสั่ง (`ruff check`, `ruff format --check`, `pytest`) ผ่านหมดในเครื่อง → พร้อมเปิด CI

> จุดที่ต้องเฝ้าดู: `ruff format` อาจจัด format โค้ดต่างจากที่เขียนไว้ (เช่น string quotes, line wrap) — review diff ก่อน commit ว่าไม่เปลี่ยน logic

---

## 4. Workflow design

- **ไฟล์:** `.github/workflows/ci.yml`
- **Triggers:** `push` (main), `pull_request` (main), `workflow_dispatch` (รันเองได้)
- **Jobs:**
  - `lint` — ruff check + format check (รันครั้งเดียว, Python 3.11, เร็ว)
  - `test` — matrix Python 3.9–3.12, `pytest` (รันหลัง lint ผ่าน)
- **Concurrency:** ยกเลิก run เก่าเมื่อ push ใหม่ (ประหยัด CI minutes)
- **Security:** `permissions: contents: read` (least privilege) + pin action versions

### เหตุผลการเลือก

| การตัดสินใจ | เหตุผล |
|---|---|
| `lint` แยกจาก `test` และเป็น `needs` | ถ้า lint พังก็ไม่ต้องเปลืองรัน matrix 4 versions |
| Matrix 3.9–3.12 | `requires-python = ">=3.9"`; numpy 2.0/sklearn 1.6 รองรับถึง 3.12 — พิสูจน์ว่าใช้ได้ทุก version ที่ประกาศ |
| `ubuntu-latest` | เร็ว/ฟรี/เพียงพอ (โมเดลเป็น CPU-only) — ไม่ต้อง macOS runner |
| `cache: pip` ของ setup-python | ลดเวลา install โดยไม่ต้องใช้ `actions/cache` แยก |
| `fail-fast: false` | ให้เห็นผลครบทุก Python version แม้ version หนึ่งพัง |

---

## 5. ไฟล์ workflow (พร้อมใช้)

สร้าง `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

permissions:
  contents: read

jobs:
  lint:
    name: Lint & format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
          cache-dependency-path: pyproject.toml
      - name: Install
        run: python -m pip install -e ".[dev]"
      - name: Ruff lint
        run: ruff check --output-format=github .
      - name: Ruff format check
        run: ruff format --check .

  test:
    name: Tests (py${{ matrix.python-version }})
    needs: lint
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
          cache-dependency-path: pyproject.toml
      - name: Install
        run: python -m pip install -e ".[dev]"
      - name: Run tests
        run: pytest -q
```

---

## 6. README badge

เพิ่มบรรทัดใต้ heading หลักของ `README.md`:

```markdown
[![CI](https://github.com/Sayomphon/Energy_Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/Sayomphon/Energy_Forecasting/actions/workflows/ci.yml)
```

---

## 7. ขั้นตอน implement (step-by-step)

1. **Phase 0** — รัน ruff/format/pytest ในเครื่องให้เขียวหมด (ดูข้อ 3), commit fixes ถ้ามี
2. สร้าง `.github/workflows/ci.yml` (ข้อ 5)
3. เพิ่ม badge ใน README (ข้อ 6)
4. Commit + push บน branch (แนะนำ `ci/github-actions` ไม่ push ตรง main)
5. เปิด PR → ดู CI รันบน GitHub → ต้องเขียวทั้ง `lint` และ `test` ครบ 4 versions
6. Merge เข้า main เมื่อเขียว → badge ใน README เปลี่ยนเป็น passing

---

## 8. Acceptance criteria (Definition of Done)

- [x] `.github/workflows/ci.yml` มีอยู่และ trigger บน push/PR ของ main
- [x] Job `lint` เขียว (ruff check + format ผ่าน) — *GitHub: pass 33s*
- [x] Job `test` เขียวครบ Python 3.9 / 3.10 / 3.11 / 3.12 — *GitHub: pass 52s / 49s / 43s / 46s*
- [x] Badge CI passing แสดงใน README
- [x] CI รันจบใน < 3 นาที และไม่แตะ network (ไม่โหลด dataset) — *แต่ละ job 33–52s บน Linux runner; tests ใช้ synthetic fixtures*
- [x] เอกสารนี้อัปเดตสถานะเป็น **done** + บันทึกใน PROJECT_LOG.md

> ทุกข้อผ่านครบ — verified บน GitHub Actions [run 29686809215](https://github.com/Sayomphon/Energy_Forecasting/actions/runs/29686809215) (PR #1)

---

## 9. ข้อควรระวัง & ความเสี่ยง

| ประเด็น | การจัดการ |
|---|---|
| **ruff findings ยังไม่เคยรัน** | Phase 0 บังคับ resolve ในเครื่องก่อน (ความเสี่ยงหลัก) |
| **`ruff format` เปลี่ยน style โค้ด** | review diff ก่อน commit; ถ้าไม่อยากบังคับ format ตอนแรก ตัด step `format --check` ออกได้ แล้วเพิ่มทีหลัง |
| **tests เผลอแตะ network บน CI** | ยืนยันว่า tests ใช้ synthetic fixtures เท่านั้น (`fetch_raw` import ucimlrepo แบบ lazy — ไม่ถูกเรียกใน test ใด) |
| **Supply-chain security** | pin actions ที่ major version (`@v4`, `@v5`); ตั้ง `permissions: contents: read` |
| **macOS BLAS RuntimeWarning** | เป็น warning เฉพาะ macOS/Accelerate — CI รันบน Linux ไม่เจอ ไม่กระทบ |
| **CI minutes** | `concurrency.cancel-in-progress` + lint-before-test ลดการเปลืองโควตา |

---

## 10. Future enhancements (ไม่อยู่ใน scope นี้)

- **Coverage**: `pytest --cov` + upload (Codecov) + coverage badge
- **pre-commit hook**: รัน ruff ก่อน commit ในเครื่อง (shift-left)
- **Notebook smoke test**: execute notebook ด้วย dataset ขนาดเล็ก/mock เป็น job แยก (nightly)
- **Dependabot**: อัปเดต action versions + deps อัตโนมัติ
- **Release workflow**: tag → build + attach artifacts

---

## สรุป

CI นี้เป็น **quality gate ที่เร็วและ offline** — จุดสำคัญที่สุดคือ **Phase 0** (รัน ruff ในเครื่องก่อน) เพราะเป็นสิ่งเดียวที่จะทำให้รอบแรกเขียวหรือแดง ส่วน workflow YAML พร้อมใช้ copy ได้เลยหลัง Phase 0 ผ่าน
