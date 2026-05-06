# Data Contract — CGR Crash Dataset

**Source:** Grand Rapids / Kent County crash records, 2008–2017  
**Rows:** 74,309 | **Columns used:** 24 features + 1 target  
**Sentinel value:** 999 in DRIVER1AGE and DRIVER2AGE → recoded to NaN before modelling

Null rates below are measured **after** sentinel recoding.  
`mostly` values are the GE `expect_column_values_to_not_be_null` thresholds — set with ~5% headroom above observed null rate to tolerate minor upstream variation without false alarms.

---

## Temporal

| Column | dtype | Valid range / values | Observed nulls | mostly |
|--------|-------|----------------------|----------------|--------|
| HOUR | int | 0–23 (99 = unknown time — sentinel, treated as out-of-range) | 0% | 1.0 |
| DAYOFWEEK | str | Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday | 0% | 1.0 |
| MONTH | str | January – December (12 values) | 0% | 1.0 |
| YEAR | int | 2008–2017 | 0% | 1.0 |

**Note — HOUR:** Raw data contains values of 99 representing unrecorded crash times. GE should flag these as out-of-range. They are not recoded before modelling (no sentinel handling configured for HOUR); expect a small fraction of rows with HOUR=99 to pass through as-is and be treated as a high numeric value by the scaler.

---

## Road & Environment

| Column | dtype | Valid range / values | Observed nulls | mostly |
|--------|-------|----------------------|----------------|--------|
| WEATHER | str | Blowing Snow, Clear, Cloudy, Fog, Rain, Severe Crosswind, Sleet or Hail, Smoke, Snow, Uncoded & Errors, Unknown | 0% | 1.0 |
| SURFCOND | str | Debris, Dry, Icy, Mud Dirt Gravel, Oily, Other, Slush, Snowy, Uncoded & Errors, Unknown, Water, Wet | 0% | 1.0 |
| LIGHTING | str | Dark Lighted, Dark Unlighted, Dawn, Daylight, Dusk, Other, Uncoded & Errors, Unknown | 0% | 1.0 |
| SPEEDLIMIT | int | 5–70 (0 = unrecorded; 99 = sentinel — both treated as out-of-range) | 0% | 1.0 |
| RDNUMLANES | int | 0–6 | 0% | 1.0 |
| RDWIDTH | float | 0–80 | 0% | 1.0 |
| ROUTECLASS | str | Connector, County Road or City Street or Not Known, Interstate Business Loop or Spur, Interstate Route, M Route, Not Located, U.S. Business Route, U.S. Route | 0% | 1.0 |
| TRUNKLINE | str | Non-Trunkline, Trunkline | 0% | 1.0 |
| RDSUBTYPE | int | 0, 30, 31, 33, 34, 35, 36, 37, 38 | 0% | 1.0 |

**Note — SPEEDLIMIT:** Value 0 appears for unrecorded speed limits; value 99 is a sentinel. Neither is a valid posted speed. Validate `min=5, max=70` with `mostly=0.99` to tolerate the small fraction of sentinel rows.

---

## Driver

| Column | dtype | Valid range / values | Observed nulls | mostly |
|--------|-------|----------------------|----------------|--------|
| DRIVER1AGE | int | 14–100 (sentinel 999 → NaN; 0 = unrecorded) | 12.2% | 0.85 |
| DRIVER1SEX | str | F, M, U | 0% | 1.0 |
| DRIVER2AGE | int | 14–100 (sentinel 999 → NaN; 117 observed — treat as data error, valid max=100) | 31.5% | 0.65 |
| DRIVER2SEX | str | F, M, U | 0% | 1.0 |

**Note — DRIVER2AGE:** A value of 117 was observed in the raw data — biologically impossible. The valid upper bound is set to 100. GE will flag any value > 100 as a violation; expect a very small number of such rows.

**Note — DRIVER1AGE/DRIVER2AGE null rates:** High nulls are structural — many crashes involve only one driver (DRIVER2AGE is absent), and age is frequently unrecorded for fleeing drivers (21.4% hit-and-run rate). The `mostly` thresholds reflect this expected missingness, not data quality failure.

---

## Vehicle

| Column | dtype | Valid values | Observed nulls | mostly |
|--------|-------|-------------|----------------|--------|
| VEH1TYPE | str | Go-cart / Golf Cart, Moped, Motorcycle, Motorhome, Off-Road Vehicle All-Terrain Vehicle, Other Non-Commercial, Passenger Car SUV Van, Pickup Truck, Truck / Bus (Commercial), Truck Under 10,000 lbs, Uncoded & Errors | 0% | 1.0 |
| VEH1USE | str | Club or Church, Commercial, Farm, In Pursuit or Emergency (in use), Military Vehicle, Other, Other Government Non-Emergency, Private, Road Construction or Maintenance, School or Education, Uncoded & Errors, Utility | 0% | 1.0 |
| VEH2TYPE | str | Go-cart / Golf Cart, Moped, Motorcycle, Motorhome, Off-Road Vehicle All-Terrain Vehicle, Other Non-Commercial, Passenger Car SUV Van, Pickup Truck, Snowmobile, Truck / Bus (Commercial), Truck Under 10,000 lbs, Uncoded & Errors | 0% | 1.0 |
| VEH2USE | str | Club or Church, Commercial, Farm, In Pursuit or Emergency (in use), Military Vehicle, Other, Other Government Non-Emergency, Private, Road Construction or Maintenance, School or Education, Uncoded & Errors, Utility | 0% | 1.0 |

---

## Crash Characteristics

| Column | dtype | Valid values | Observed nulls | mostly |
|--------|-------|-------------|----------------|--------|
| CRASHTYPE | str | 23 crash type codes (see profiling output) | 0% | 1.0 |
| TRAFCTLDEV | str | Signal, Stop Sign, Stop Sign with Flashing Beacon, Uncoded & Errors, Yield Sign | 57.0% | 0.40 |
| NONTRAFFIC | str | No | 0% | 1.0 |

**Note — TRAFCTLDEV:** 57% null is expected — only crashes at controlled intersections carry this field. The `mostly=0.40` threshold documents this as a known structural characteristic, not a data quality issue.

**Note — NONTRAFFIC:** Contains only the value `'No'` across all 74,309 rows — zero variance. This feature contributes nothing to model discrimination. It is retained in the pipeline for contract completeness but its inclusion in `params.yaml` features should be reconsidered before training.

---

## Target

| Column | dtype | Valid values | Observed nulls | mostly |
|--------|-------|-------------|----------------|--------|
| CRASHSEVER | str | Fatal, Injury, Property Damage Only | 0% | 1.0 |

**Binary encoding:** `PDO → 0` (majority class, 81.8%), `Injury or Fatal → 1` (minority class, 18.2%).  
Class weights: `w₀ = 0.61`, `w₁ = 2.74` (constitution III — no SMOTE).

---

## Feature Leakage Audit (Constitution I)

**Leakage definition:** Post-crash outcomes that are unknown at prediction time must never be used as model inputs.

### POST-CRASH COLUMNS (EXCLUDED — Leakage Risk)

The following columns represent crash outcomes and are **strictly prohibited** as features:

| Column | Reason | Status |
|--------|--------|--------|
| NUMOFKILL | Number of fatalities — crash outcome | ❌ EXCLUDED |
| NUMOFINJ | Number of injured — crash outcome | ❌ EXCLUDED |
| NUMOFUNINJ | Number of uninjured — crash outcome (T123a audit) | ❌ EXCLUDED |

**Audit finding (T123a):** `NUMOFUNINJ` confirmed as post-crash leakage. This column counts occupants who were **not** injured, which can only be determined after the crash occurs and medical assessments are complete.

### PRE-CRASH COLUMNS (SAFE — Available at Prediction Time)

The following columns represent conditions known **before** or **at the moment of** the crash and are safe to use:

| Column | Type | Reason | Status |
|--------|------|--------|--------|
| NUMOFVEHIC | Crash | Number of vehicles involved (T123a audit) | ✅ SAFE |
| NUMOFOCCUP | Crash | Number of occupants in vehicles | ✅ SAFE |
| SPEEDLIMIT | Road | Posted speed limit at crash location (T123a audit) | ✅ SAFE |
| DRIVER1AGE | Driver | Driver 1 age at time of crash (T123a audit) | ✅ SAFE |
| DRIVER2AGE | Driver | Driver 2 age at time of crash | ✅ SAFE |

**Audit findings (T123a):**
- `NUMOFVEHIC` — Safe. Vehicle count is a pre-crash condition (number of vehicles involved in the collision).
- `SPEEDLIMIT` — Safe. Posted speed limit is a road environmental attribute known before the crash.
- `DRIVER1AGE` — Safe. Driver age is a pre-crash demographic attribute.

**Note:** `NUMOFVEHIC` and `NUMOFOCCUP` are **not** currently in the active feature set (`params.yaml`) but are available for future feature engineering (e.g., danger index features in T123d).
