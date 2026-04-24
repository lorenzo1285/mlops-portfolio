# EDA Findings — CGR Crash Data

**Dataset:** Grand Rapids, Kent County, Michigan — 74,309 crashes across 142 variables
**Period:** January 2008 – December 2017

---

## 1. Dataset Overview

The dataset is single-city (Grand Rapids) and single-county (Kent), so geographic variation is limited to road-level differences within the city. Roughly 9% of rows carry null values concentrated in driver-age fields (sentinel value 999 was recoded to NaN) and a handful of road-geometry columns.

---

## 2. Crash Severity

| Severity | Count | Share |
|---|---|---|
| Property Damage Only | 60,761 | 81.8% |
| Injury | 13,443 | 18.1% |
| Fatal | 105 | 0.14% |

The overwhelming majority of crashes result in no injury. However, the absolute injury count (17,157 individuals) and 109 fatalities over a decade represent a meaningful public-safety burden for a single city.

---

## 3. Volume Trend

Crash volume grew steadily throughout the decade, rising 25% from 7,113 in 2008 to 8,898 in 2017. The increase is consistent year-over-year with no sharp discontinuities, suggesting it tracks population/traffic growth rather than a data artefact.

---

## 4. Temporal Patterns

**Time of day:** Crashes peak sharply during afternoon rush hour — hours 15, 16, and 17 (3–5 pm) account for the highest volume. Fatal crashes follow a different pattern, peaking at hours 13, 19, 22, and 2 am, suggesting impairment and higher speeds outside rush hour.

**Day of week:** Friday is the busiest day (12,546 crashes). Volume drops significantly on weekends, with Sunday the lowest (7,240). Weekday commuter traffic is the dominant driver.

**Month:** January (7,415) and December (7,502) are the highest-volume months, reflecting winter driving conditions. There is a secondary dip in spring/summer (April–August), counter-intuitive but consistent with fewer icy-road multi-car pile-ups.

---

## 5. Environmental Conditions

**Weather:** 50% of crashes occur in clear conditions, reflecting that most driving happens in good weather. However, snow (10.8%) and rain (10.8%) together account for more than a fifth of crashes despite representing far less than a fifth of total driving time — indicating elevated risk per mile.

**Surface condition:** 62.6% dry, but icy (7.0%) and snowy (8.3%) surfaces are disproportionately represented relative to their share of annual road-hours.

**Lighting:** 66.9% of crashes occur in daylight. Dark-lighted roads account for 23.4%, and dark-unlighted for 3.5%. The fatality-rate difference between lit and unlit darkness is worth further modelling.

---

## 6. Crash Types & Hazardous Actions

Rear-end straight collisions are the most common crash type (21,746 — 29.3%), followed by side-swipe same direction (11,214 — 15.1%) and angle straight (8,703 — 11.7%). Fixed-object crashes (7,276) are notable as they tend to involve single vehicles at higher speeds or under impairment.

The leading driver hazardous actions are:

| Action | Count |
|---|---|
| Fail to Stop / ACD | 14,251 |
| Failed to Yield | 8,597 |
| Speed Too Fast | 6,283 |
| Disobeyed Traffic Control Device | 2,572 |
| Improper Lane Use | 2,420 |

---

## 7. High-Risk Subgroups

**Hit-and-run:** 15,898 crashes (21.4%) — strikingly, more than 1 in 5 crashes involves a driver fleeing the scene. This is the single most common "flag" in the dataset and likely suppresses injury-severity coding for those records.

**Alcohol-involved:** 3,261 crashes (4.4%). Among drinking-involved crashes, the injury rate jumps to 33.0% and the fatal rate to 1.4% — roughly 10× the dataset-wide fatal rate (0.14%).

**Pedestrians:** 1,070 crashes (1.4%), with 85.7% resulting in injury and 2.4% fatal — the highest fatal share of any subgroup analysed.

**Cyclists:** 926 crashes (1.2%), with 79.9% resulting in injury and 1.1% fatal.

**Motorcycles:** 713 crashes (1.0%) — injury/fatality breakdown is expected to be similarly elevated given exposure.

---

## 8. Speed Limit & Fatality Risk

Crashes at higher posted speed limits carry a materially higher fatality rate, even though the 25 mph zone accounts for the most crashes in absolute terms (29,900 — predominantly urban streets):

| Speed Limit | Crashes | Fatalities | Fatal Rate |
|---|---|---|---|
| 50 mph | 402 | 1 | 0.25% |
| 45 mph | 6,913 | 16 | 0.23% |
| 40 mph | 2,006 | 4 | 0.20% |
| 70 mph | 9,627 | 19 | 0.20% |
| 25 mph | 29,900 | 36 | 0.12% |

---

## 9. Driver Demographics

The mean driver age is 36.8 years (median 32). Male drivers (36,025) outnumber female (30,769) in involvement, with a meaningful unknown/unrecorded group (7,515). Age-bucketed analysis shows the 16–25 cohort has a slightly elevated injury rate relative to older drivers, consistent with inexperience and risk-taking behaviour.

---

## 10. Key Takeaways

1. **Volume is rising** — a 25% decade-on-decade increase without a clear structural intervention suggests ongoing risk growth.
2. **Rush hour drives frequency; nights/weekends drive severity** — resource deployment strategies should separate frequency reduction (pm peak) from fatality reduction (night, weekend).
3. **Hit-and-run is endemic** — at 21.4%, it is far above national averages and warrants dedicated investigation.
4. **Alcohol involvement carries 10× the fatal risk** — a small fraction of crashes (4.4%) but a disproportionate share of fatalities.
5. **Vulnerable road users face the highest injury probability** — pedestrians (85.7% injury rate) and cyclists (79.9%) are severely over-represented in serious outcomes relative to their share of crashes.
6. **Winter conditions multiply risk** — snow and ice together represent ~15% of crashes but a disproportionate share given actual exposure time.
