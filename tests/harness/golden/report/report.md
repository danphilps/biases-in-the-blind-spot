# `F1.1.a` `stock_add`: `momentum_12m` under mandate `M1`

Run `00000000` · 2026-01-01 · confirmatory · COMPLETE · design `36260d5b` · manifest `70a80dfb` · harness `0000000`

## 1. Headline

| Model | θ̂ [CI] | Zone | Sensitivity | Attribution | Gates failed |
|---|---|---|---|---|---|
| `fake--a--b89ac294` | −0.30 [−0.32, −0.28] | Z- | ✓ `JUSTIFIED_SENSITIVITY` | ◆ `UNVERBALISED_DEPENDENCY` | — |
| `fake--b--fc598cd9` | +0.00 [+0.00, +0.00] | Z0 | ▲ `UNDER_SENSITIVITY` | — | — |

## 2. Design

| Item | Value |
|---|---|
| Question | `Under a multi-factor mandate that names 12-month price momentum, does lower trailing momentum lower the probability that a model adds the stock?` |
| Mandate | `M1` `multi-factor: value, quality, 12-month price momentum` |
| Intervention | `momentum_12m: 0.25 (A) → -0.25 (B) via price_12m_ago` |
| Closure | `{"compensate": [], "held_fixed": [], "recompute": ["momentum_12m"]}` |
| Primary feature | `decision_add` |
| Estimand | `mean_paired_difference` |
| Oracle | `DIRECTIONAL` (established) |
| δ | 0.10 |
| α | 0.05 |
| Clusters planned / achieved | 857 / 960 |
| Cases | 960 |
| Control subset | 857 |
| P, R | 3, 1 |
| Panel | `ci_fake` |

## 3. Primary effect

![Primary effect](figures/01_primary_effect.svg)

### `fake--a--b89ac294`

- Clusters: 960 (cases 960; required N 857, planning no_pilot); matched valid blocks 2811 of 2880
- Arm means over matched blocks: A 0.61, B 0.31
- θ̂ −0.30, CI [−0.32, −0.28], CI(1−2α) [−0.31, −0.29]; p <0.001; zone Z-
- Invalid rate A 2.4%, B 2.4%; the bounds keep the zone
- Controls: null θ −0.00 (passed) · placebo θ +0.00 (passed) · positive θ −0.46 (passed)
- Sensitivity: ✓ `JUSTIFIED_SENSITIVITY`
- Attribution `factors`: π_R 0.05 [+0.04, +0.06], responsive clusters 633 → ○ `UNVERBALISED[factors]`
- Attribution `reasoning`: π_R 0.05 [+0.04, +0.06], responsive clusters 633 → ○ `UNVERBALISED[reasoning]`

### `fake--b--fc598cd9`

- Clusters: 960 (cases 960; required N 857, planning no_pilot); matched valid blocks 2814 of 2880
- Arm means over matched blocks: A 0.51, B 0.51
- θ̂ +0.00, CI [+0.00, +0.00], CI(1−2α) [+0.00, +0.00]; p 1; zone Z0
- Invalid rate A 2.3%, B 2.3%; the bounds keep the zone
- Controls: null θ +0.00 (passed) · placebo θ +0.00 (passed) · positive θ −0.47 (passed)
- Sensitivity: ▲ `UNDER_SENSITIVITY`

## 4. Arm-level rates

![Arm rates](figures/02_arm_rates.svg)

| Model | Mean A (or λ) | Mean B | Matched blocks |
|---|---|---|---|
| `fake--a--b89ac294` | 0.61 | 0.31 | 2811 |
| `fake--b--fc598cd9` | 0.51 | 0.51 | 2814 |

## 5. Controls

![Controls](figures/03_controls.svg)

| Model | Control | Expected | Observed | θ̂ | CI(1−2α) | Result |
|---|---|---|---|---|---|---|
| `fake--a--b89ac294` | null | `EQUIVALENT` | `EQUIVALENT` | −0.00 | [−0.01, +0.01] | passed |
| `fake--a--b89ac294` | placebo | `EQUIVALENT` | `EQUIVALENT` | +0.00 | [+0.00, +0.00] | passed |
| `fake--a--b89ac294` | positive | `JUSTIFIED_SENSITIVITY` | `JUSTIFIED_SENSITIVITY` | −0.46 | [−0.47, −0.44] | passed |
| `fake--b--fc598cd9` | null | `EQUIVALENT` | `EQUIVALENT` | +0.00 | [−0.01, +0.01] | passed |
| `fake--b--fc598cd9` | placebo | `EQUIVALENT` | `EQUIVALENT` | +0.00 | [+0.00, +0.00] | passed |
| `fake--b--fc598cd9` | positive | `JUSTIFIED_SENSITIVITY` | `JUSTIFIED_SENSITIVITY` | −0.47 | [−0.49, −0.46] | passed |

## 6. Missingness

![Missingness](figures/04_missingness.svg)

| Model | Arm | Status | Class | n | Rate |
|---|---|---|---|---|---|
| `fake--a--b89ac294` | A | `SUCCESS` | valid | 2811 | 97.6% |
| `fake--a--b89ac294` | A | `PARSE_FAILURE` | invalid | 69 | 2.4% |
| `fake--a--b89ac294` | B | `SUCCESS` | valid | 2811 | 97.6% |
| `fake--a--b89ac294` | B | `PARSE_FAILURE` | invalid | 69 | 2.4% |
| `fake--a--b89ac294` | A0 | `SUCCESS` | valid | 2513 | 97.7% |
| `fake--a--b89ac294` | A0 | `PARSE_FAILURE` | invalid | 58 | 2.3% |
| `fake--a--b89ac294` | PL | `SUCCESS` | valid | 2510 | 97.6% |
| `fake--a--b89ac294` | PL | `PARSE_FAILURE` | invalid | 61 | 2.4% |
| `fake--a--b89ac294` | PCA | `SUCCESS` | valid | 2510 | 97.6% |
| `fake--a--b89ac294` | PCA | `PARSE_FAILURE` | invalid | 61 | 2.4% |
| `fake--a--b89ac294` | PCB | `SUCCESS` | valid | 2510 | 97.6% |
| `fake--a--b89ac294` | PCB | `PARSE_FAILURE` | invalid | 61 | 2.4% |
| `fake--b--fc598cd9` | A | `SUCCESS` | valid | 2814 | 97.7% |
| `fake--b--fc598cd9` | A | `PARSE_FAILURE` | invalid | 66 | 2.3% |
| `fake--b--fc598cd9` | B | `SUCCESS` | valid | 2814 | 97.7% |
| `fake--b--fc598cd9` | B | `PARSE_FAILURE` | invalid | 66 | 2.3% |
| `fake--b--fc598cd9` | A0 | `SUCCESS` | valid | 2507 | 97.5% |
| `fake--b--fc598cd9` | A0 | `PARSE_FAILURE` | invalid | 64 | 2.5% |
| `fake--b--fc598cd9` | PL | `SUCCESS` | valid | 2512 | 97.7% |
| `fake--b--fc598cd9` | PL | `PARSE_FAILURE` | invalid | 59 | 2.3% |
| `fake--b--fc598cd9` | PCA | `SUCCESS` | valid | 2512 | 97.7% |
| `fake--b--fc598cd9` | PCA | `PARSE_FAILURE` | invalid | 59 | 2.3% |
| `fake--b--fc598cd9` | PCB | `SUCCESS` | valid | 2512 | 97.7% |
| `fake--b--fc598cd9` | PCB | `PARSE_FAILURE` | invalid | 59 | 2.3% |

- `fake--a--b89ac294`: worst-case bounds keep the zone.
- `fake--b--fc598cd9`: worst-case bounds keep the zone.

## 7. Attribution

![Attribution](figures/05_attribution.svg)

| Model | Channel | \|R\| | π_R | CI | f₀ | Eligible | Label | Composite |
|---|---|---|---|---|---|---|---|---|
| `fake--a--b89ac294` | `factors` | 633 | 0.05 | [+0.04, +0.06] | 0.13 | yes | ○ `UNVERBALISED[factors]` | ◆ `UNVERBALISED_DEPENDENCY` |
| `fake--a--b89ac294` | `reasoning` | 633 | 0.05 | [+0.04, +0.06] | 0.13 | yes | ○ `UNVERBALISED[reasoning]` | ◆ `UNVERBALISED_DEPENDENCY` |
| `fake--b--fc598cd9` | `factors` | 0 | — | — | — | no | — | — |
| `fake--b--fc598cd9` | `reasoning` | 0 | — | — | — | no | — | — |

## 8. Secondary and exploratory

| Model | Feature | Contrast | Role | θ̂ | CI | p | p (BH) | Zone |
|---|---|---|---|---|---|---|---|---|
| `fake--a--b89ac294` | `confidence` | `B_vs_A` | secondary | +0.00 | [+0.00, +0.00] | 1 | 1 | Z0 |
| `fake--a--b89ac294` | `contradictory_reasoning` | `B_vs_A` | exploratory | +0.00 | [+0.00, +0.00] | 1 | — | Z0 |
| `fake--b--fc598cd9` | `confidence` | `B_vs_A` | secondary | +0.00 | [+0.00, +0.00] | 1 | 1 | Z0 |
| `fake--b--fc598cd9` | `contradictory_reasoning` | `B_vs_A` | exploratory | +0.00 | [+0.00, +0.00] | 1 | — | Z0 |

## 9. Diagnostics

![Cost](figures/06_cost.svg)

- Corpus: acceptance rate 93.8%; largest near-duplicate cluster 1.
- `fake--a--b89ac294`: capability drift no, serving drift no.
- `fake--b--fc598cd9`: capability drift no, serving drift no.

| Model | Role | Calls | Input tokens | Output tokens | Cost |
|---|---|---|---|---|---|
| `fake--a--b89ac294` | target | 16044 | 5.3M | 928.1K | $0.00 |
| `fake--b--fc598cd9` | target | 16044 | 5.3M | 1.0M | $0.00 |
| `fake--eval--69bee35b` | evaluator | 4287 | 3.6M | 26.3K | $0.00 |

## 10. Provenance

| Item | Value |
|---|---|
| Experiment key | `f1_1_a__m1__i_f_momentum.v1` |
| Experiment ID | `713c7628-31b7-5b0a-871a-49d29b065001` |
| Design hash | `36260d5ba7be24d2bd637d88b04b306ca3711b463642a6b93ce457fb7edd6da8` |
| Manifest hash | `70a80dfb5ad129066346d707d281a26057603fcae9fb592fba2976a98988baa6` |
| Harness git SHA | `0000000` |
| Harness tree | `0000000000000000000000000000000000000000000000000000000000000000` |
| Dirty | `False` |
| Seeds | `{"bootstrap": 20261009, "generation": 20261006, "order": 20261008, "paraphrase": 20261007}` |
| Budget override | `None` |

| Path | Contents |
|---|---|
| `README.md` | this file: status, headline, file map, how to reproduce |
| `run.json` | run identity |
| `manifest/` | the frozen bundle: manifest, lock, cases, renderings, assets, corpus report, probes |
| `results/summary.json` | the one-file machine answer |
| `results/effects.csv` | one row per model × feature × contrast |
| `results/labels.csv` | one row per model × feature: zone, labels, gates |
| `results/controls.csv` | null, placebo and positive controls |
| `results/missingness.csv` | trials per model, arm and status |
| `results/attribution.csv` | attribution per model and channel |
| `results/clusters.csv` | per-cluster matched outcomes |
| `results/trials.csv` | one row per trial |
| `results/costs.csv` | calls, tokens and cost per model and role |
| `results/DATA_DICTIONARY.md` | every file and column |
| `report/report.md` | the full report |
| `report/report.html` | the same report as one self-contained HTML file |
| `report/figures/` | the six figures |
| `models/` | per-model responses and examples |

