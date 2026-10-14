# Scientific readiness register

Status of the 2026-10-14 repository audit ("Financial LLM Bias Benchmark: repository audit and comprehensive improvement plan"), item by item. Spec change CR-9 (`docs/financial_harness_spec.md` §41) records the code and specification changes. The audit's backlog IDs (CR-01 to CR-26) are a separate numbering from the spec's change log (CR-1 to CR-9).

Rule for this register: an item is **done** only when code, spec and tests landed together. An item that needs scientific content (stimuli, banks, oracle values, tolerances, labelled data) or a user decision is **blocked**, never filled by the coding agent (`AGENTS.md` rules 1 and 2; D-14, D-23).

## Summary

| Status | Items |
|---|---|
| Done in CR-9 | CR-01, CR-07 (classification part), CR-08 (ceiling rule), CR-09 (S2 scoring), CR-14, CR-16, CR-18, CR-02 (prioritisation only) |
| Already in the harness before the audit | Matched-block pairing and cluster inference (§23), TOST equivalence (§23.3), per-model BH in two claim families (§25), randomised block and arm order (§18), separate δ and control margins (§22), UNVERBALISED_DEPENDENCY vs UNVERBALISED_BIAS (§24.3), DISCOVERY labels for M0 and unestablished oracles (§24.2), frozen bundles (§0.10) |
| Blocked on a user decision | CR-03 (D-26), CR-04 (D-2), CR-05 (D-3), CR-10, CR-13 (pilot budget) |
| Blocked on human-authored content | CR-02 (the authoring itself), CR-06 (semantic review), CR-07 (validated banks, D-25, D-27), CR-11, CR-12, CR-15, CR-17, CR-20 to CR-26 (D-23) |

## P0: required before live V1

| ID | Audit action | Status | What landed / what is needed |
|---|---|---|---|
| CR-01 | Live programme = the seven scientific specs; S1-cal separate | **Done** | `programmes/live.yaml` selects S1 m0/m1/m2 and S2–S5; `programmes/calibration_live.yaml` holds the two S1-cal specs. The 2026-10-12 cost estimate ($3,780, 142,536 calls) applied to the old five-spec selection and is withdrawn; re-run `programme plan`. The caps ($4,000, 150,000 calls) are the user's (D-2) and were not changed, so `plan` will refuse the programme if the new upper bound exceeds them. |
| CR-02 | Human authoring and approval of the 116 inventoried files, prioritised by V1 dependency | **Prioritised; authoring blocked** | `AUTHORING_INVENTORY.md` now tags each file: 61 block `live.yaml`, 5 block only `calibration_live.yaml`, 50 block neither. The authoring and approval are human work (D-14). |
| CR-03 | Two-call decision-then-attribution protocol | **Blocked: D-26** | D-26 is PROVISIONAL single-call `rationale_after`. A two-call trial changes the frozen bundle (§0.10: the second-turn template is prompt-visible text that must be human-written), the runner (multi-turn trial, two parses, two attempts per trial), replay (§31) and the cassettes. Needs the user to resolve D-26 and a human-written second-turn template. Not started, so as not to build against an unresolved decision. |
| CR-04 | Independent evaluator from a third family | **Blocked: D-2** | `panels/live.yaml` has `evaluator: null`. The model, provider key and probe are the user's choice. Until set, S1 (semantic `rationale` attribution) and S2 (semantic primary) fail `validate` on the live panel. |
| CR-05 | Human calibration (≈ 400 balanced items) and adjudication | **Blocked: D-3** | `calibrate` and the export/merge tooling exist (`taxonomy_build/export_calibration_items.py`). Labellers, item counts and the adjudication process are the user's. The committed calibration records are fake-evaluator software tests only. |
| CR-06 | Audit every intervention pair for structural, economic and semantic isolation | **Partly in place; semantic review blocked** | Structural: `validate` already checks the visible diff and leak terms (§9.1); the canonical fixtures in `tests/harness/fixtures/canonical/` hold arm A/B states and renderings for three cases per spec. This change bumped every V1 key to version 3 and the regenerated fixtures' cases are byte-identical, so no stimulus moved. Economic closure rules beyond the DAG (accounting identities) and the semantic review need the expert panel. |
| CR-07 | Repair strict-invariance cue banks; complete D-25 | **Classification done; banks blocked** | New D-27: I.ID.name, I.ID.ceo, I.ID.ceo_ethnicity, I.PRES.order, I.USER.stance, I.HYPE.label and I.CODE.comment are DISCOVERY until validated (§T4.2). Their battery experiments are now discovery mode, `confidence: contested`, and can never set `is_bias` or UNVERBALISED_BIAS. Only I.PRES.distractor (S2) stays a confirmatory INVARIANCE cue in V1. The validated banks (D-25, D-27) are human content. |
| CR-08 | Positive controls with headroom; demonstrate power | **Ceiling rule done; power demonstration needs a pilot** | §22: for a LEVEL primary, a positive control whose PCA already meets τ is SATURATED and does not gate (`oracles.saturate_positive_control`; `ANALYSIS_VERSION` 2). Empirical power needs the pilot of CR-13. |
| CR-09 | S2 inclusion scoring; S3–S5 competence/bias separation | **S2 done; S3–S5 already separated** | S2's rule now reads the whole requested output (`summary` and `key_points`, one sentence or entry at a time; `output_inclusion_rule@2`), and the evaluator sees both fields. The v1 rule is kept so stored runs re-evaluate unchanged. S3–S5 primaries are LEVEL (competence), their B − A contrasts DISCOVERY (D-21), and LEVEL labels are never `is_bias` (§24.2). |
| CR-10 | Freeze the validated package, model revisions and evaluator assets | **Mechanism exists; blocked on CR-02 to CR-05** | `freeze` already hashes cases, renderings, assets and code identity (§0.10). A preregistration manifest can only be frozen once the content and evaluator exist. |

## P1: scientific quality and V1.1

| ID | Audit action | Status | What landed / what is needed |
|---|---|---|---|
| CR-11 | Financially coherent companies and transcripts | **Blocked: D-23, D-24** | Sector-conditional distributions, company-name banks consistent with the sector, and transcript realism are scenario content. The S1 name/sector mismatch the audit found is in the agent-placeholder bank `fundamental_judgement/banks/company_names.yaml`, which a person must replace. |
| CR-12 | Remove response-format priming (unprompted contracts) | **Blocked: D-14** | An unprompted S3 contract is prompt-visible text. S1 already has no factor menu (CR-8 of the spec). |
| CR-13 | Pilot-based power and variance estimation | **Blocked: budget (D-2)** | `statistical_spec.planning_ref` already consumes a frozen discovery pilot (§22, §23.6). A pilot programme needs discovery-mode copies of the specs on disjoint seeds and cases (§2.6) and a budget. |
| CR-14 | Cross-mandate matched analysis | **Done** | `programme report` writes `results/cross_variant.csv`: γ = mean_c(d_c^x − d_c^ref), matched by cluster across two runs of one item and intervention on the identical case set (fingerprint of case index, cluster and arm A/B states), with the §23.3 t-test and CI. Reference = the item's first variant, so S1 gives γ₁₀ = θ_M1 − θ_M0 and γ₂₀ = θ_M2 − θ_M0. Exploratory: never a label and never in the FDR families. The salience-control rendering (audit §4.7) is prompt-visible content and is not done. |
| CR-15 | Extended point-in-time and numerical code tests | **Blocked: D-23** | New hidden tests (revised data, reporting lags, NaNs, unsorted inputs) are oracle content and must also be justified by the public task text (§T6.1). |
| CR-16 | Generated siblings' research questions match their intervention | **Done** | `taxonomy expand` derives each generated spec's research question from its own intervention (relevant pairs) or cue (battery), never the base spec's. |
| CR-17 | Expert-panel review of V1.1 content | **Blocked: D-13, D-23** | |
| CR-18 | Simulation-based false-positive and power tests | **Done** | `statistics/operating.py` simulates the audit's scenario matrix through the production rules; `tests/harness/test_harness_operating_characteristics.py` is a CI gate: false bias rate ≤ α at θ = 0 and at a sub-threshold θ, power ≥ 0.80 for each intended label, a broken pairing is UNSTABLE and never a bias, unevaluable attribution never yields UNVERBALISED_BIAS, a model correct in every arm is never INSTRUMENT_INSENSITIVE. |
| CR-19 | Attribution uncertainty, adjudication status and competence classes in reports | **Partly in place** | π_R and f₀ CIs, channel and eligibility are already in `attribution.csv`; LEVEL labels already separate competence. A human-adjudication column needs the adjudication process of CR-05. |

## P2: representative benchmark (CR-20 to CR-26)

Blocked on D-23 (scientific content: new domains, oracles, generators and fixtures) and D-13 (expert panel). No code milestone blocks them: M9a's infrastructure reports each missing input in `generated/expand_report.json`.

## Audit points not adopted, with reasons

- **"Live programme omits the transcript-summary experiment."** Correct for the old file; fixed by CR-01.
- **"Financial oracles mix behavioural expectations, numerical thresholds and financial correctness."** The harness already separates oracle types (§17) and confidence (`established`, `conventional`, `contested`), and only `established` confirmatory oracles can set `is_bias`. The audit's proposed extra fields (`oracle_scope`, `oracle_assumptions`, `oracle_limitations`) would hold oracle documentation, which is human content; the schema change is deferred until the expert panel writes them.
- **New label names (COMPETENCE_FAILURE, DISCOVERED_DEPENDENCY, OVER/UNDER_SENSITIVITY).** The existing labels already cover these: BELOW_STANDARD is a competence failure, SENSITIVE under DISCOVERY is a discovered dependency, OVER_SENSITIVITY and UNDER_SENSITIVITY exist (§24.2). Renaming would break stored runs' labels without adding information.
- **Audit §2 (S4) "rolling windows and cross-sectional normalisation can still leak".** True, but deciding validity from publication timestamps needs timing data the S4 brief does not contain; adding it is scenario content (D-23). S5's executable HT.LEAK test is the deterministic leakage oracle.
