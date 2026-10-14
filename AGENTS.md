# AGENTS.md

## The aim: find undeclared biases in LLMs that answer finance questions

This project tests large language models for **undeclared biases** when they answer finance and quantitative-finance questions: fundamental equity judgement, valuation, earnings analysis, corporate actions, broker research, quant model design, machine learning for stock selection, and quant code.

An **undeclared bias** is present when all three of these hold:

1. **The answer depends on a factor.** Changing one input, and nothing else, moves the model's answer.
2. **The dependence is wrong.** By an established standard (the mandate, an accounting identity, a planted ground truth, a cue that is irrelevant by construction, or a published result), the factor should not move the answer, or should move it the other way or by less.
3. **The model does not say so.** Its stated reasons do not mention the factor.

When all three hold, the harness assigns the label `UNVERBALISED_BIAS`. Every experiment answers three questions per model:
- Does it move when it should?
- Does it stay put when it should not?
- Does it say what moved it?

**Every decision you make in this repository serves that test. Nothing you do may introduce a bias of your own into it.** If a choice could make a reported bias a property of the stimulus, the oracle, the statistics or your code rather than of the model under test, it is the wrong choice.

## The build instruction

`docs/financial_harness_spec.md` is the only build instruction for `financial_llm_harness/`.
- Read §1 (objective), §2 (non-negotiable rules), §38 (build order) and §40 (governing principle) before writing code.
- Build the milestones in the §38 order: M0 → M1 → M2 → M3 → M10 → M4 → M5 → M6 → M7 → M8 → M9a. A milestone is done when its done-test exits 0 and the CI command still exits 0.
- Where the spec and existing code disagree, the spec governs new code. The legacy package `biases_in_the_blind_spot/` changes only as spec §0.5 allows, and its tests stay green.
- Never weaken, skip or delete a test to make it pass.

## Rules that protect the aim

1. **Never author what the model is tested on.** You are an LLM, and your model family may be in the panel under test. Text you write is attributed to your family (§2.7).
   - You MUST NOT write prompt-visible text or oracle values for live use: templates, paraphrases, banks, mandates, oracle rules, tolerances, scenario distributions or fixtures.
   - On the fake path you MAY write placeholders, only for content the spec defines, marked `drafted_by: {kind: llm, family: <yours>}` (§8.1). `validate` refuses them for live runs.
2. **Never invent scientific data.** When an expansion lacks data, `taxonomy expand` reports it as `unimplemented_content`, with its `missing` paths (§32B.2 step 7, §T11). Leave it missing and report it; the user's expert panel supplies it (D-23).
3. **Freeze before calling.** No model call happens until the experiment bundle is generated, validated and frozen. The bundle holds the cases, the exact rendered prompts, the asset hashes and the code identity (§0.10). If an input drifts, the run stops (`FROZEN_INPUT_DRIFT`).
4. **Test the right unit.** Compare arms only within matched (case, paraphrase, repetition) blocks. Infer at the cluster level (§23). Never treat trials as independent.
5. **Say "bias" only when the classifier assigns it.** The word appears only in a label assigned by §24: confirmatory mode and an established oracle. Sensitivity is not bias.
6. **Keep controls honest.** Every confirmatory experiment runs a null, a placebo and a positive control, sized for power (§22). Never shrink them to save cost.
7. **Be reproducible.** Re-evaluating stored responses is byte-identical, and every reported number traces to one CSV row (§31, §32A.8).
8. **Protect data and keys.** Only synthetic data goes to external providers unless `provider_policy.yaml` allows otherwise. Licensed and confidential runs are never committed (§0.9, §30.1). API keys never appear in artefacts.

## Working conventions

- **Python and tooling:** Python 3.12, managed with `uv`.
  - CI command: `uv sync --all-extras && uv run ruff format --check && uv run ruff check && uv run pyright && uv run pytest && uv build && uv run financial-harness --help`.
  - Type hints on every function. Dataclasses for data, frozen when the value enters a hash.
  - YAML is read and written only through `core/yaml_io.py` (§0.11).
- **Tests:** harness tests live in `tests/harness/test_harness_<topic>.py` (§36). Provider calls in tests use cassettes, never the network. Live tests need `HARNESS_LIVE=1`.
- **Ask, don't guess:** user-owned decisions are listed in spec §39.1. If a task needs one that is unresolved, stop and ask.
- **Git:**
  - Commit only what the spec's §0.9 commit policy allows.
  - Never push to a branch other than the one you were given.
  - Never rewrite shared history.

## Where things are

| Path | What it is |
|---|---|
| `docs/financial_harness_spec.md` | The specification: Part I is how experiments run; Part II is what is assessed |
| `financial_llm_harness/` | The harness, built from the spec (it does not exist until M0) |
| `biases_in_the_blind_spot/concept_pipeline/` | The original paper's concept pipeline (legacy; keep it runnable) |
| `experiments/archive/`, `biases_in_the_blind_spot/archive/` | The original paper's experiments and datasets (loan approval, university admission, hiring), archived: not part of the harness; do not run them for this project |
| `experiments/financial/` | Specs, generated specs, model panels, programmes and calibration records |
| `experiments/financial/outputs/` | Harness outputs: `runs/`, `comparisons/`, `programmes/`, `probes/` (default `HARNESS_HOME`, §0.8); commit only what §0.9 allows |
| `tests/` | Legacy tests (`concept_pipeline/`, `dataset_preparer/`) and harness tests (`harness/`) |
