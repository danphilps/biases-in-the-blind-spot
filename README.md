# Undeclared biases in LLM finance answers

This repository tests large language models for **undeclared biases** when they answer finance and quantitative-finance questions: fundamental equity judgement, valuation, earnings analysis, corporate actions, broker research, quant model design, machine learning for stock selection and quant code.

It is a fork of the code for [Biases in the Blind Spot: Detecting What LLMs Fail to Mention](https://arxiv.org/abs/2602.10117) (Arcuschin et al., 2026). The original concept pipeline is kept and still runs ([Original paper pipeline](#original-paper-pipeline)). The new work is the **financial harness**, `financial_llm_harness/`, built from `docs/financial_harness_spec.md`.

## What is tested

An undeclared bias is present when all three hold:

1. **The answer depends on a factor.** Changing one input, and nothing else, moves the answer.
2. **The dependence is wrong.** By an established standard (the mandate, an accounting identity, a planted ground truth, a cue irrelevant by construction, or a published result), the factor should not move the answer, or should move it the other way or by less.
3. **The model does not say so.** Its stated reasons do not mention the factor.

When all three hold, the harness labels the result `UNVERBALISED_BIAS`. Each experiment answers three questions per model:
- Does the answer move when it should?
- Does it stay put when it should not?
- Does the model say what moved it?

The design rules are:
- Arms are compared within matched (case, paraphrase, repetition) blocks, with cluster-level inference.
- Every confirmatory experiment carries a null, a placebo and a positive control.
- Multiplicity is corrected per model.
- "Bias" is reported only in confirmatory mode against an established oracle; sensitivity alone is not bias.
- Every experiment is frozen before any model call: cases, rendered prompts, asset hashes and code identity. Re-evaluating stored responses is byte-identical.

`AGENTS.md` states the rules that protect the test. The spec (§1, §2, §38, §40) is the build instruction.

## Status

| Milestone (spec §38) | Content | State |
|---|---|---|
| M0–M2 | Core primitives, DB, archive, CLI; S1 (stock-add judgement) on the fake path; resume, replay, full reporting | Done |
| M3 | Provider adapters (OpenAI-compatible, Anthropic, Gemini), request profiles, cassettes, probe, budget, compare | Done |
| M10 | Programmes with per-model false-discovery-rate control; `question_answers.csv` | Done |
| M4–M5 | S3 covariance choice, S4 temporal-normalisation design, S5 temporal-normalisation code (Docker sandbox) | Done |
| M6–M7 | Semantic evaluator and calibration, S2 transcript summary; discovery → confirmation workflow | Done |
| M8 | Part II question registry (118 items), `taxonomy expand`, the undeclared-bias battery | Done |
| M9a | V1.1 infrastructure: remaining families, code-test checks, scorers, expert-panel and survey tooling, deny-list screening | Done |
| M9b | V1.1 scientific content | Blocked on user-supplied data (D-23); never authored by an agent |

The whole pipeline runs on the **fake panel** (`experiments/financial/panels/ci_fake.yaml`): two deterministic fake models with known built-in effects, used to show that the harness recovers them. **No result in this repository is a finding about a real model.**

### What a live run still needs

`validate` refuses a live (non-fake) run until each of these is in place. The D-numbers refer to the user-owned decisions in spec §39.1.
- **Human-written text (D-14, §2.7).** Every template, paraphrase, name bank, mandate and oracle file is an agent-drafted placeholder. The 110 files to replace and approve are listed in `experiments/financial/AUTHORING_INVENTORY.md`.
- **Deny-list (§7).** `experiments/financial/deny_list.yaml`, filled from a licensed or official list of listed-company names and tickers. The format is in `deny_list.example.yaml`. Every rendered company, CEO and broker name is screened against it.
- **Evaluator (D-2).** Semantic features need an evaluator model from a family outside the panel, with its own key and probe. `panels/live.yaml` has none yet.
- **Calibration labellers (D-3)** for confirmatory semantic features, and **named expert-panel members (D-13)** for checklist sign-off.
- **Docker (D-6)** for the code tasks: S5 and the K.M.1 battery.
- **Storage.** The V1 fake programme wrote about 20 GB for two models. Size the archive space before a live panel.

## Quick start (fake path)

```bash
uv sync --all-extras

# One experiment: generate → validate → freeze → run → evaluate → analyse → report
uv run financial-harness run experiments/financial/specs/stock_attractive_momentum_m0.yaml --panel ci_fake --yes

# The S1 explicit-use (m1) and explicit-ignore (m2) controls as one programme
uv run financial-harness programme plan experiments/financial/programmes/ci_fake.yaml
uv run financial-harness programme run  experiments/financial/programmes/ci_fake.yaml --yes

# Every V1 experiment (slice specs plus the generated battery); needs Docker and ~20 GB
uv run financial-harness programme run  experiments/financial/programmes/v1_fake.yaml --yes
```

The Part II question registry has its own commands:

```bash
uv run financial-harness taxonomy validate
uv run financial-harness taxonomy list --priority P1
uv run financial-harness taxonomy expand --wave V1 --allow-partial   # writes experiments/financial/generated/; lists the D-25 identity-cue pairs
uv run financial-harness taxonomy expand --wave V1.1 --allow-partial # lists the data each V1.1 pair still lacks
uv run financial-harness taxonomy coverage                           # rewrites financial_llm_harness/taxonomy/COVERAGE.md
```

`financial-harness --help` lists every command: `generate`, `validate`, `freeze`, `plan`, `run`, `status`, `evaluate`, `analyse`, `report`, `replay`, `archive`, `compare`, `models`, `probe`, `calibrate`, `sandbox`, `programme`, `discovery` and `taxonomy`.

## Outputs

Outputs go to `experiments/financial/outputs/` (override with `HARNESS_HOME`): `runs/<run>/` and `programmes/<programme>/`. They are git-ignored.

Per run, `results/labels.csv` answers the three questions per model:
- `sensitivity_label` and `is_bias`: does the answer move, and is that a bias under the oracle?
- `attribution_label`: does the model say so?
- `composite_label`: `UNVERBALISED_BIAS` when all three conditions hold.

The other per-run files:
- `effects.csv`: θ̂ with its confidence interval and p-values.
- `attribution.csv`: verbalisation rates per channel.
- `controls.csv`: the null, placebo and positive-control gates.
- `report/report.md`: the narrative report.

A programme adds `results/answers.csv`, `question_answers.csv` (one verdict per question per model) and the per-model Benjamini–Hochberg table `fdr.csv`.

## Live runs

The live panel `experiments/financial/panels/live.yaml` holds Anthropic `claude-opus-5-5` and OpenAI `gpt-6-luna`. Each model is estimated, labelled and multiplicity-corrected on its own.

Only providers whose key is set in the environment or `.env` are called (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). A panel model without a key is skipped at `generate`, recorded in the manifest as `sampling_spec.models_skipped` and listed by `plan`. A frozen manifest whose model has lost its key is refused (exit 2), never silently narrowed.

```bash
uv run financial-harness models list --provider anthropic        # confirm the model IDs
uv run financial-harness models list --provider openai
uv run financial-harness probe --panel live                      # ≤ 20 calls per keyed model; needed within 30 days of a run
uv run financial-harness programme plan experiments/financial/programmes/live.yaml
uv run financial-harness programme run  experiments/financial/programmes/live.yaml --yes
# or one experiment:
uv run financial-harness run experiments/financial/specs/covariance_choice.yaml --panel live --yes
```

Before a live run is accepted, clear the list in [What a live run still needs](#what-a-live-run-still-needs), and also:
- **Placeholder family (§2.7).** The placeholders are Anthropic-drafted, and §2.7 bars text drafted by any target's own family.
- **Evaluator.** Without an evaluator, the S1 specs' semantic reasoning channel is `SKIPPED_INVALID`; the structured `factors` channel needs none.
- **Prices and budget.** The `pricing.yaml` rates for both models come from secondary sources, so confirm them. The programme caps (D-2) cover the plan's upper bound of about $3,780 for the five specs in `programmes/live.yaml`.

## Repository layout

| Path | What it is |
|---|---|
| `docs/financial_harness_spec.md` | The specification. Part I covers how experiments run; Part II covers what is assessed (the question registry). |
| `financial_llm_harness/` | The harness: `core/`, `generation/`, `execution/`, `evaluation/`, `statistics/`, `reporting/`, `models/` (provider adapters), `families/` (task families and their placeholder content) and `taxonomy/` (Part II registry, interventions, bases, battery). |
| `experiments/financial/specs/` | The six canonical slice specs (S1–S5, with S1 in two variants). |
| `experiments/financial/generated/` | Specs written by `taxonomy expand`. |
| `experiments/financial/panels/`, `programmes/` | Model panels (`ci_fake`, `live`) and programmes (`ci_fake`, `v1_fake`, `v1_1_fake`, `live`). |
| `experiments/financial/calibration/` | Calibration records for the semantic evaluator. |
| `experiments/financial/taxonomy_build/` | Scripts that build the registry, defaults, checklists and authoring inventory. |
| `tests/harness/` | Harness tests (provider calls use cassettes; live tests need `HARNESS_LIVE=1`). |
| `biases_in_the_blind_spot/`, `experiments/{loan_approval,university_admission,llm_bias}/` | The original paper's pipeline. |

## Original paper pipeline

The original pipeline lives in `biases_in_the_blind_spot/` (the concept pipeline) and `experiments/` (the paper's experiments). The repository contains the results JSON for each experiment in the `results` directory.

Each experiment runs in two steps:
1. Run `prepare_dataset.py` to set up the initial dataset and generate concepts.
2. Run `concept_pipeline.py` to run the pipeline. Set `model_name` in the script to the model you want to use.

Results are saved as JSON in the `results` directory. Put API keys for Anthropic, OpenAI and OpenRouter in a `.env` file at the repository root, or set them as environment variables.

### Synthetic datasets

The generators for the synthetic loan-approval and university-admission datasets are in `biases_in_the_blind_spot/datasets`. They need `OPENAI_API_KEY`.

```bash
# Loan approval (default 10,000 samples)
uv run python -m biases_in_the_blind_spot.datasets.loan_approval.generate_loan_approval_dataset -n 5000

# University admission (default 1,000 samples)
uv run python -m biases_in_the_blind_spot.datasets.university_admission.generate_university_admission_dataset -n 2000
```

Common options: `-n` (number of samples), `-o` (output file), `-m` (model name), `--seed` (random seed).

## Development

Python 3.12, managed with [uv](https://docs.astral.sh/uv/). This is the CI command; every change must pass it:

```bash
uv sync --all-extras && uv run ruff format --check && uv run ruff check && uv run pyright \
  && uv run pytest && uv build && uv run financial-harness --help
```

## Citation

The original paper:

```bibtex
@misc{arcuschin2026biasesblindspotdetecting,
      title={Biases in the Blind Spot: Detecting What LLMs Fail to Mention}, 
      author={Iván Arcuschin and David Chanin and Adrià Garriga-Alonso and Oana-Maria Camburu},
      year={2026},
      eprint={2602.10117},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.10117}, 
}
```
