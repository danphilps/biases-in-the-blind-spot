# Biases in the Blind Spot: Detecting What LLMs Fail to Mention

Code for the paper [Biases in the Blind Spot: Detecting What LLMs Fail to Mention](https://arxiv.org/abs/2602.10117).

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Installation

1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
2. Install dependencies:
   ```bash
   uv sync
   ```

## Code Structure

The code is organized into the following directories:

- `biases_in_the_blind_spot`: The main code for the concept pipeline.
- `experiments`: The experiments for the paper.
- `tests`: The tests for the code.

### Running experiments

This repo already contains our results json for each experiment in the `results` directory.

If you would like to run the experiments yourself, each experiment requires 2 steps to run:

- first, run the `prepare_dataset.py` script to set up initial dataset / generate concepts
- then, run the `concept_pipeline.py` script to run the pipeline
  - NOTE: You need to set the `model_name` in the `concept_pipeline.py` script to the model you want to use

Results are saved in the `results` directory as JSON files. You should also create a `.env` file in your root directory with API keys for Anthropic, OpenAI, and OpenRouter, or you can set environment variables for the same.

### Synthetic datasets

The code to generate our synthetic loan approval and university admission datasets is in the `biases_in_the_blind_spot/datasets` directory. If you would like to extend these datasets, you can rerun the generation scripts with a larger number of generated samples.

**Requirements:** Set your OpenAI API key via `OPENAI_API_KEY` environment variable or in a `.env` file.

**Loan Approval Dataset:**

```bash
# Generate with defaults (10,000 samples)
uv run python -m biases_in_the_blind_spot.datasets.loan_approval.generate_loan_approval_dataset

# Generate a custom number of samples
uv run python -m biases_in_the_blind_spot.datasets.loan_approval.generate_loan_approval_dataset -n 5000

# See all options
uv run python -m biases_in_the_blind_spot.datasets.loan_approval.generate_loan_approval_dataset --help
```

**University Admission Dataset:**

```bash
# Generate with defaults (1,000 samples)
uv run python -m biases_in_the_blind_spot.datasets.university_admission.generate_university_admission_dataset

# Generate a custom number of samples
uv run python -m biases_in_the_blind_spot.datasets.university_admission.generate_university_admission_dataset -n 2000
```

**Common options:** `-n` (number of samples), `-o` (output file), `-m` (model name), `--seed` (random seed).

## Financial harness: live runs

`financial_llm_harness/` (built from `docs/financial_harness_spec.md`) tests finance answers for undeclared biases. The live panel `experiments/financial/panels/live.yaml` holds Anthropic `claude-opus-5-5` and OpenAI `gpt-6-luna`; each model is estimated, labelled and multiplicity-corrected on its own.

Only providers whose key is set in the environment or `.env` are called (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). A panel model without a key is skipped at `generate`, recorded in the manifest as `sampling_spec.models_skipped`, and listed by `plan`; a frozen manifest whose model has lost its key is refused (exit 2), never silently narrowed.

```bash
uv run financial-harness models list --provider anthropic        # confirm the model IDs
uv run financial-harness models list --provider openai
uv run financial-harness probe --panel live                      # ≤ 20 calls per keyed model; needed within 30 days of a run
uv run financial-harness programme plan experiments/financial/programmes/live.yaml
uv run financial-harness programme run  experiments/financial/programmes/live.yaml --yes
# or one experiment:
uv run financial-harness run experiments/financial/specs/covariance_choice.yaml --panel live --yes
```

Outputs go to `experiments/financial/outputs/` (`runs/<run>/`, `programmes/<programme>/`). Per run, `results/labels.csv` answers the three questions per model: `sensitivity_label` and `is_bias` (does the answer move, and is that a bias under the oracle), `attribution_label` (does the model say so) and `composite_label` (`UNVERBALISED_BIAS` when all three hold). `effects.csv` holds θ̂ with its CI and p-values, `attribution.csv` the verbalisation rates per channel, `controls.csv` the null, placebo and positive-control gates, and `report/report.md` the narrative. A programme adds `results/answers.csv`, `question_answers.csv` and the per-model BH table `fdr.csv`.

**Economy route (pilot → confirm).** Without a pilot, each binary contrast is sized at the variance bound σ² = 1.0, so every experiment runs about 16,000 calls per model, and the controls account for 64% of them. A 100-case discovery pilot measures the real variances; `pilot plan` then sizes the confirmatory run to the same δ, margins, α and power:

```bash
uv run financial-harness pilot make experiments/financial/specs/momentum_stock_add.yaml -o pilot_s1.yaml
uv run financial-harness run pilot_s1.yaml --panel live --yes             # 1,800 calls per model
uv run financial-harness pilot plan experiments/financial/specs/momentum_stock_add.yaml --run <PILOT_RUN_ID> -o s1_planned.yaml
uv run financial-harness run s1_planned.yaml --panel live --yes
```

The live panel sends trials through the batch APIs (half price, results within 24 h).

Before a live run is accepted:
- **Human-authored text (D-14, §2.7).** The templates, paraphrases, banks, mandates and oracle files are agent-drafted placeholders. `validate` refuses any live manifest that uses an agent-drafted or unapproved file until people replace them and approve their provenance blocks; §2.7 also bars text drafted by any target's own family (the placeholders are Anthropic-drafted).
- **Evaluator.** The S1 specs' semantic reasoning channel needs an evaluator from a family outside the panel (neither Anthropic nor OpenAI), with its own key and probe. Without it those specs are `SKIPPED_INVALID`; the structured `factors` channel needs no evaluator.
- **Prices and budget.** The `pricing.yaml` rates for both models come from secondary sources; confirm them. The programme caps (D-2) cover the plan's upper bound of about $3,780 for all five specs.
- **Sandbox.** The code task (`temporal_norm_code`) needs Docker (`financial-harness sandbox build`).

## Development

### Code Quality

This project uses:

- **Ruff** for linting and code formatting
- **Pyright** for type checking
- **Pytest** for testing

```bash
# Lint code
uv run ruff check .

# Format code
uv run ruff format .

# Type check
uv run pyright .

# Testing
uv run pytest
```

## Citation

If you use this in your work, please cite:

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
