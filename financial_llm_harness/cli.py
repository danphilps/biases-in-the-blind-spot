"""``financial-harness`` command line (spec §32, E-8).

argparse; ``load_dotenv()`` once; the single ``asyncio.run`` lives in the
execution entry points called from here; typed exceptions map to exit codes:
0 OK · 1 error · 2 validation failure · 3 budget stop · 4 incomplete run ·
5 policy refusal · 6 stale or missing probe. Only this module prints.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path

from financial_llm_harness.errors import HarnessError, NotImplementedCommand

Handler = Callable[[argparse.Namespace], tuple[int, str]]


def _resolve(dotted: str, name: str) -> Handler:
    """Import ``module:function`` lazily, so ``--help`` imports nothing heavy.

    A command function returns ``(exit_code, stdout_text)``; this module prints.
    """

    def handler(args: argparse.Namespace) -> tuple[int, str]:
        import importlib
        import importlib.util

        module_name, _, func = dotted.partition(":")
        if importlib.util.find_spec(module_name) is None:
            raise NotImplementedCommand(f"{name}: not implemented")
        module = importlib.import_module(module_name)
        target = getattr(module, func, None)
        if target is None:
            raise NotImplementedCommand(f"{name}: not implemented")
        code, text = target(args)
        return int(code), str(text)

    return handler


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="financial-harness",
        description="Behavioural test harness for undeclared biases of LLMs in finance answers.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, target: str, help_text: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_text)
        sp.set_defaults(handler=_resolve(target, name))
        return sp

    sp = add(
        "generate",
        "financial_llm_harness.commands:cmd_generate",
        "write the §0.10 bundle without the lock",
    )
    sp.add_argument("spec", type=Path)
    sp.add_argument("-o", "--output", type=Path, required=True)
    sp.add_argument("--panel")

    sp = add(
        "validate",
        "financial_llm_harness.commands:cmd_validate",
        "case, corpus, power, panel and provenance checks",
    )
    sp.add_argument("manifest", type=Path)

    sp = add(
        "freeze", "financial_llm_harness.commands:cmd_freeze", "write manifest.lock"
    )
    sp.add_argument("manifest", type=Path)

    sp = add(
        "plan",
        "financial_llm_harness.commands:cmd_plan",
        "counts, tokens, cost and power",
    )
    sp.add_argument("manifest", type=Path)

    sp = add(
        "run",
        "financial_llm_harness.commands:cmd_run",
        "run a spec or manifest (§31A L1–L13)",
    )
    sp.add_argument("file", type=Path, nargs="?")
    sp.add_argument("--panel")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--yes", action="store_true")
    sp.add_argument("--resume", metavar="RUN_ID")
    sp.add_argument("--max-cost-usd", type=float)
    sp.add_argument("--max-calls", type=int)

    sp = add(
        "archive",
        "financial_llm_harness.commands:cmd_archive",
        "write run_archive.tar.gz and ARCHIVE.sha256",
    )
    sp.add_argument("run_id")

    sp = add(
        "status",
        "financial_llm_harness.commands:cmd_status",
        "run status, trial counts and spend",
    )
    sp.add_argument("run_id")

    sp = add(
        "evaluate", "financial_llm_harness.commands:cmd_evaluate", "run §31A L8–L10"
    )
    sp.add_argument("run_id")

    sp = add(
        "analyse", "financial_llm_harness.commands:cmd_analyse", "run §31A L9 and L11"
    )
    sp.add_argument("run_id")

    for name in ("render", "report"):
        sp = add(
            name,
            "financial_llm_harness.commands:cmd_render",
            "regenerate every output file from the DB",
        )
        sp.add_argument("run_id")

    sp = add(
        "replay",
        "financial_llm_harness.commands:cmd_replay",
        "re-evaluate or re-execute a run",
    )
    sp.add_argument("run_id")
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("--reevaluate", action="store_true")
    g.add_argument("--reexecute", action="store_true")
    sp.add_argument("--re-judge", action="store_true")
    sp.add_argument("--yes", action="store_true")

    sp = add(
        "compare",
        "financial_llm_harness.commands:cmd_compare",
        "cross-run label matrix",
    )
    sp.add_argument("run_ids", nargs="+")
    sp.add_argument("--name", required=True)

    sp = add(
        "models",
        "financial_llm_harness.commands:cmd_models",
        "list model IDs from a provider",
    )
    sp.add_argument("action", choices=["list"])
    sp.add_argument("--provider", required=True)
    sp.add_argument("--region")

    sp = add(
        "probe",
        "financial_llm_harness.commands:cmd_probe",
        "capability probe (§20.4); without MODEL_KEY, every panel model with an API key",
    )
    sp.add_argument("model_key", nargs="?")
    sp.add_argument("--panel", required=True)

    sp = add(
        "calibrate",
        "financial_llm_harness.commands:cmd_calibrate",
        "evaluator calibration record (§16)",
    )
    sp.add_argument("--feature", required=True)
    sp.add_argument("--labels", type=Path, required=True)
    sp.add_argument("--panel", required=True)

    sp = add(
        "sandbox",
        "financial_llm_harness.commands:cmd_sandbox",
        "build the sandbox images (§13.4)",
    )
    sp.add_argument("action", choices=["build"])

    sp = add(
        "programme", "financial_llm_harness.commands:cmd_programme", "programmes (§32B)"
    )
    sp.add_argument("action", choices=["plan", "run", "status", "report"])
    sp.add_argument("programme", type=Path, nargs="?")
    sp.add_argument("--resume", metavar="PROGRAMME_RUN_ID")
    sp.add_argument("--yes", action="store_true")

    sp = add(
        "pilot",
        "financial_llm_harness.commands:cmd_pilot",
        "size a confirmatory spec from a small discovery pilot (§22, §23.6)",
    )
    sp.add_argument("action", choices=["make", "plan"])
    sp.add_argument("spec", type=Path, help="the confirmatory spec")
    sp.add_argument("-o", "--output", type=Path, required=True)
    sp.add_argument("--n-cases", type=int, default=100, help="make: pilot cases")
    sp.add_argument("--run", metavar="PILOT_RUN_ID", help="plan: the pilot run")
    sp.add_argument(
        "--no-inflate",
        action="store_true",
        help="plan: use sd_c² itself, not its 80%% upper confidence limit",
    )

    sp = add(
        "discovery",
        "financial_llm_harness.commands:cmd_discovery",
        "legacy discovery bridge (§27.1)",
    )
    sp.add_argument("action", choices=["import-legacy"])
    sp.add_argument("--results-dir", type=Path, required=True)
    sp.add_argument("--experiment-key", required=True)
    sp.add_argument("--dataset", required=True)
    sp.add_argument("-o", "--output", type=Path, required=True)

    sp = add(
        "taxonomy", "financial_llm_harness.commands:cmd_taxonomy", "Part II registry"
    )
    sp.add_argument(
        "action",
        choices=["validate", "coverage", "list", "expand", "survey-export", "weights"],
    )
    sp.add_argument("--domain", choices=["F", "Q", "M", "K"])
    sp.add_argument("--priority")
    sp.add_argument("--item")
    sp.add_argument("--wave")
    sp.add_argument("--allow-partial", action="store_true")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--role")
    sp.add_argument("--responses", type=Path)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code (§32)."""
    import dotenv

    dotenv.load_dotenv()
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        code, text = args.handler(args)
        if text:
            sys.stdout.write(text if text.endswith("\n") else text + "\n")
    except HarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = exc.exit_code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        code = 1
    except Exception:  # noqa: BLE001 - map every unhandled error to exit 1
        traceback.print_exc()
        code = 1
    return code


def entry() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entry()
