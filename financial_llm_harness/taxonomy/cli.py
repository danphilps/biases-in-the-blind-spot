"""``financial-harness taxonomy …`` (spec §32, §T1, §T8, §T10, §32B.2)."""

from __future__ import annotations

import argparse

from financial_llm_harness.errors import NotImplementedCommand

DOMAIN_CODES = {"F": "fundamental", "Q": "quant_models", "M": "ml", "K": "code"}


def _require(module: str, action: str) -> None:
    import importlib.util

    if importlib.util.find_spec(module) is None:
        raise NotImplementedCommand(f"taxonomy {action}: not implemented")


def dispatch(args: argparse.Namespace) -> tuple[int, str]:
    action = args.action
    if action in ("expand", "coverage", "survey-export", "weights"):
        module = {"survey-export": "survey"}.get(action, action)
        _require(f"financial_llm_harness.taxonomy.{module}", action)
    if action == "validate":
        from financial_llm_harness.taxonomy.registry import validate_registry

        reg = validate_registry()
        waves: dict[str, int] = {}
        for item in reg.items.values():
            waves[item.wave] = waves.get(item.wave, 0) + 1
        counts = ", ".join(f"{w} {n}" for w, n in sorted(waves.items()))
        return 0, f"taxonomy validate: {len(reg.items)} items OK ({counts})"
    if action == "list":
        return 0, list_items(args)
    if action == "expand":
        from financial_llm_harness.taxonomy.expand import run_expand

        code = run_expand(
            args.item, args.wave, allow_partial=args.allow_partial, force=args.force
        )
        return code, ""
    if action == "coverage":
        from financial_llm_harness.taxonomy.coverage import write_coverage

        path = write_coverage()
        return 0, f"wrote {path}"
    if action == "survey-export":
        from financial_llm_harness.taxonomy.survey import export_survey

        paths = export_survey(args.role)
        return 0, "\n".join(f"wrote {p}" for p in paths)
    if action == "weights":
        from financial_llm_harness.taxonomy.weights import compute_weights

        if args.responses is None:
            from financial_llm_harness.errors import HarnessValidationError

            raise HarnessValidationError(
                "taxonomy weights needs --responses PATH (§T10.3)"
            )
        path = compute_weights(args.responses)
        return 0, f"wrote {path}"
    raise NotImplementedCommand(f"taxonomy {action}")


def list_items(args: argparse.Namespace) -> str:
    from financial_llm_harness.taxonomy.registry import load_registry

    reg = load_registry()
    rows = []
    for item in reg.items.values():
        d = item.data
        if args.domain and d.get("domain") != DOMAIN_CODES[args.domain]:
            continue
        if args.priority and d.get("priority") != args.priority:
            continue
        if args.wave and item.wave != args.wave:
            continue
        rows.append(
            f"{item.id}\t{item.wave}\t{d.get('priority')}\t{d.get('interaction')}\t"
            f"{d.get('status')}\t{d.get('question')}"
        )
    return "\n".join(["id\twave\tpriority\tinteraction\tstatus\tquestion", *rows])
