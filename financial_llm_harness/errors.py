"""Typed exceptions mapped to CLI exit codes (spec §0.7, §32, E-11)."""

from __future__ import annotations


class HarnessError(Exception):
    """Generic harness failure (exit 1)."""

    exit_code: int = 1


class HarnessValidationError(HarnessError):
    """Input, manifest or bundle validation failure (exit 2)."""

    exit_code = 2


class FrozenInputDrift(HarnessValidationError):
    """A frozen input or executable source no longer matches its hash (§4; exit 2)."""

    def __init__(self, mismatches: list[str]) -> None:
        self.mismatches = list(mismatches)
        super().__init__(
            "FROZEN_INPUT_DRIFT: " + "; ".join(self.mismatches)
            if self.mismatches
            else "FROZEN_INPUT_DRIFT"
        )


class BudgetExhausted(HarnessError):
    """A budget cap would be breached (§30; exit 3)."""

    exit_code = 3


class IncompleteRun(HarnessError):
    """A run ended INCOMPLETE (§29; exit 4)."""

    exit_code = 4


class PolicyRefusal(HarnessError):
    """The data policy forbids sending this manifest to a provider (§2.14; exit 5)."""

    exit_code = 5


class StaleProbe(HarnessError):
    """A required probe file is missing or stale (§20.4; exit 6)."""

    exit_code = 6


class LockHeld(HarnessError):
    """Another process holds the run's writer lock (§32; exit 1)."""

    exit_code = 1


class NotImplementedCommand(HarnessError):
    """A CLI command whose milestone is not built (exit 1)."""

    exit_code = 1
