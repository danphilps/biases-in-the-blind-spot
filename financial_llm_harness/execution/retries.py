"""Retry policy (spec §15).

Transport failures (connection error, HTTP 5xx or 408, timeout before a complete
response) retry the identical request with the same seed, backoff 2^k s, at most
5 tries. HTTP 429 honours ``retry-after`` (default 60 s) under a separate bound:
20 consecutive 429s or 30 minutes of cumulative waiting end the trial RATE_LIMIT.
Any other 4xx is API_ERROR. Model behaviour is never retried.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_TRANSPORT_TRIES = 5
MAX_CONSECUTIVE_429 = 20
MAX_429_WAIT_S = 30 * 60
DEFAULT_RETRY_AFTER_S = 60.0
MAX_BATCH_RESUBMISSIONS = 2
BATCH_CANCEL_AFTER_S = 25 * 3600


@dataclass
class RetryState:
    transport_tries: int = 0
    consecutive_429: int = 0
    waited_429_s: float = 0.0


def classify(transport_status: str, http_status: int | None) -> str:
    """complete | retry_transport | retry_429 | api_error."""
    if transport_status == "complete":
        return "complete"
    if transport_status in ("connection_error", "timeout"):
        return "retry_transport"
    if transport_status == "http_error":
        if http_status == 429:
            return "retry_429"
        if http_status is not None and (http_status >= 500 or http_status == 408):
            return "retry_transport"
        return "api_error"
    return "api_error"


def next_action(
    state: RetryState, kind: str, retry_after_s: float | None
) -> tuple[str, float]:
    """(action, wait seconds); action ∈ {retry, RATE_LIMIT, API_ERROR, TIMEOUT}."""
    if kind == "retry_429":
        wait = (
            DEFAULT_RETRY_AFTER_S
            if retry_after_s is None
            else max(0.0, float(retry_after_s))
        )
        state.consecutive_429 += 1
        if (
            state.consecutive_429 >= MAX_CONSECUTIVE_429
            or state.waited_429_s + wait > MAX_429_WAIT_S
        ):
            return "RATE_LIMIT", 0.0
        state.waited_429_s += wait
        return "retry", wait
    state.consecutive_429 = 0
    if kind == "retry_transport":
        state.transport_tries += 1
        if state.transport_tries >= MAX_TRANSPORT_TRIES:
            return "TRANSPORT_EXHAUSTED", 0.0
        return "retry", float(2**state.transport_tries)
    return "API_ERROR", 0.0
