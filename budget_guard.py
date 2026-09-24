"""
BudgetGuard for scan_harness — wired to YOUR calculate_cost_usd + pricing table.

Drop in at: src/scan_harness/adapter/budget_guard.py

Design notes specific to this codebase:
  * Reuses calculate_cost_usd() and self.pricing. No second rate table.
  * Raises your existing MaxTunsExceededError so callers need no change.
  * Enforced on BeforeModelCallEvent (the only hook that can stop a call:
    AfterModelCallEvent fires before metrics update and reads zero).
  * Settles on AfterInvocationEvent, because BeforeModelCall never observes
    the FINAL model call of a run — without this, a single-turn run meters $0.
  * Usage baseline is keyed per-Agent: _build_agent() makes a new Agent each
    call and Strands' accumulated_usage restarts at zero on each one.
"""

from __future__ import annotations

import logging
import threading
import weakref
from typing import Any

from strands.hooks import (AfterInvocationEvent, BeforeModelCallEvent,
                           HookProvider, HookRegistry)

LOGGER = logging.getLogger(__name__)

_USAGE_KEYS = ("inputTokens", "outputTokens", "totalTokens",
               "cacheReadInputTokens", "cacheWriteInputTokens")


class BudgetExceededError(RuntimeError):
    """Cumulative USD cap would be breached by the next model call."""


def as_guard_error(exc: BaseException, *types: type) -> BaseException | None:
    """Unwrap a guard error from Strands' EventLoopException.

    Strands wraps exceptions raised from a hook during tool-execution
    recursion. Raised on the first model call it propagates bare; raised
    after a tool has run it arrives wrapped. Always unwrap.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, types):
            return cur
        cur = cur.__cause__ or cur.__context__
    return None


class BudgetGuard(HookProvider):
    def __init__(self, model: str, cost_fn, pricing=None, turn_error_cls=RuntimeError):
        """
        model        — AIP ARN / model id, passed straight to cost_fn
        cost_fn      — your calculate_cost_usd
        pricing      — your self.pricing rate-table override
        turn_error_cls — your MaxTunsExceededError
        """
        self.model = model
        self._cost_fn = cost_fn
        self.pricing = pricing
        self._turn_error_cls = turn_error_cls

        self.max_usd: float | None = None
        self.max_turns: int | None = None
        self.reserve = True          # stop before the call that would breach

        self._lock = threading.Lock()
        self._seen: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()
        self._reset_counters()

    # -- lifecycle ----------------------------------------------------------
    def _reset_counters(self) -> None:
        self._spent_usd = 0.0
        self._peak_call_usd = 0.0
        self._turns = 0
        self._totals = {k: 0 for k in _USAGE_KEYS}

    def reset(self, max_usd: float | None = None, max_turns: int | None = None) -> None:
        """Start a fresh accounting window (call once per adapter invocation)."""
        with self._lock:
            self._reset_counters()
            self._seen = weakref.WeakKeyDictionary()
            self.max_usd = max_usd
            self.max_turns = max_turns

    # -- readouts -----------------------------------------------------------
    @property
    def turns(self) -> int:
        return self._turns

    @property
    def spent_usd(self) -> float:
        return self._spent_usd

    @property
    def totals(self) -> dict:
        """Token totals in Bedrock's key names, ready for your Usage()."""
        return dict(self._totals)

    def usage_kwargs(self) -> dict:
        """Maps straight onto your Usage dataclass fields."""
        t = self._totals
        return {
            "input_tokens": t["inputTokens"],
            "output_tokens": t["outputTokens"],
            "cache_read_tokens": t["cacheReadInputTokens"],
            "cache_write_tokens": t["cacheWriteInputTokens"],
            "total_cost_usd": round(self._spent_usd, 6),
        }

    # -- hooks --------------------------------------------------------------
    def register_hooks(self, registry: HookRegistry, **_: object) -> None:
        registry.add_callback(BeforeModelCallEvent, self._check)
        registry.add_callback(AfterInvocationEvent, self._settle)

    def _cost_of(self, delta: dict) -> float:
        return self._cost_fn(
            model=self.model,
            input_tokens=delta["inputTokens"],
            output_tokens=delta["outputTokens"],
            cache_read_tokens=delta["cacheReadInputTokens"],
            cache_write_tokens=delta["cacheWriteInputTokens"],
            pricing=self.pricing,
        )

    def _accrue(self, agent: Any) -> None:
        """Fold un-metered usage into the totals. Caller holds the lock."""
        current = agent.event_loop_metrics.accumulated_usage
        baseline = self._seen.get(agent) or {}
        delta = {k: max(0, int(current.get(k, 0)) - int(baseline.get(k, 0)))
                 for k in _USAGE_KEYS}
        if any(delta.values()):
            call_usd = self._cost_of(delta)
            self._spent_usd += call_usd
            self._peak_call_usd = max(self._peak_call_usd, call_usd)
            for k, v in delta.items():
                self._totals[k] += v
        self._seen[agent] = {k: int(current.get(k, 0)) for k in _USAGE_KEYS}

    def _settle(self, event: AfterInvocationEvent) -> None:
        with self._lock:
            self._accrue(event.agent)

    def _check(self, event: BeforeModelCallEvent) -> None:
        with self._lock:
            self._accrue(event.agent)

            if self.max_turns is not None and self._turns >= self.max_turns:
                LOGGER.warning("[BudgetGuard] max_turns %d reached (spent $%.4f)",
                               self.max_turns, self._spent_usd)
                raise self._turn_error_cls(
                    f"Max turns limit ({self.max_turns}) exceeded")

            self._turns += 1

            if self.max_usd is None:
                return
            projected = self._spent_usd + (self._peak_call_usd if self.reserve else 0.0)
            if projected > self.max_usd:
                LOGGER.warning("[BudgetGuard] budget stop: spent $%.4f projected $%.4f cap $%.2f",
                               self._spent_usd, projected, self.max_usd)
                raise BudgetExceededError(
                    f"Budget exceeded: spent ${self._spent_usd:.4f}, "
                    f"next call projected ${projected:.4f}, cap ${self.max_usd:.2f}")