"""
Turn and budget guards for scan_harness.
Drop at: src/scan_harness/adapter/usage_guard.py

Two independent guards you attach to a Strands Agent as hooks:

    TurnGuard    caps the number of model calls
    BudgetGuard  caps USD spend, using your calculate_cost_usd()

Both are armed by calling reset() at the start of each adapter invocation.


WHY THESE HOOKS AND NOT OTHERS
------------------------------
BeforeModelCallEvent is the only place a call can be stopped before you pay
for it. AfterModelCallEvent looks like the obvious choice but fires BEFORE
Strands updates its metrics, so it reads zero on the first call and lags by
one call forever.

BudgetGuard also listens to AfterInvocationEvent. BeforeModelCall never sees
the LAST model call of a run (nothing fires after it), so without a final
settlement a single-turn run would report $0.00.
"""

import logging

from strands.hooks import (AfterInvocationEvent, BeforeModelCallEvent,
                           HookProvider, HookRegistry)

LOGGER = logging.getLogger(__name__)

# The five token counters Bedrock reports.
USAGE_KEYS = (
    "inputTokens",
    "outputTokens",
    "totalTokens",
    "cacheReadInputTokens",
    "cacheWriteInputTokens",
)


class TurnLimitExceeded(RuntimeError):
    """Used only if you don't pass your own error class to TurnGuard."""


class BudgetExceededError(RuntimeError):
    """The USD cap would be broken by the next model call."""


def as_guard_error(exc, *types):
    """Find a guard error inside exc, or return None.

    Strands wraps exceptions raised from a hook in EventLoopException once a
    tool has run, so `except MaxTunsExceededError` misses most real trips.
    Always unwrap:

        except Exception as e:
            hit = as_guard_error(e, MaxTunsExceededError, BudgetExceededError)
            if hit is not None:
                raise hit from e
            raise
    """
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, types):
            return current
        current = current.__cause__ or current.__context__
    return None


class TurnGuard(HookProvider):
    """Stops the agent after max_turns model calls.

    A turn is one model call plus any tool execution that follows it.

        guard = TurnGuard(error_cls=MaxTunsExceededError)
        guard.reset(max_turns=40)
        agent = Agent(..., hooks=[guard])
    """

    def __init__(self, error_cls=TurnLimitExceeded):
        self.error_cls = error_cls
        self.max_turns = None
        self.turns = 0

    def reset(self, max_turns=None):
        """Arm the guard for a new invocation."""
        self.max_turns = max_turns
        self.turns = 0

    @property
    def remaining(self):
        if self.max_turns is None:
            return None
        return max(0, self.max_turns - self.turns)

    @property
    def breakdown(self):
        return {
            "turns": self.turns,
            "max_turns": self.max_turns,
            "remaining": self.remaining,
        }

    def register_hooks(self, registry: HookRegistry, **kwargs):
        registry.add_callback(BeforeModelCallEvent, self._check)

    def _check(self, event):
        # Checked before the increment, so exactly max_turns calls run.
        if self.max_turns is not None and self.turns >= self.max_turns:
            LOGGER.warning("[TurnGuard] max_turns %s reached", self.max_turns)
            raise self.error_cls(f"Max turns limit ({self.max_turns}) exceeded")
        self.turns += 1


class BudgetGuard(HookProvider):
    """Stops the agent before USD spend passes max_usd.

        guard = BudgetGuard(model=self.model,
                            cost_fn=calculate_cost_usd,
                            pricing=self.pricing)
        guard.reset(max_usd=5.00)
        agent = Agent(..., hooks=[guard])
        ...
        usage = Usage(**guard.usage_kwargs())

    Note on Bedrock billing: inputTokens EXCLUDES the cache tokens, which are
    reported and priced separately. All four counts are passed to cost_fn, so
    keep them separate rather than adding cache tokens into inputTokens.
    """

    def __init__(self, model, cost_fn, pricing=None, reserve=True):
        self.model = model
        self.cost_fn = cost_fn
        self.pricing = pricing
        # reserve=True stops when the NEXT call is projected to break the cap,
        # so spend never goes over. reserve=False lets the run cross the line
        # first and stops after.
        self.reserve = reserve

        self.max_usd = None
        self.spent_usd = 0.0
        self.model_calls = 0
        self.totals = {key: 0 for key in USAGE_KEYS}
        self._last_call_usd = 0.0
        # Token counts already accounted for. Strands reports usage as a
        # running total per Agent, so we subtract this to get each new call.
        self._counted = {key: 0 for key in USAGE_KEYS}

    def reset(self, max_usd=None):
        """Arm the guard for a new invocation."""
        self.max_usd = max_usd
        self.spent_usd = 0.0
        self.model_calls = 0
        self.totals = {key: 0 for key in USAGE_KEYS}
        self._last_call_usd = 0.0
        self._counted = {key: 0 for key in USAGE_KEYS}

    @property
    def remaining_usd(self):
        if self.max_usd is None:
            return None
        return max(0.0, self.max_usd - self.spent_usd)

    def usage_kwargs(self):
        """Maps onto the fields of your Usage dataclass."""
        return {
            "input_tokens": self.totals["inputTokens"],
            "output_tokens": self.totals["outputTokens"],
            "cache_read_tokens": self.totals["cacheReadInputTokens"],
            "cache_write_tokens": self.totals["cacheWriteInputTokens"],
            "total_cost_usd": round(self.spent_usd, 6),
        }

    @property
    def breakdown(self):
        result = {
            "model": self.model,
            "model_calls": self.model_calls,
            "spent_usd": round(self.spent_usd, 6),
            "max_usd": self.max_usd,
            "remaining_usd": self.remaining_usd,
        }
        result.update(self.totals)
        return result

    def register_hooks(self, registry: HookRegistry, **kwargs):
        registry.add_callback(BeforeModelCallEvent, self._check)
        registry.add_callback(AfterInvocationEvent, self._settle)

    def _accrue(self, agent):
        """Add whatever usage we have not counted yet."""
        current = agent.event_loop_metrics.accumulated_usage

        # A brand new Agent starts its counters at zero. If the numbers went
        # backwards, _build_agent() made a fresh one, so count from zero.
        if current.get("totalTokens", 0) < self._counted["totalTokens"]:
            self._counted = {key: 0 for key in USAGE_KEYS}

        new = {}
        for key in USAGE_KEYS:
            new[key] = max(0, int(current.get(key, 0)) - self._counted[key])

        if any(new.values()):
            self._last_call_usd = self.cost_fn(
                model=self.model,
                input_tokens=new["inputTokens"],
                output_tokens=new["outputTokens"],
                cache_read_tokens=new["cacheReadInputTokens"],
                cache_write_tokens=new["cacheWriteInputTokens"],
                pricing=self.pricing,
            )
            self.spent_usd += self._last_call_usd
            for key in USAGE_KEYS:
                self.totals[key] += new[key]

        for key in USAGE_KEYS:
            self._counted[key] = int(current.get(key, 0))

    def _settle(self, event):
        """Final accounting once the run is over. Never raises."""
        self._accrue(event.agent)

    def _check(self, event):
        self._accrue(event.agent)
        self.model_calls += 1

        if self.max_usd is None:
            return

        projected = self.spent_usd
        if self.reserve:
            # The next call costs about what the last one did, and usually a
            # little more, since each call resends the whole history.
            projected += self._last_call_usd

        if projected > self.max_usd:
            LOGGER.warning(
                "[BudgetGuard] stopping: spent $%.4f, next call about $%.4f, cap $%.2f",
                self.spent_usd, projected, self.max_usd,
            )
            raise BudgetExceededError(
                f"Budget exceeded: spent ${self.spent_usd:.4f}, "
                f"next call projected ${projected:.4f}, cap ${self.max_usd:.2f}"
            )