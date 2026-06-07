"""Phase-batching scheduler — the rule that encodes VRAM-swap minimization.

These are pure-function tests (no GPU, no DB): the live worker and the
`/api/jobs/plan` preview share `select_in_tier`/`plan_queue`, so the predicted
swap count can never drift from what actually runs. The load-bearing invariant
is *one swap for a mixed batch*.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.core.enums import JobType
from app.core.scheduler import Worker, plan_queue, select_in_tier

_BASE = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass
class FakeJob:
    model_id: str
    type: object  # JobType or plain string (SQLite hands back strings)
    priority: int = 0
    seq: int = 0

    @property
    def created_at(self) -> datetime:
        return _BASE + timedelta(seconds=self.seq)


def _job(model_id, type_, priority=0, seq=0):
    return FakeJob(model_id=model_id, type=type_, priority=priority, seq=seq)


# ----------------------------------------------------------- select_in_tier


def test_select_prefers_resident_model():
    tier = [_job("img", JobType.IMAGE, seq=0), _job("llm", JobType.LLM, seq=1)]
    chosen = select_in_tier(tier, resident_model="llm", resident_type="llm")
    assert chosen.model_id == "llm"


def test_select_falls_back_to_resident_type_then_oldest():
    # resident model not present, but a same-type job is -> pick that one.
    tier = [_job("img", JobType.IMAGE, seq=0), _job("llm-b", JobType.LLM, seq=1)]
    chosen = select_in_tier(tier, resident_model="llm-a", resident_type="llm")
    assert chosen.model_id == "llm-b"


def test_select_no_resident_returns_oldest():
    tier = [_job("a", JobType.LLM, seq=0), _job("b", JobType.IMAGE, seq=1)]
    chosen = select_in_tier(tier, resident_model=None, resident_type=None)
    assert chosen.model_id == "a"  # tier is pre-sorted oldest-first


def test_select_handles_string_job_types():
    # SQLite returns enum columns as plain strings; the rule must still match.
    tier = [_job("img", "image", seq=0), _job("llm", "llm", seq=1)]
    chosen = select_in_tier(tier, resident_model="missing", resident_type="llm")
    assert chosen.model_id == "llm"


# --------------------------------------------------------------- plan_queue


def test_mixed_batch_is_one_swap():
    """The headline invariant: interleaved LLM/image jobs drain as
    LLM,LLM -> (one swap) -> image,image."""
    jobs = [
        _job("llm", JobType.LLM, seq=0),
        _job("img", JobType.IMAGE, seq=1),
        _job("llm", JobType.LLM, seq=2),
        _job("img", JobType.IMAGE, seq=3),
    ]
    swaps, steps = plan_queue(jobs, current_model_id=None, current_job_type=None)
    assert swaps == 1
    assert [(s.model_id, s.count) for s in steps] == [("llm", 2), ("img", 2)]


def test_first_load_from_idle_is_not_a_swap():
    jobs = [_job("llm", JobType.LLM, seq=0)]
    swaps, steps = plan_queue(jobs, current_model_id=None, current_job_type=None)
    assert swaps == 0
    assert [(s.model_id, s.count) for s in steps] == [("llm", 1)]


def test_switching_from_resident_counts_as_swap():
    jobs = [_job("img", JobType.IMAGE, seq=0)]
    swaps, _ = plan_queue(jobs, current_model_id="llm", current_job_type="llm")
    assert swaps == 1


def test_staying_on_resident_is_zero_swaps():
    jobs = [_job("llm", JobType.LLM, seq=0), _job("llm", JobType.LLM, seq=1)]
    swaps, steps = plan_queue(jobs, current_model_id="llm", current_job_type="llm")
    assert swaps == 0
    assert [(s.model_id, s.count) for s in steps] == [("llm", 2)]


def test_priority_tier_runs_before_lower_even_if_it_forces_a_swap():
    # A high-priority image job must run before queued LLM work.
    jobs = [
        _job("llm", JobType.LLM, priority=0, seq=0),
        _job("img", JobType.IMAGE, priority=10, seq=1),
    ]
    swaps, steps = plan_queue(jobs, current_model_id="llm", current_job_type="llm")
    assert steps[0].model_id == "img"
    # Priority wins over swap-minimization: the high-prio image must jump the
    # queue, which costs the extra swap back to the LLM (llm -> img -> llm).
    assert swaps == 2


def test_three_models_interleaved_minimizes_swaps():
    jobs = [
        _job("llm", JobType.LLM, seq=0),
        _job("flux", JobType.IMAGE, seq=1),
        _job("sdxl", JobType.IMAGE, seq=2),
        _job("flux", JobType.IMAGE, seq=3),
        _job("llm", JobType.LLM, seq=4),
    ]
    swaps, steps = plan_queue(jobs, current_model_id=None, current_job_type=None)
    # llm,llm -> swap -> flux,flux (same-model batched) -> swap -> sdxl
    assert [(s.model_id, s.count) for s in steps] == [
        ("llm", 2),
        ("flux", 2),
        ("sdxl", 1),
    ]
    assert swaps == 2


# ----------------------------------------------------------- _strip_reasoning


def test_strip_reasoning_removes_think_block():
    assert Worker._strip_reasoning("before<think>secret</think>after") == "beforeafter"


def test_strip_reasoning_handles_thinking_alias_and_case():
    assert Worker._strip_reasoning("<THINKING>x</Thinking>answer") == "answer"


def test_strip_reasoning_is_multiline():
    text = "<think>line1\nline2\nline3</think>\n\nFinal."
    assert Worker._strip_reasoning(text) == "Final."


def test_strip_reasoning_leaves_plain_text():
    assert Worker._strip_reasoning("  just an answer  ") == "just an answer"
