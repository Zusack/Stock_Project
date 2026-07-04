"""ETA estimation for Data Ingest (especially resumed runs)."""

from src.services.ingest_runner import (
    estimate_ingest_eta_seconds,
    format_ingest_eta_label,
)


def test_resume_run_does_not_use_prior_finished_as_zero_second_completions():
    """Prior finished count must not shrink average time per ticker."""
    # Resumed: 2360 already done; 600s elapsed; only baseline subtracted → no ETA yet.
    assert (
        estimate_ingest_eta_seconds(
            elapsed_sec=600.0,
            finished_total=2360,
            finished_baseline=2360,
            remaining=2840,
        )
        is None
    )
    # One ticker finished this session: 60s elapsed → 60s/ticker * 2840 remaining.
    eta = estimate_ingest_eta_seconds(
        elapsed_sec=60.0,
        finished_total=2361,
        finished_baseline=2360,
        remaining=2840,
    )
    assert eta is not None
    assert abs(eta - 170_400.0) < 1.0


def test_fresh_run_uses_session_throughput():
    eta = estimate_ingest_eta_seconds(
        elapsed_sec=100.0,
        finished_total=10,
        finished_baseline=0,
        remaining=90,
    )
    assert eta == 900.0


def test_format_eta_label_finishing():
    assert format_ingest_eta_label(
        elapsed_sec=10.0,
        finished_total=100,
        finished_baseline=0,
        remaining=0,
        total=100,
    ) == "Finishing up…"
