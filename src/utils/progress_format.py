"""Shared duration and ETA formatting for long-running UI tasks."""

from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Human-readable duration (for ETA or elapsed)."""
    if seconds != seconds or seconds < 0:  # NaN
        return ""
    total = int(round(seconds))
    if total <= 0:
        return "less than a minute"
    if total < 60:
        return f"{total} sec"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        if secs:
            return f"{minutes} min {secs} sec"
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    if minutes:
        return f"{hours} hr {minutes} min"
    return f"{hours} hr"


def format_eta_remaining(seconds: float) -> str:
    """Human-readable ETA from estimated seconds left."""
    label = format_duration(seconds)
    if not label:
        return ""
    if label == "less than a minute":
        return "Less than a minute remaining"
    return f"About {label} remaining"


def estimate_eta_seconds(
    *,
    elapsed_sec: float,
    done: int,
    total: int,
    min_samples: int = 3,
) -> float | None:
    """Estimate seconds remaining from throughput in the current session."""
    if total <= 0 or done <= 0 or done >= total:
        return 0.0 if done >= total and total > 0 else None
    if elapsed_sec < 0 or not (elapsed_sec == elapsed_sec):
        return None
    if done < min_samples:
        return None
    remaining = total - done
    return (elapsed_sec / done) * remaining


def format_elapsed(seconds: float) -> str:
    """User-facing elapsed time label."""
    label = format_duration(seconds)
    if not label:
        return ""
    return f"Elapsed {label}"
