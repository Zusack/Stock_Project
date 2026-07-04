"""Shared types for news ingestion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NewsHeadline:
    ticker: str
    date: str
    title: str
    publisher: str
    link: str
    source: str
    summary: str = ""
