from __future__ import annotations

import os

import flet as ft

from src.app import main as desktop_main


def run() -> None:
    os.environ.setdefault("APP_PROFILE", "desktop")
    ft.run(main=desktop_main)


if __name__ == "__main__":
    run()
