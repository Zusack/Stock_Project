"""Tests for Flet 0.85 snackbar helper."""

from unittest.mock import MagicMock

import flet as ft

from src.views.components.feedback import show_snackbar


def test_show_snackbar_uses_show_dialog():
    page = MagicMock()
    show_snackbar(page, "Connected", severity="success")
    page.show_dialog.assert_called_once()
    page.update.assert_called_once()
    snack = page.show_dialog.call_args[0][0]
    assert isinstance(snack, ft.SnackBar)
