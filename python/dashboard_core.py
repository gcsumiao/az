"""Compatibility wrapper around the extracted `core/` implementation.

This keeps the existing Streamlit `app.py` working while we migrate the production UI to v0/shadcn + API.
"""

from __future__ import annotations

from core.data import *  # noqa: F401,F403

