"""Shared, fail-closed access check for the public demo functions."""

import hmac
import os


def access_error(provided: object) -> tuple[int, str] | None:
    expected = os.environ.get("PHN_PASSWORD") or ""
    if not expected:
        return 503, "Demo access is not configured."
    if not isinstance(provided, str) or not hmac.compare_digest(
        provided.encode("utf-8"), expected.encode("utf-8")
    ):
        return 401, "Wrong password."
    return None
