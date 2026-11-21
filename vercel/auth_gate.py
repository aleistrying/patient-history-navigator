"""Shared, fail-closed access check for the public demo functions.

Two credentials open the demo: the shared password typed into the gate form
(PHN_PASSWORD) and an optional link key (PHN_ACCESS_KEY) that lets a shared link
such as /?key=... open the page without typing anything. Both are environment
variables; neither is stored in the repository.
"""

import hmac
import os


def _matches(provided: bytes, expected: str) -> bool:
    return bool(expected) and hmac.compare_digest(provided, expected.encode("utf-8"))


def access_error(provided: object) -> tuple[int, str] | None:
    password = os.environ.get("PHN_PASSWORD") or ""
    link_key = os.environ.get("PHN_ACCESS_KEY") or ""
    if not password and not link_key:
        return 503, "Demo access is not configured."
    if not isinstance(provided, str):
        return 401, "Wrong password."
    candidate = provided.encode("utf-8")
    # Evaluate both so timing does not reveal which credential was closer.
    ok_password = _matches(candidate, password)
    ok_key = _matches(candidate, link_key)
    if ok_password or ok_key:
        return None
    return 401, "Wrong password."
