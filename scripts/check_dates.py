"""Check `normalise_date`, the guard between model output and SQL.

Dates reach SQL from the model's tool-call arguments, and the model does not always
send a date. It answers "between 2014 and 2026" with bare years, and it has emitted
values carrying stray JSON punctuation. `normalise_date` turns what arrives into
something `DATE()` accepts, or into `None` so the filter is dropped rather than
passed on broken.

That makes it the one place where a malformed value could reach a query, so it is
worth its own check. Everything here is a pure function call: no workspace, no
credentials, no network.

The `end_of_year` flag exists because a bare year means different things at each end
of a range. As a start date, 2014 means 2014-01-01; as an end date it means
2014-12-31, and a range that ended on 2014-01-01 would silently exclude the year the
user asked about.

    python3 scripts/check_dates.py [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "databricks"))

from patient_tools import normalise_date  # noqa: E402

# (input, expected as a start date, expected as an end date, what it guards)
CASES: list[tuple[object, str | None, str | None, str]] = [
    ("2014-03-09", "2014-03-09", "2014-03-09", "a full date passes through unchanged"),
    ("2014", "2014-01-01", "2014-12-31", "a bare year widens to the whole year"),
    ("2014-03", "2014-03-01", "2014-03-28", "a bare month widens within the month"),
    # The model has emitted this exact value, punctuation and all, from malformed
    # JSON. Trimming it keeps the filter the question asked for; dropping it would
    # silently widen the query to all time.
    ("2023-01-01', ", "2023-01-01", "2023-01-01", "stray JSON punctuation is trimmed"),
    ("  2019-07-04  ", "2019-07-04", "2019-07-04", "surrounding whitespace is trimmed"),
    ("last Tuesday", None, None, "prose is refused rather than guessed at"),
    ("", None, None, "an empty string means no filter"),
    (None, None, None, "a missing argument means no filter"),
    # The reason this function returns None instead of passing text through: every
    # date is interpolated into SQL, so anything unparseable must not survive.
    ("';DROP TABLE x--", None, None, "SQL punctuation never becomes a date"),
    ("2014-13-45", None, None, "an impossible date is refused"),
    ("14-03-09", None, None, "a two-digit year is refused, not guessed"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    passed = failed = 0
    for value, want_start, want_end, why in CASES:
        got_start = normalise_date(value)
        got_end = normalise_date(value, end_of_year=True)
        ok = got_start == want_start and got_end == want_end
        passed, failed = passed + ok, failed + (not ok)
        if ok and not args.verbose:
            print(f"  ok   {value!r:20} -> {got_start!s:12} / {got_end!s:12} {why}")
        elif ok:
            print(f"  ok   {value!r:20} start={got_start!r} end={got_end!r}  {why}")
        else:
            print(f"  FAIL {value!r:20} {why}")
            print(f"         start: expected {want_start!r}, got {got_start!r}")
            print(f"         end:   expected {want_end!r}, got {got_end!r}")

    print(f"\n  {passed} passed, {failed} failed, {len(CASES)} cases")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
