"""Portable, lexical names for CSV metadata (never filesystem resolution)."""

import ntpath


def csv_basename(value: object) -> str:
    """Treat either slash as a separator, even a literal POSIX backslash."""
    return ntpath.basename(str(value)) if value is not None else ""
