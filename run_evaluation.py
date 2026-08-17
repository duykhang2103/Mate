"""Local stop notice for the archived SafePruneVid Modal pipeline."""

import sys


ARCHIVE_MESSAGE = (
    "SafePruneVid was archived on 2026-08-17. "
    "The old Modal pipeline is disabled before cloud allocation. "
    "See archive/safeprunevid/README.md."
)


def run_pipeline(stage=None):
    """Reject legacy SafePruneVid launches without allocating cloud compute."""
    del stage
    raise RuntimeError(ARCHIVE_MESSAGE)


if __name__ == "__main__":
    print(ARCHIVE_MESSAGE, file=sys.stderr)
    raise SystemExit(2)
