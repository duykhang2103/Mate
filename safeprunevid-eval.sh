#!/usr/bin/env bash

set -euo pipefail

printf '%s\n' \
    "SafePruneVid was archived on 2026-08-17 after failing its acceptance gate." \
    "No additional epsilon, full, router, or MVBench stages should be launched." \
    "See archive/safeprunevid/README.md for evidence and artifact locations." \
    "For reproduction only, use the guarded runner in archive/safeprunevid/." >&2
exit 2
