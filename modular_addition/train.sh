#!/usr/bin/env bash
# Compatibility entry point for paper sweep groups.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/paper_sweeps.sh" "$@"
