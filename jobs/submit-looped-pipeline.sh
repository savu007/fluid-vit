#!/bin/bash
set -euo pipefail
bash "$(dirname -- "${BASH_SOURCE[0]}")/submit-looped-experiment.sh" pipeline
