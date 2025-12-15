#!/usr/bin/env bash

set -euo pipefail

exec python3 -m uvicorn api.main:app --reload --port 8000

