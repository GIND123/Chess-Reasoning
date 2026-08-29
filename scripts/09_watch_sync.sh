#!/usr/bin/env bash
# Poll the Modal volumes for finished shards and mirror them to the Hub.
# Dynamic fallback for a run already in flight: the Volume is the working store,
# the Hub is what survives losing the app.
set -u
REPO="${CHESSR_HUB_REPO:-GOVINDFROM/chess-process-verified}"
STAGE="${1:-generations}"          # generations | tables
VOL="${2:-govind-llm-shards}"
LOCAL="data/interim/${STAGE}"
INTERVAL="${INTERVAL:-180}"
mkdir -p "$LOCAL"
seen=""
while true; do
  files=$(modal volume ls "$VOL" 2>/dev/null | grep -oE "[A-Za-z0-9_.-]+\.jsonl" | sort -u || true)
  for f in $files; do
    case " $seen " in *" $f "*) continue;; esac
    if modal volume get "$VOL" "/$f" "$LOCAL/$f" --force >/dev/null 2>&1; then
      if python3 -c "
import sys,os;sys.path.insert(0,'src')
from chessr.hub import HubSync
HubSync(repo_id='$REPO').put_file('$LOCAL/$f','${STAGE}/$f')
" 2>/dev/null; then
        echo "synced $STAGE/$f"
        seen="$seen $f"
      fi
    fi
  done
  sleep "$INTERVAL"
done
