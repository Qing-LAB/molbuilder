#!/bin/bash
# Stage 2 of the vacuity funnel: re-run every candidate WITH ROOM.
#
# Stage 1 (tools/vacuity.py over the whole suite, budget 12) is a SCREEN.  Its
# misses are budget-capped, and a budget-capped miss deletes working guards:
# tests/test_convergence_targets_nested.py came back VACUOUS at 12 and dies on
# the 13th mutant.  So nothing from stage 1 is a verdict until it survives this.
#
#   usage: tools/vacuity_confirm.sh <stage1.jsonl> <out.jsonl> [budget]
set -u
IN="${1:?stage-1 jsonl}"; OUT="${2:?output jsonl}"; BUDGET="${3:-60}"
source /home/qqing/miniconda3/etc/profile.d/conda.sh && conda activate molbuilder
cd "$(dirname "$0")/.."
: > "$OUT"
python - "$IN" <<'PY' > /tmp/vacuity_candidates.$$
import json, sys
for line in open(sys.argv[1]):
    try: d = json.loads(line)
    except Exception: continue
    if d.get("verdict") in ("VACUOUS", "VACUOUS-AT-BUDGET", "NO-MUTANTS"):
        print(d["file"])
PY
n=$(wc -l < /tmp/vacuity_candidates.$$)
echo "confirming $n candidates at budget $BUDGET"
# SERIAL, and with a generous per-mutant timeout: stage 2 is small, and the
# three TOOL-ERRORs of stage 1 were the outer 900s limit killing a slow file
# under 8-way load -- a timeout is not a verdict either.
i=0
while read -r f; do
  i=$((i+1)); d=$(mktemp -d); echo "[$i/$n] $f"
  SUBSUMPTION_TMP="$d" timeout 2700 python tools/vacuity.py "$f" "$BUDGET" >> "$OUT" \
    || echo "{\"file\": \"$f\", \"verdict\": \"NOT-MEASURED-TIMEOUT\"}" >> "$OUT"
  rm -rf "$d"
done < /tmp/vacuity_candidates.$$
rm -f /tmp/vacuity_candidates.$$
echo "--- confirmed ---"; cat "$OUT"
