#!/usr/bin/env bash
# Submit the H100 matrix only after its one-time RUN_ROOT environment is usable.
set -euo pipefail

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
SETUP_PID="${SETUP_PID:?SETUP_PID is required}"
ENV_PYTHON="$RUN_ROOT/code/.venv/bin/python"
STATUS_DIR="$RUN_ROOT/status"
SUBMIT_FILE="$RUN_ROOT/code/condor/submit_full_matrix_h100.sub"

mkdir -p "$STATUS_DIR"

for _ in $(seq 1 240); do
    if [[ -x "$ENV_PYTHON" ]] && "$ENV_PYTHON" -c 'import vllm, torch, pyarrow, yaml' >/dev/null 2>&1; then
        chmod +x "$RUN_ROOT/code/condor/run_matrix_worker.sh"
        condor_submit "$SUBMIT_FILE" > "$STATUS_DIR/condor_submit.txt"
        condor_q -nobatch >> "$STATUS_DIR/condor_submit.txt"
        printf '{"status":"submitted","submitted_at":"%s"}\n' "$(date --iso-8601=seconds)" \
            > "$STATUS_DIR/submission.json"
        exit 0
    fi
    if ! kill -0 "$SETUP_PID" 2>/dev/null; then
        printf '{"status":"environment_setup_failed","checked_at":"%s"}\n' "$(date --iso-8601=seconds)" \
            > "$STATUS_DIR/submission.json"
        exit 2
    fi
    sleep 30
done

printf '{"status":"environment_setup_timed_out","checked_at":"%s"}\n' "$(date --iso-8601=seconds)" \
    > "$STATUS_DIR/submission.json"
exit 3
