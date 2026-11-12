#!/usr/bin/env bash
set -euo pipefail

# Local-only monitor. It queries Artemisa over SSH; it does not run a daemon
# on the cluster. The marker prevents repeated eviction of small jobs while
# Condor is still waiting to dispatch the H100 queue.
remote_host="${ARTEMISA_HOST:-artemisa}"
# Cluster IDs change between campaigns.  The sentinel defaults make the monitor
# discover the current user's H100-targeted jobs instead of silently watching
# an obsolete cluster.  Set either variable only when a campaign needs a
# deliberately narrow scope.
big_cluster="${H100_CLUSTER_ID:-__AUTO__}"
small_cluster="${SMALL_A100_CLUSTER_ID:-__AUTO__}"
interval_seconds="${H100_MONITOR_INTERVAL_SECONDS:-1200}"
marker_file="${H100_MONITOR_MARKER:-/tmp/llms_experiments_h100_small_released}"
h100_min_priority="${H100_MIN_PRIORITY:-10000}"
ssh_options=(-o ConnectTimeout=20 -o ServerAliveInterval=20 -o ServerAliveCountMax=2)

while true; do
    printf 'H100 monitor %s\n' "$(date -Is)"
    if ! ssh "${ssh_options[@]}" "$remote_host" bash -s -- "$big_cluster" "$small_cluster" "$marker_file" "$h100_min_priority" <<'REMOTE'
set -euo pipefail
big_cluster="$1"
small_cluster="$2"
marker_file="$3"
h100_min_priority="$4"

idle_h100="$(condor_status -constraint 'regexp("H100", Gpus_DeviceName)' -af State Activity | awk '$1 == "Unclaimed" && $2 == "Idle" {n++} END {print n + 0}')"

if [[ "$big_cluster" != "__AUTO__" ]]; then
    h100_constraint="JobStatus == 1 && JobPrio > $h100_min_priority && (isUndefined(JobBatchName) || !regexp(\"scout\", JobBatchName))"
    h100_running_constraint='JobStatus == 2 && regexp("h100", Environment)'
    h100_held_constraint='JobStatus == 5 && regexp("h100", Environment)'
    pending_h100="$(condor_q "$big_cluster" -constraint "$h100_constraint" -af ClusterId ProcId | awk 'NF {n++} END {print n + 0}')"
    running_h100="$(condor_q "$big_cluster" -constraint "$h100_running_constraint" -run -af ClusterId ProcId | awk -v c="$big_cluster" '$1 == c {n++} END {print n + 0}')"
    held_h100="$(condor_q "$big_cluster" -constraint "$h100_held_constraint" -af ClusterId ProcId | awk 'NF {n++} END {print n + 0}')"
    pending_jobs="$(condor_q "$big_cluster" -constraint "$h100_constraint" -af ClusterId ProcId | awk 'NF {print $1 "." $2}')"
else
    # Requirements is a ClassAd string; matching it keeps this independent of
    # campaign-specific cluster IDs and batch names.
    h100_constraint="JobStatus == 1 && regexp(\"h100\", Environment) && JobPrio > $h100_min_priority && (isUndefined(JobBatchName) || !regexp(\"scout\", JobBatchName))"
    h100_running_constraint='JobStatus == 2 && regexp("h100", Environment)'
    h100_held_constraint='JobStatus == 5 && regexp("h100", Environment)'
    pending_h100="$(condor_q -constraint "$h100_constraint" -af ClusterId ProcId | awk 'NF {n++} END {print n + 0}')"
    running_h100="$(condor_q -constraint "$h100_running_constraint" -af ClusterId ProcId | awk 'NF {n++} END {print n + 0}')"
    held_h100="$(condor_q -constraint "$h100_held_constraint" -af ClusterId ProcId | awk 'NF {n++} END {print n + 0}')"
    pending_jobs="$(condor_q -constraint "$h100_constraint" -af ClusterId ProcId | awk 'NF {print $1 "." $2}')"
fi

# .0 is not stable: completed or removed jobs disappear from the queue. Test
# an actually pending job, and use the number of machines reported as able to
# run it as the dispatchability signal. A failed historical match alone is not
# enough to classify the current pending job.
pending_job="$(printf '%s\n' "$pending_jobs" | awk 'NF {print; exit}')"
analysis=""
if [[ -n "$pending_job" ]]; then
    analysis="$(condor_q -better-analyze "$pending_job" 2>&1 || true)"
fi
if grep -Eq '^[[:space:]]*[1-9][0-9]* are able to run your job' <<<"$analysis"; then
    dispatchable=yes
else
    dispatchable=no
fi

printf 'H100_IDLE=%s H100_PENDING=%s H100_RUNNING=%s H100_HELD=%s PENDING_JOB=%s DISPATCHABLE=%s\n' "$idle_h100" "$pending_h100" "$running_h100" "$held_h100" "${pending_job:-none}" "$dispatchable"

if [[ "$dispatchable" == yes && "$idle_h100" -gt 0 && "$running_h100" == 0 && "$pending_h100" -gt 0 && ! -e "$marker_file" ]]; then
    if [[ "$small_cluster" != "__AUTO__" ]]; then
        candidate="$(condor_q "$small_cluster" -run -af ClusterId ProcId 2>/dev/null | awk -v c="$small_cluster" '$1 == c {print $1 "." $2; exit}')"
    else
        # Cluster IDs also change for the A100 cohort. Prefer an explicitly
        # small-model batch name, then fall back to any current-user A100 job.
        candidate="$(condor_q -run -af ClusterId ProcId Environment 2>/dev/null | awk 'BEGIN {IGNORECASE=1} /qwen3\.5-(2b|4b|9b)|llama-3\.2-1b|ministra.*3b|small/ && /a100/ {print $1 "." $2; exit}')"
        if [[ -z "$candidate" ]]; then
            candidate="$(condor_q -run -af ClusterId ProcId Environment 2>/dev/null | awk 'BEGIN {IGNORECASE=1} /a100/ {print $1 "." $2; exit}')"
        fi
    fi
    if [[ -n "$candidate" ]]; then
        condor_hold "$candidate"
        touch "$marker_file"
        printf 'ACTION held_small=%s\n' "$candidate"
    else
        printf 'ACTION no_running_small_candidate\n'
    fi
fi
REMOTE
    then
        printf 'H100 monitor: remote check failed; retrying after %ss\n' "$interval_seconds"
    fi
    sleep "$interval_seconds"
done
