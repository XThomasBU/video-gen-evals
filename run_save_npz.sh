#!/usr/bin/env bash
set -euo pipefail

# ===== Config (edit as needed) =====
ACTION_LIST_FILE="/home/coder/projects/video_evals/video-gen-evals/src_final/K700_classes.txt"
PY_ENTRY="src_final.human_mesh.save_to_npz"
ENV_NAME="tkhmr"
NUM_GPUS=8                        # GPUs: 0..7
PARALLEL_PER_GPU=5                # <-- run this many actions concurrently per GPU
TIMEOUT_SECS=${TIMEOUT_SECS:-28800}  # per-action timeout (2h); change as needed
RETRIES=${RETRIES:-0}             # how many times to retry a failed/timeout item
LOG_ROOT="/home/coder/projects/video_evals/video-gen-evals/local_runs_logs"

# Optional: headless GL if your renderer needs it
export PYGLET_HEADLESS=${PYGLET_HEADLESS:-True}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export EGL_PLATFORM=${EGL_PLATFORM:-surfaceless}
# If EGL troubles you can switch to OSMesa:
# export PYOPENGL_PLATFORM=osmesa; unset EGL_PLATFORM

# Threading knobs
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS:-4}

# ===== Conda / tools =====
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
# non-interactive hook (helps in some setups)
if [[ -x "/home/coder/miniconda3/bin/conda" ]]; then
  eval "$(/home/coder/miniconda3/bin/conda shell.bash hook)"
fi
conda activate "${ENV_NAME}"

command -v ffmpeg >/dev/null || {
  echo "[WARN] ffmpeg not found on PATH. Install with 'sudo apt-get install -y ffmpeg' or 'conda install -y -c conda-forge ffmpeg'."
}

mkdir -p "${LOG_ROOT}"

# ===== Read & normalize the action list (single-line CSV OK) =====
RAW_LINE="$(cat "${ACTION_LIST_FILE}")"
NORMALIZED_LINE="$(
  printf '%s' "${RAW_LINE}" \
    | tr '\n' ' ' \
    | sed -e 's/^\s*//; s/\s*$//' \
          -e 's/[][]//g' \
          -e 's/"//g; s/'"'"'//g' \
          -e 's/[;|]/,/g'
)"
IFS=',' read -r -a CLASSES <<< "${NORMALIZED_LINE}"

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

TOTAL=${#CLASSES[@]}
if (( TOTAL == 0 )); then
  echo "[ERROR] No classes parsed from ${ACTION_LIST_FILE}"
  exit 1
fi

echo "[$(date)] Total classes parsed: ${TOTAL}"
echo "[$(date)] Spawning ${NUM_GPUS} GPU workers (0..$((NUM_GPUS-1))), ${PARALLEL_PER_GPU} per GPU. Logs -> ${LOG_ROOT}"

# Ensure we clean up children if this script is interrupted
pids=()
cleanup() {
  echo "[INFO] Caught signal, terminating ${#pids[@]} workers..."
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait || true
}
trap cleanup INT TERM

# ===== helper: run one action with timeout+retry =====
run_one_action() {
  local gpu_id="$1" idx="$2" action="$3" logf="$4"
  local attempt=0
  while :; do
    attempt=$((attempt + 1))
    if timeout -s SIGINT "${TIMEOUT_SECS}" \
         python -u -m "${PY_ENTRY}" --action "${action}" \
         > "${logf}" 2>&1; then
      echo "[$(date)] [GPU ${gpu_id}] OK: (#${idx}/${TOTAL}) ${action}"
      return 0
    else
      echo "[$(date)] [GPU ${gpu_id}] FAIL (attempt ${attempt}): ${action}. See ${logf}"
      if (( attempt > RETRIES )); then
        echo "[$(date)] [GPU ${gpu_id}] giving up: ${action}"
        return 1
      fi
      sleep 5
    fi
  done
}

# ===== Worker: each GPU takes indices g, g+NUM_GPUS, g+2*NUM_GPUS, ... with a per-GPU concurrency limit =====
run_worker() {
  local gpu_id="$1"
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  echo "[$(date)] [GPU ${gpu_id}] starting"

  # Track background child PIDs for this GPU
  local children=()
  local in_flight=0
  local processed=0

  # Iterate over this GPU's slice with stride = NUM_GPUS
  local idx="$gpu_id"
  while (( idx < TOTAL )); do
    local raw="${CLASSES[idx]}" action ts logf
    action="$(trim "${raw}")"
    if [[ -z "${action}" ]]; then
      idx=$(( idx + NUM_GPUS ))
      continue
    fi

    ts="$(date +%Y%m%d_%H%M%S)"
    logf="${LOG_ROOT}/gpu${gpu_id}_${idx}_$(echo "${action}" | tr ' /' '__').log"
    echo "[$(date)] [GPU ${gpu_id}] queue -> (#${idx}/${TOTAL}) ${action}"

    # throttle to PARALLEL_PER_GPU
    while (( in_flight >= PARALLEL_PER_GPU )); do
      # wait -n is Bash 5+. If unavailable, fall back to "wait for any" via jobs.
      if wait -n 2>/dev/null; then
        in_flight=$(( in_flight - 1 ))
      else
        # Fallback: wait for the first child PID then shrink
        if ((${#children[@]})); then
          wait "${children[0]}" || true
          children=("${children[@]:1}")
          in_flight=$(( in_flight - 1 ))
        fi
      fi
    done

    # Launch this action in background on this GPU
    (
      # Each child must inherit CUDA_VISIBLE_DEVICES; already exported above.
      run_one_action "${gpu_id}" "${idx}" "${action}" "${logf}"
    ) &
    children+=("$!")
    in_flight=$(( in_flight + 1 ))
    processed=$(( processed + 1 ))

    idx=$(( idx + NUM_GPUS ))
  done

  # Drain remaining tasks
  for cpid in "${children[@]:-}"; do wait "$cpid" || true; done
  echo "[$(date)] [GPU ${gpu_id}] done (launched ${processed} actions)."
}

# ===== Launch workers: one per GPU =====
for gpu in $(seq 0 $((NUM_GPUS-1))); do
  run_worker "$gpu" &
  pids+=("$!")
done

# ===== Wait for all workers and propagate failure if any =====
rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then rc=1; fi
done

echo "[$(date)] All GPU workers finished. Exit code: ${rc}"
exit "${rc}"