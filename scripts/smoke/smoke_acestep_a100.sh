#!/usr/bin/env bash
# =============================================================================
# ACE-Step-1.5 — A100 P0 POC smoke harness
# =============================================================================
# Drives the ACE-Step REST API (acestep.api_server) end-to-end on a real GPU
# node and produces a speed/VRAM matrix + audio product sanity, following the
# methodology in gpustack/docs/新引擎内嵌gpustack-工程化方法论.md §2.1.
#
# Flow (per case): submit POST /release_task -> poll POST /query_result every 5s
#   -> sample VRAM (MAX across all cards) / GPU util / Shmem / MemAvailable
#   -> accept ONLY status==succeeded -> product de-risk (size / duration / silence).
#
# Two modes:
#   1) Managed container:  give IMAGE + CKPT_DIR, the script docker-runs the API.
#   2) External server:    give BASE_URL of an already-running api_server.
#
# Usage:
#   # managed container (starts + tears down the API container):
#   IMAGE=crpi-.../reputationly/acestep:arm64-a100-latest \
#   CKPT_DIR=/nfs-models/ace-step/checkpoints \
#   GPUS='"device=0"' \
#   LM_MODEL=acestep-5Hz-lm-4B  LM_BACKEND=pt \
#   REF_AUDIO=/nfs-data/smoke/ref.wav  SRC_AUDIO=/nfs-data/smoke/src.wav \
#   bash scripts/smoke/smoke_acestep_a100.sh
#
#   # external server already up on :8001:
#   BASE_URL=http://127.0.0.1:8001 bash scripts/smoke/smoke_acestep_a100.sh
#
# De-risk knobs: SKIP_GPU_GUARD=1 to skip the pre-flight idle-GPU check.
# =============================================================================
set -uo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMAGE="${IMAGE:-}"
CKPT_DIR="${CKPT_DIR:-}"
BASE_URL="${BASE_URL:-}"
GPUS="${GPUS:-\"device=0\"}"
PORT="${PORT:-8001}"
CONTAINER_NAME="${CONTAINER_NAME:-acestep-smoke}"

CONFIG_PATH="${CONFIG_PATH:-acestep-v15-turbo}"   # DiT config/model name under checkpoints
LM_MODEL="${LM_MODEL:-acestep-5Hz-lm-4B}"         # 5Hz LM name (4B/1.7B/0.6B)
LM_BACKEND="${LM_BACKEND:-pt}"                     # pt avoids nano-vllm/flash-attn on arm A100
AUDIO_FORMAT="${AUDIO_FORMAT:-mp3}"

REF_AUDIO="${REF_AUDIO:-}"   # host path for cover reference (mounted into container)
SRC_AUDIO="${SRC_AUDIO:-}"   # host path for repaint source (mounted into container)

DURATION="${DURATION:-30}"        # seconds of music to generate (keep POC short)
INFER_STEPS="${INFER_STEPS:-8}"   # turbo default
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"  # model load can take minutes (DiT + LM)
JOB_TIMEOUT="${JOB_TIMEOUT:-1200}"        # per-generation cap
POLL_INTERVAL="${POLL_INTERVAL:-5}"

OUT_DIR="${OUT_DIR:-./smoke_out_$(date -u +%Y%m%d-%H%M%S 2>/dev/null || echo run)}"
MIN_BYTES="${MIN_BYTES:-20480}"   # <20KB => fail (empty-file false-green guard)

mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/report.txt"
: > "$REPORT"

log()  { printf '\033[36m[smoke]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[smoke][WARN]\033[0m %s\n' "$*"; }
err()  { printf '\033[31m[smoke][ERR ]\033[0m %s\n' "$*"; }

_started_container=0
cleanup() {
  if [ "$_started_container" = "1" ]; then
    log "tearing down container $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# GPU pre-flight (methodology: clear stray procs before measuring)
# ---------------------------------------------------------------------------
gpu_used_max() {
  # MAX memory.used (MiB) across ALL visible cards — multi-card takes the max.
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | sort -n | tail -1 | tr -d ' '
}
gpu_util_max() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | sort -n | tail -1 | tr -d ' '
}
mem_kb() { awk -v k="$1" '$1==k":"{print $2}' /proc/meminfo 2>/dev/null; }

if [ "${SKIP_GPU_GUARD:-0}" != "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  used="$(gpu_used_max)"
  if [ -n "${used:-}" ] && [ "$used" -gt 2000 ]; then
    warn "a GPU already holds ${used} MiB — stray process? (SKIP_GPU_GUARD=1 to bypass)"
  fi
fi

# ---------------------------------------------------------------------------
# Bring up the API (managed container) or use external BASE_URL
# ---------------------------------------------------------------------------
if [ -z "$BASE_URL" ]; then
  [ -n "$IMAGE" ]    || { err "set IMAGE (or BASE_URL for an external server)"; exit 2; }
  [ -n "$CKPT_DIR" ] || { err "set CKPT_DIR (checkpoints mount)"; exit 2; }

  mounts=(-v "$CKPT_DIR:/app/checkpoints")
  [ -n "$REF_AUDIO" ] && mounts+=(-v "$REF_AUDIO:$REF_AUDIO:ro")
  [ -n "$SRC_AUDIO" ] && mounts+=(-v "$SRC_AUDIO:$SRC_AUDIO:ro")

  log "starting container $CONTAINER_NAME (image=$IMAGE gpus=$GPUS lm=$LM_MODEL backend=$LM_BACKEND)"
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker run -d --name "$CONTAINER_NAME" --gpus "$GPUS" --network host --memory=240g \
    -e ACESTEP_MODE=api \
    -e ACESTEP_API_HOST=0.0.0.0 -e ACESTEP_API_PORT="$PORT" \
    -e ACESTEP_INIT_SERVICE=true \
    -e ACESTEP_CONFIG_PATH="$CONFIG_PATH" \
    -e ACESTEP_LM_MODEL_PATH="$LM_MODEL" \
    -e ACESTEP_LLM_BACKEND="$LM_BACKEND" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    "${mounts[@]}" "$IMAGE" >/dev/null || { err "docker run failed"; exit 2; }
  _started_container=1
  BASE_URL="http://127.0.0.1:$PORT"
fi
log "API base: $BASE_URL"

# ---------------------------------------------------------------------------
# Wait for /health, record load time
# ---------------------------------------------------------------------------
log "waiting for /health (timeout ${HEALTH_TIMEOUT}s) ..."
t0=$(date +%s)
until curl -sf "$BASE_URL/health" >/dev/null 2>&1; do
  now=$(date +%s)
  if [ $((now - t0)) -ge "$HEALTH_TIMEOUT" ]; then
    err "health timeout after ${HEALTH_TIMEOUT}s"
    [ "$_started_container" = "1" ] && docker logs --tail 60 "$CONTAINER_NAME" 2>&1 | sed 's/^/  | /'
    exit 3
  fi
  sleep 3
done
LOAD_SEC=$(( $(date +%s) - t0 ))
log "healthy after ${LOAD_SEC}s (model load)"
echo "load_seconds=$LOAD_SEC  config=$CONFIG_PATH  lm=$LM_MODEL  backend=$LM_BACKEND" >> "$REPORT"

# ---------------------------------------------------------------------------
# Python driver: submit + poll + download one case (stdlib only)
# ---------------------------------------------------------------------------
run_case() {
  # args: name  payload_json_file  out_audio_path
  local name="$1" payload="$2" outp="$3"
  local pid maxfile utilfile
  maxfile="$(mktemp)"; utilfile="$(mktemp)"; echo 0 > "$maxfile"; echo 0 > "$utilfile"

  # background VRAM/util sampler (MAX across cards)
  ( while :; do
      u="$(gpu_used_max)"; g="$(gpu_util_max)"
      [ -n "$u" ] && [ "$u" -gt "$(cat "$maxfile")" ] && echo "$u" > "$maxfile"
      [ -n "$g" ] && [ "$g" -gt "$(cat "$utilfile")" ] && echo "$g" > "$utilfile"
      sleep 2
    done ) & pid=$!

  local line status infer_sec
  line="$(BASE_URL="$BASE_URL" JOB_TIMEOUT="$JOB_TIMEOUT" POLL_INTERVAL="$POLL_INTERVAL" \
          OUT_PATH="$outp" PAYLOAD_FILE="$payload" python3 - <<'PY'
import json, os, sys, time, urllib.request

base = os.environ["BASE_URL"].rstrip("/")
timeout = int(os.environ["JOB_TIMEOUT"])
interval = int(os.environ["POLL_INTERVAL"])
out_path = os.environ["OUT_PATH"]
payload = json.load(open(os.environ["PAYLOAD_FILE"]))

def post(path, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

def emit(status, infer_sec, url=""):
    print(f"RESULT|{status}|{infer_sec}|{url}")

try:
    resp = post("/release_task", payload)
except Exception as e:
    emit("submit_error", 0, str(e)[:80]); sys.exit(0)

data = (resp or {}).get("data") or {}
task_id = data.get("task_id")
if not task_id:
    emit("no_task_id", 0, json.dumps(resp)[:120]); sys.exit(0)

t0 = time.time()
while True:
    if time.time() - t0 > timeout:
        emit("timeout", int(time.time() - t0)); sys.exit(0)
    time.sleep(interval)
    try:
        q = post("/query_result", {"task_id_list": json.dumps([task_id])})
    except Exception:
        continue
    items = (q or {}).get("data") or []
    if not items:
        continue
    item = items[0]
    st = item.get("status")            # 0 running/queued, 1 succeeded, 2 failed
    if st == 2:
        emit("failed", int(time.time() - t0), str(item.get("progress_text",""))[:80]); sys.exit(0)
    if st == 1:
        infer_sec = int(time.time() - t0)
        try:
            inner = json.loads(item.get("result") or "[]")
        except Exception:
            inner = []
        f = inner[0].get("file") if inner and isinstance(inner[0], dict) else None
        if not f:
            emit("no_audio", infer_sec); sys.exit(0)
        # f is like /v1/audio?path=<urlencoded abs path>
        url = f if f.startswith("http") else base + f
        try:
            urllib.request.urlretrieve(url, out_path)
        except Exception as e:
            emit("download_error", infer_sec, str(e)[:80]); sys.exit(0)
        emit("succeeded", infer_sec, url); sys.exit(0)
PY
)"
  kill "$pid" >/dev/null 2>&1 || true
  local vram util
  vram="$(cat "$maxfile")"; util="$(cat "$utilfile")"; rm -f "$maxfile" "$utilfile"

  status="$(echo "$line" | awk -F'|' '/^RESULT\|/{print $2}')"
  infer_sec="$(echo "$line" | awk -F'|' '/^RESULT\|/{print $3}')"
  [ -z "$status" ] && status="driver_error"

  # ---- product de-risk (audio) ----
  local verdict="$status" bytes=0 dur="-"
  if [ "$status" = "succeeded" ] && [ -f "$outp" ]; then
    bytes=$(wc -c < "$outp" | tr -d ' ')
    if [ "$bytes" -lt "$MIN_BYTES" ]; then
      verdict="FAIL(tiny:${bytes}B)"
    elif command -v ffprobe >/dev/null 2>&1; then
      dur="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$outp" 2>/dev/null | cut -d. -f1)"
      [ -z "$dur" ] && dur="?"
      if command -v ffmpeg >/dev/null 2>&1; then
        # whole-clip silence => fail (audio equivalent of black-screen)
        silence="$(ffmpeg -hide_banner -nostats -i "$outp" -af silencedetect=noise=-50dB:d=2 -f null - 2>&1 \
                   | grep -c 'silence_start: 0' || true)"
        [ "${silence:-0}" -ge 1 ] && verdict="WARN(silent-head)"
      fi
      [ "$verdict" = "succeeded" ] && verdict="OK"
    else
      verdict="OK(no-ffprobe)"
    fi
  fi

  printf '%-16s | %-10s | load=%-4ss infer=%-5ss | vram=%-6sMiB util=%-3s%% | %sB dur=%ss | %s\n' \
    "$name" "$verdict" "$LOAD_SEC" "${infer_sec:-?}" "$vram" "$util" "$bytes" "$dur" "$outp" | tee -a "$REPORT"
}

# ---------------------------------------------------------------------------
# Build payloads (stdlib json, safe for unicode lyrics)
# ---------------------------------------------------------------------------
mkp() {
  # args: task_type  extra_json  -> writes payload file, echoes its path
  local tt="$1" extra="$2"
  local f="$OUT_DIR/payload_${tt}.json"
  DURATION="$DURATION" STEPS="$INFER_STEPS" FMT="$AUDIO_FORMAT" LM="$LM_MODEL" \
  BACKEND="$LM_BACKEND" TT="$tt" EXTRA="$extra" python3 - "$f" <<'PY'
import json, os, sys
p = {
    "prompt": "An upbeat electronic pop track with bright synths, punchy drums and a catchy melodic hook.",
    "lyrics": "[inst]",
    "thinking": True,
    "task_type": os.environ["TT"],
    "audio_duration": float(os.environ["DURATION"]),
    "inference_steps": int(os.environ["STEPS"]),
    "guidance_scale": 7.0,
    "use_random_seed": False,
    "seed": 42,
    "audio_format": os.environ["FMT"],
    "lm_model_path": os.environ["LM"],
    "lm_backend": os.environ["BACKEND"],
    "vocal_language": "en",
    "bpm": 120,
}
extra = os.environ.get("EXTRA") or ""
if extra:
    p.update(json.loads(extra))
json.dump(p, open(sys.argv[1], "w"), ensure_ascii=False)
PY
  echo "$f"
}

# ---------------------------------------------------------------------------
# Run matrix
# ---------------------------------------------------------------------------
log "results ->"
echo "case             | verdict    | timing                  | resources                | product" | tee -a "$REPORT"

# A) text2music (pure text, instrumental)
run_case "t2m-inst" "$(mkp text2music '{}')" "$OUT_DIR/t2m.${AUDIO_FORMAT}"

# B) text2music with lyrics (vocal)
run_case "t2m-vocal" "$(mkp text2music '{"lyrics":"[Verse]\nRunning through the night\n[Chorus]\nWe are alive"}')" \
  "$OUT_DIR/t2m_vocal.${AUDIO_FORMAT}"

# C) cover (needs reference audio)
if [ -n "$REF_AUDIO" ]; then
  run_case "cover" "$(mkp cover "$(printf '{"reference_audio_path":"%s","audio_cover_strength":1.0}' "$REF_AUDIO")")" \
    "$OUT_DIR/cover.${AUDIO_FORMAT}"
else
  warn "REF_AUDIO not set — skipping cover case"
  echo "cover            | SKIP       | (set REF_AUDIO)" >> "$REPORT"
fi

# D) repaint (needs source audio + region)
if [ -n "$SRC_AUDIO" ]; then
  run_case "repaint" "$(mkp repaint "$(printf '{"src_audio_path":"%s","repainting_start":5.0,"repainting_end":15.0,"repaint_mode":"balanced","repaint_strength":0.5}' "$SRC_AUDIO")")" \
    "$OUT_DIR/repaint.${AUDIO_FORMAT}"
else
  warn "SRC_AUDIO not set — skipping repaint case"
  echo "repaint          | SKIP       | (set SRC_AUDIO)" >> "$REPORT"
fi

# ---------------------------------------------------------------------------
# Post-run resource snapshot
# ---------------------------------------------------------------------------
if [ -e /proc/meminfo ]; then
  shmem_kb="$(mem_kb Shmem)"; avail_kb="$(mem_kb MemAvailable)"
  printf 'host: Shmem=%sMiB MemAvailable=%sMiB\n' \
    "$(( ${shmem_kb:-0} / 1024 ))" "$(( ${avail_kb:-0} / 1024 ))" | tee -a "$REPORT"
fi

log "report written to $REPORT"
log "audio artifacts in $OUT_DIR"
