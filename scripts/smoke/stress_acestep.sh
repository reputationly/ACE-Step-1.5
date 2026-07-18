#!/usr/bin/env bash
# =============================================================================
# ACE-Step-1.5 — single-container stress sweep (duration x steps x batch)
# =============================================================================
# Loads ONE server once (model load ~2min on NFS) then loops many requests to
# find the VRAM ceiling / quality ceiling, mirroring LightX2V's
# test_wan_i2v_stress.sh. Do NOT reload per case.
#
# Usage:
#   IMAGE=<acr>/reputationly/acestep:arm64-a100-latest \
#   CKPT_DIR=/nfs-models/.../ACE-Step-1.5 \
#   GPUS='"device=0"' CONFIG_PATH=acestep-v15-xl-turbo LM_MODEL=acestep-5Hz-lm-4B \
#   DURATIONS="30 60 120 240 480 600" STEPS="8" BATCHES="2" \
#   bash scripts/smoke/stress_acestep.sh
#
# Each row records: gen(hot) time, MAX-across-cards peak VRAM, product size,
# and (if ffmpeg present) real duration + head-silence. First request also pays
# the one-time model load — its time is flagged (load+gen), later rows are hot.
# =============================================================================
set -uo pipefail

IMAGE="${IMAGE:?set IMAGE}"
CKPT_DIR="${CKPT_DIR:?set CKPT_DIR}"
GPUS="${GPUS:-\"device=0\"}"
PORT="${PORT:-8001}"
CONTAINER_NAME="${CONTAINER_NAME:-acestep-stress}"
CONFIG_PATH="${CONFIG_PATH:-acestep-v15-turbo}"
LM_MODEL="${LM_MODEL:-acestep-5Hz-lm-4B}"
LM_BACKEND="${LM_BACKEND:-pt}"
AUDIO_FORMAT="${AUDIO_FORMAT:-mp3}"

DURATIONS="${DURATIONS:-30 60 120 240}"
STEPS="${STEPS:-8}"
BATCHES="${BATCHES:-2}"
TASK="${TASK:-text2music}"

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
JOB_TIMEOUT="${JOB_TIMEOUT:-3600}"     # long durations generate longer
POLL_INTERVAL="${POLL_INTERVAL:-5}"
MIN_BYTES="${MIN_BYTES:-20480}"

OUT_DIR="${OUT_DIR:-./stress_out_$(date -u +%Y%m%d-%H%M%S 2>/dev/null || echo run)}"
mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/report.txt"; : > "$REPORT"

log()  { printf '\033[36m[stress]\033[0m %s\n' "$*"; }
gpu_used_max() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1 | tr -d ' '; }

cleanup() { docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# ---- start one container ----
log "starting $CONTAINER_NAME (dit=$CONFIG_PATH lm=$LM_MODEL)"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER_NAME" --gpus "$GPUS" --network host --memory=240g \
  -e ACESTEP_MODE=api -e ACESTEP_API_HOST=0.0.0.0 -e ACESTEP_API_PORT="$PORT" \
  -e ACESTEP_CONFIG_PATH="$CONFIG_PATH" -e ACESTEP_LM_MODEL_PATH="$LM_MODEL" \
  -e ACESTEP_LLM_BACKEND="$LM_BACKEND" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$CKPT_DIR:/app/checkpoints" "$IMAGE" >/dev/null || { echo "docker run failed"; exit 2; }
BASE_URL="http://127.0.0.1:$PORT"

# ---- wait /health ----
log "waiting /health (timeout ${HEALTH_TIMEOUT}s) ..."
t0=$(date +%s)
until curl -sf "$BASE_URL/health" >/dev/null 2>&1; do
  [ $(( $(date +%s)-t0 )) -ge "$HEALTH_TIMEOUT" ] && { echo "health timeout"; docker logs --tail 40 "$CONTAINER_NAME"; exit 3; }
  sleep 3
done
log "server healthy (models lazy-load on first request below)"

# ---- one request driver ----
run_one() {
  # args: duration steps batch label; echoes "gen_sec|peak_vram|bytes|dur|verdict"
  local dur="$1" steps="$2" batch="$3"
  local payload="$OUT_DIR/p_${dur}_${steps}_${batch}.json"
  local outp="$OUT_DIR/a_${dur}s_st${steps}_b${batch}.${AUDIO_FORMAT}"
  DUR="$dur" STEPS="$steps" BATCH="$batch" TASK="$TASK" FMT="$AUDIO_FORMAT" LM="$LM_MODEL" python3 - "$payload" <<'PY'
import json, os, sys
p = {
  "prompt": "an upbeat electronic pop track with bright synths and punchy drums",
  "lyrics": "[inst]", "thinking": True, "task_type": os.environ["TASK"],
  "audio_duration": float(os.environ["DUR"]), "inference_steps": int(os.environ["STEPS"]),
  "batch_size": int(os.environ["BATCH"]), "guidance_scale": 7.0,
  "use_random_seed": False, "seed": 42, "audio_format": os.environ["FMT"],
  "lm_model_path": os.environ["LM"], "lm_backend": "pt", "vocal_language": "en", "bpm": 120,
}
json.dump(p, open(sys.argv[1], "w"))
PY

  local maxfile pid; maxfile="$(mktemp)"; echo 0 > "$maxfile"
  ( while :; do u="$(gpu_used_max)"; [ -n "$u" ] && [ "$u" -gt "$(cat "$maxfile")" ] && echo "$u" > "$maxfile"; sleep 2; done ) & pid=$!

  local line
  line="$(BASE_URL="$BASE_URL" JOB_TIMEOUT="$JOB_TIMEOUT" POLL_INTERVAL="$POLL_INTERVAL" OUT_PATH="$outp" PAYLOAD_FILE="$payload" python3 - <<'PY'
import json, os, time, urllib.request
base=os.environ["BASE_URL"].rstrip("/"); to=int(os.environ["JOB_TIMEOUT"]); iv=int(os.environ["POLL_INTERVAL"])
outp=os.environ["OUT_PATH"]; payload=json.load(open(os.environ["PAYLOAD_FILE"]))
def post(path,obj):
    r=urllib.request.Request(base+path,data=json.dumps(obj).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=120) as x: return json.load(x)
try: resp=post("/release_task",payload)
except Exception as e: print(f"R|submit_error|0"); raise SystemExit
tid=(resp.get("data") or {}).get("task_id")
if not tid: print("R|no_task_id|0"); raise SystemExit
t0=time.time()
while True:
    if time.time()-t0>to: print(f"R|timeout|{int(time.time()-t0)}"); break
    time.sleep(iv)
    try: q=post("/query_result",{"task_id_list":json.dumps([tid])})
    except Exception: continue
    it=(q.get("data") or [None])[0]
    if not it: continue
    s=it.get("status")
    if s==2: print(f"R|failed|{int(time.time()-t0)}"); break
    if s==1:
        sec=int(time.time()-t0)
        try: inner=json.loads(it.get("result") or "[]")
        except Exception: inner=[]
        f=inner[0].get("file") if inner and isinstance(inner[0],dict) else None
        if f:
            try: urllib.request.urlretrieve(f if f.startswith("http") else base+f, outp)
            except Exception: pass
        print(f"R|ok|{sec}"); break
PY
)"
  kill "$pid" >/dev/null 2>&1 || true
  local vram; vram="$(cat "$maxfile")"; rm -f "$maxfile"
  local status sec; status="$(echo "$line" | awk -F'|' '/^R\|/{print $2}')"; sec="$(echo "$line" | awk -F'|' '/^R\|/{print $3}')"
  local bytes=0 rdur="-" verdict="$status"
  if [ "$status" = "ok" ] && [ -f "$outp" ]; then
    bytes=$(wc -c < "$outp" | tr -d ' ')
    [ "$bytes" -lt "$MIN_BYTES" ] && verdict="FAIL(tiny)" || verdict="OK"
    if command -v ffprobe >/dev/null 2>&1; then rdur="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$outp" 2>/dev/null | cut -d. -f1)"; fi
  fi
  printf '%-6ss st%-3s b%-2s | %-10s | gen=%-5ss | vram=%-6sMiB | %sB dur=%ss\n' \
    "$dur" "$steps" "$batch" "$verdict" "${sec:-?}" "$vram" "$bytes" "${rdur:-?}" | tee -a "$REPORT"
}

# ---- sweep ----
echo "dit=$CONFIG_PATH lm=$LM_MODEL task=$TASK  (first row includes one-time model load)" | tee -a "$REPORT"
printf '%-6s %-4s %-3s | %-10s | %-9s | %-12s | product\n' "dur" "step" "bat" "verdict" "gen" "vram" | tee -a "$REPORT"
first=1
for b in $BATCHES; do
  for st in $STEPS; do
    for d in $DURATIONS; do
      [ "$first" = "1" ] && { log "first request pays model load (~2min) ..."; first=0; }
      run_one "$d" "$st" "$b"
    done
  done
done

# ---- host snapshot ----
if [ -e /proc/meminfo ]; then
  awk '/Shmem:|MemAvailable:/{printf "%s %dMiB  ",$1,$2/1024}' /proc/meminfo | tee -a "$REPORT"; echo | tee -a "$REPORT"
fi
log "report: $REPORT ; audio in $OUT_DIR"
