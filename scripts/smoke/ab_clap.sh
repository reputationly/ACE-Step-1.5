#!/usr/bin/env bash
# =============================================================================
# ab_clap.sh — objective A/B of two ACE-Step DiT variants via CLAP alignment
# =============================================================================
# Generates the SAME K prompts (same seed) on two DiT variants running on two
# GPUs in parallel, then scores each clip's CLAP audio<->caption cosine
# similarity (higher = better prompt adherence) and runs a paired Wilcoxon test.
# Turns "XL sounds a bit better" into a reproducible number with significance.
#
# CLAP measures semantic/genre adherence, NOT raw fidelity — pair it with the
# spectral check for a fuller picture. Same seed => LM CoT identical => the DiT
# is the only variable.
#
# Usage:
#   IMAGE=<acr>/reputationly/acestep:arm64-a100-latest \
#   CKPT_DIR=/nfs-models/.../ACE-Step-1.5 \
#   DIT_A=acestep-v15-turbo DIT_B=acestep-v15-xl-turbo \
#   GPU_A=0 GPU_B=1 N=10 \
#   bash scripts/smoke/ab_clap.sh
# =============================================================================
set -uo pipefail

IMAGE="${IMAGE:?set IMAGE}"
CKPT_DIR="${CKPT_DIR:?set CKPT_DIR}"
DIT_A="${DIT_A:-acestep-v15-turbo}"
DIT_B="${DIT_B:-acestep-v15-xl-turbo}"
LM_MODEL="${LM_MODEL:-acestep-5Hz-lm-4B}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
PORT_A="${PORT_A:-8001}"
PORT_B="${PORT_B:-8002}"
N="${N:-10}"                       # number of prompts (max = built-in list size)
DURATION="${DURATION:-30}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CLAP_MODEL="${CLAP_MODEL:-laion/larger_clap_music_and_speech}"
HF_CACHE="${HF_CACHE:-/root/.cache/huggingface}"   # persist CLAP download
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
JOB_TIMEOUT="${JOB_TIMEOUT:-1200}"

OUT_DIR="${OUT_DIR:-/root/ab_clap_$(date -u +%Y%m%d-%H%M%S 2>/dev/null || echo run)}"
mkdir -p "$OUT_DIR/A" "$OUT_DIR/B" "$HF_CACHE"
log(){ printf '\033[36m[ab]\033[0m %s\n' "$*"; }

CA="ab-a-$GPU_A"; CB="ab-b-$GPU_B"
cleanup(){ docker rm -f "$CA" "$CB" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# --- prompt set (caption drives CLAP text; lyrics decide vocal/instrumental) ---
python3 - "$OUT_DIR/prompts.json" "$N" <<'PY'
import json, sys
P=[
 {"id":"pop_rock","caption":"an explosive high-energy pop-rock anthem with punchy drums, distorted guitars and a powerful male lead vocal","lyrics":"[Verse]\nRise up through the fire tonight\n[Chorus]\nWe are the ones who never fall","language":"en"},
 {"id":"lofi","caption":"a mellow lo-fi hip-hop instrumental with warm vinyl crackle, jazzy chords and a laid-back boom-bap beat","lyrics":"[inst]","language":"en"},
 {"id":"ballad","caption":"an emotional piano ballad with a clear female lead vocal and soft strings","lyrics":"[Verse]\nQuiet rain against the glass\n[Chorus]\nHold me while the moment lasts","language":"en"},
 {"id":"edm","caption":"a festival EDM track with a big supersaw drop, four-on-the-floor kick and euphoric synth leads","lyrics":"[inst]","language":"en"},
 {"id":"folk","caption":"a gentle acoustic folk song with fingerpicked guitar and a warm male vocal","lyrics":"[Verse]\nDown the road where rivers bend\n[Chorus]\nHome is where the willows bend","language":"en"},
 {"id":"metal","caption":"an aggressive heavy metal instrumental with fast palm-muted riffs, double-kick drums and a shredding guitar solo","lyrics":"[inst]","language":"en"},
 {"id":"jazz","caption":"a smooth jazz trio with brushed drums, upright bass and expressive piano improvisation","lyrics":"[inst]","language":"en"},
 {"id":"trap","caption":"a hard-hitting trap beat with booming 808 bass, crisp hi-hats and a confident male rap vocal","lyrics":"[Verse]\nStack it up I never fold\n[Hook]\nWe run this city cold","language":"en"},
 {"id":"cinematic","caption":"an epic cinematic orchestral piece with soaring strings, brass fanfares and thunderous percussion","lyrics":"[inst]","language":"en"},
 {"id":"synthwave","caption":"a retro 80s synthwave track with nostalgic analog synths, gated reverb drums and a driving bassline","lyrics":"[inst]","language":"en"},
]
n=min(int(sys.argv[2]), len(P))
json.dump(P[:n], open(sys.argv[1],"w"), ensure_ascii=False)
print(f"wrote {n} prompts")
PY

# --- start both DiT servers on separate GPUs ---
start(){ # name gpu port dit
  docker rm -f "$1" >/dev/null 2>&1 || true
  docker run -d --name "$1" --gpus "\"device=$2\"" --network host --memory=120g \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e ACESTEP_CONFIG_PATH="$4" -e ACESTEP_LM_MODEL_PATH="$LM_MODEL" \
    -v "$CKPT_DIR:/app/checkpoints" \
    "$IMAGE" python3 -m uvicorn acestep.api_server:app --host 0.0.0.0 --port "$3" --workers 1 >/dev/null
}
log "starting A=$DIT_A (gpu$GPU_A:$PORT_A) + B=$DIT_B (gpu$GPU_B:$PORT_B)"
start "$CA" "$GPU_A" "$PORT_A" "$DIT_A"
start "$CB" "$GPU_B" "$PORT_B" "$DIT_B"

wait_health(){ local u="$1"; local t0; t0=$(date +%s)
  until curl -sf "$u/health" >/dev/null 2>&1; do
    [ $(( $(date +%s)-t0 )) -ge "$HEALTH_TIMEOUT" ] && { echo "health timeout $u"; exit 3; }; sleep 3; done; }
wait_health "http://127.0.0.1:$PORT_A"; wait_health "http://127.0.0.1:$PORT_B"
log "both servers healthy; generating (first request per server loads models ~2min)"

# --- generate: submit all prompts to both servers, poll, download ---
URL_A="http://127.0.0.1:$PORT_A" URL_B="http://127.0.0.1:$PORT_B" \
OUT_DIR="$OUT_DIR" DURATION="$DURATION" LM_MODEL="$LM_MODEL" JOB_TIMEOUT="$JOB_TIMEOUT" python3 <<'PY'
import json, os, time, urllib.request
ua=os.environ["URL_A"]; ub=os.environ["URL_B"]; out=os.environ["OUT_DIR"]
dur=float(os.environ["DURATION"]); lm=os.environ["LM_MODEL"]; to=int(os.environ["JOB_TIMEOUT"])
prompts=json.load(open(f"{out}/prompts.json"))
def post(base,path,obj):
    r=urllib.request.Request(base+path,data=json.dumps(obj).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=120) as x: return json.load(x)
def submit(base,pr,seed):
    body={"prompt":pr["caption"],"lyrics":pr["lyrics"],"task_type":"text2music","audio_duration":dur,
          "inference_steps":8,"lm_model_path":lm,"lm_backend":"pt","audio_format":"mp3",
          "vocal_language":pr["language"],"use_random_seed":False,"seed":seed,"batch_size":1}
    return post(base,"/release_task",body)["data"]["task_id"]
jobs=[]  # (server_letter, base, prompt_id, task_id)
for i,pr in enumerate(prompts):
    seed=1000+i
    jobs.append(("A",ua,pr["id"],submit(ua,pr,seed)))
    jobs.append(("B",ub,pr["id"],submit(ub,pr,seed)))
print(f"submitted {len(jobs)} jobs across 2 servers")
pending={(s,pid):(base,tid) for s,base,pid,tid in jobs}; t0=time.time()
while pending:
    if time.time()-t0>to: print("TIMEOUT, remaining:",list(pending)); break
    time.sleep(5)
    for key in list(pending):
        base,tid=pending[key]; s,pid=key
        try: q=post(base,"/query_result",{"task_id_list":json.dumps([tid])})
        except Exception: continue
        it=(q.get("data") or [None])[0]
        if not it: continue
        st=it.get("status")
        if st==1:
            f=json.loads(it.get("result") or "[]")[0].get("file")
            urllib.request.urlretrieve(base+f, f"{out}/{s}/{pid}.mp3")
            print(f"  done {s}/{pid}"); del pending[key]
        elif st==2:
            print(f"  FAILED {s}/{pid}"); del pending[key]
print("generation complete")
PY

cleanup   # free the GPUs before CLAP scoring

# --- CLAP scoring (online: downloads CLAP once to HF_CACHE) ---
cat > "$OUT_DIR/clap_score.py" <<'PY'
import json, os, subprocess, numpy as np, torch
from transformers import ClapModel, ClapProcessor
from scipy.stats import wilcoxon
mid=os.environ.get("CLAP_MODEL","laion/larger_clap_music_and_speech")
dev="cuda" if torch.cuda.is_available() else "cpu"
proc=ClapProcessor.from_pretrained(mid); model=ClapModel.from_pretrained(mid).to(dev).eval()
def load(p,sr=48000):
    o=subprocess.run(["ffmpeg","-v","error","-i",p,"-ac","1","-ar",str(sr),"-f","f32le","-"],capture_output=True).stdout
    return np.frombuffer(o,dtype=np.float32).copy()
def sim(text,path):
    t=proc(text=[text],return_tensors="pt",padding=True).to(dev)
    a=proc(audios=[load(path)],sampling_rate=48000,return_tensors="pt").to(dev)
    with torch.no_grad():
        te=model.get_text_features(**t); ae=model.get_audio_features(**a)
    return torch.nn.functional.cosine_similarity(te,ae).item()
prompts=json.load(open("/work/prompts.json")); A=[];B=[]
print(f"{'prompt':12s}  {'A':>7s}  {'B':>7s}  {'Δ(B-A)':>8s}")
for pr in prompts:
    i=pr["id"]
    try: a=sim(pr["caption"],f"/work/A/{i}.mp3"); b=sim(pr["caption"],f"/work/B/{i}.mp3")
    except Exception as e: print(f"{i:12s}  skip ({e})"); continue
    A.append(a);B.append(b); print(f"{i:12s}  {a:7.4f}  {b:7.4f}  {b-a:+8.4f}")
A=np.array(A);B=np.array(B)
print(f"\nA({os.environ['DIT_A']}) mean={A.mean():.4f}  B({os.environ['DIT_B']}) mean={B.mean():.4f}  Δ(B-A)={B.mean()-A.mean():+.4f}")
if len(A)>=6 and np.any(A!=B):
    w=wilcoxon(B,A); wins=int((B>A).sum())
    print(f"paired Wilcoxon p={w.pvalue:.4f}  B>A in {wins}/{len(A)}")
    v=("B(%s) 显著更优"%os.environ['DIT_B']) if (w.pvalue<0.05 and B.mean()>A.mean()) else \
      ("A(%s) 显著更优"%os.environ['DIT_A']) if (w.pvalue<0.05 and A.mean()>B.mean()) else "无显著差异(用便宜的那个)"
    print("verdict:", v)
else:
    print("样本太少(<6)不做显著性;看均值差方向")
PY

log "scoring with CLAP ($CLAP_MODEL) — downloads ~2G once to $HF_CACHE"
docker run --rm --gpus '"device='"$GPU_A"'"' --network host \
  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  -e HF_ENDPOINT="$HF_ENDPOINT" -e HF_HUB_ENABLE_HF_TRANSFER=0 -e HF_HUB_DISABLE_XET=1 \
  -e CLAP_MODEL="$CLAP_MODEL" -e DIT_A="$DIT_A" -e DIT_B="$DIT_B" \
  -v "$OUT_DIR:/work" -v "$HF_CACHE:/root/.cache/huggingface" \
  "$IMAGE" python3 /work/clap_score.py | tee "$OUT_DIR/clap_report.txt"

log "report: $OUT_DIR/clap_report.txt ; audio in $OUT_DIR/{A,B}"
