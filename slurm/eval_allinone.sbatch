#!/bin/bash
#SBATCH --job-name=apex-eval
#SBATCH --account=torch_pr_147_courant
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|b200
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --output=/scratch/eh3323/apex/logs/eval/%x-%j.out
# DEFAULT eval path: ONE allocation (single task, N GPUs, 16 CPU/GPU) runs vLLM (+GPU keepalive,
# via slurm/serve_model.sbatch in the background) and the benchmark runners on the same node
# against that local server. Benchmarks in BENCHES run concurrently; each lane is the normal
# runner (slurm/eval_<bench>.sbatch executed as a script), so outputs/resume are identical to
# the separate serve + CPU-job path.
#
#   sbatch --job-name=eval-base-hle-swe --export=ALL,RUN=base,MODEL_DIR=$APEX/models/Qwen3.6-35B-A3B,BENCHES=hle+swepro slurm/eval_allinone.sbatch
#   sbatch --job-name=eval-base-tb2 --gres=gpu:4 --cpus-per-task=64 --mem=600G \
#     --export=ALL,RUN=base,MODEL_DIR=...,BENCHES=tb2,TB2_ATTEMPTS="1 2 3 4 5" slurm/eval_allinone.sbatch
#   smoke: add SMOKE=1 (each lane runs its smoke subset + net probe)
#
# List values are '+'-separated inside --export (sbatch splits --export on commas: BENCHES=hle,asi
# would silently pass only 'hle'); ',' still works when the variable is exported in the shell instead.
# Parameters (env): RUN, MODEL_DIR (required); BENCHES (default tb2+swepro+asi+hle); SMOKE;
#   TB2_ATTEMPTS ("1"; run sequentially), SWE_REPS ("1"; e.g. 1+2+3, sequential), ASI_LEVELS (b1+b2+b3+b4; one asibench invocation),
#   concurrency overrides TB2_N, SWE_N, ASI_PAR, HLE_N; TB2_MEM_PER_TRIAL_GB (48, used only for sizing).
# Default sizing: RAM left after a vLLM reserve (30 GB/GPU + 20 GB) is split evenly across the
# sandbox lanes present (tb2/swepro/asi); TB2 trials are budgeted at their nominal 48 GB (max 10),
# SWE tasks at 4 GB (their task memory cap; max 51, <= 2x CPUs), ASI runs at 8 GB (<= CPUs/2, max 16).
# CPU is NOT exclusively reserved per trial (singularity cannot enforce it); TB2 keeps its nominal
# 32 CPU / 48 GB override as configured.
set -uo pipefail
source /scratch/eh3323/apex/env.sh
: "${RUN:?}" "${MODEL_DIR:?}"
REPO=$APEX/repo
BENCHES=${BENCHES:-tb2,swepro,asi,hle}; BENCHES=${BENCHES//+/,}; SMOKE=${SMOKE:-0}
TB2_ATTEMPTS=${TB2_ATTEMPTS:-1}; TB2_ATTEMPTS=${TB2_ATTEMPTS//+/ }; SWE_REPS=${SWE_REPS:-1}; SWE_REPS=${SWE_REPS//+/ }; ASI_LEVELS=${ASI_LEVELS:-b1,b2,b3,b4}; ASI_LEVELS=${ASI_LEVELS//+/,}
NGPU=$(nvidia-smi -L | wc -l); CPUS=${SLURM_CPUS_PER_TASK:-16}; MEM_GB=$(( ${SLURM_MEM_PER_NODE:-163840} / 1024 ))
export ENDPOINT_NAME="${RUN}@${SLURM_JOB_ID}"   # private endpoint for this allocation
AIO_LOG=$APEX/logs/allinone-${RUN}-${SLURM_JOB_ID}; mkdir -p "$AIO_LOG"

has() { [[ ",$BENCHES," == *",$1,"* ]]; }
clamp() { local v=$1 lo=$2 hi=$3; (( v < lo )) && v=$lo; (( v > hi )) && v=$hi; echo $v; }
nlanes=0; for b in tb2 swepro asi; do has $b && nlanes=$((nlanes+1)); done
AVAIL=$(( MEM_GB - 30*NGPU - 20 )); SHARE=$(( nlanes > 0 ? AVAIL / nlanes : AVAIL ))
TB2_N=${TB2_N:-$(clamp $(( SHARE / ${TB2_MEM_PER_TRIAL_GB:-48} )) 1 10)}
SWE_N=${SWE_N:-$(clamp $(( SHARE / 4 < 2*CPUS ? SHARE / 4 : 2*CPUS )) 1 51)}
ASI_PAR=${ASI_PAR:-$(clamp $(( SHARE / 8 < CPUS / 2 ? SHARE / 8 : CPUS / 2 )) 1 16)}
HLE_N=${HLE_N:-$(( nlanes > 0 ? 128 * NGPU : 256 * NGPU ))}
echo "[aio] host=$(hostname) ngpu=$NGPU cpus=$CPUS mem=${MEM_GB}G benches=$BENCHES smoke=$SMOKE"
echo "[aio] sizing: TB2_N=$TB2_N SWE_N=$SWE_N ASI_PAR=$ASI_PAR HLE_N=$HLE_N (share ${SHARE}G/lane)"
echo "{\"job\":\"$SLURM_JOB_ID\",\"benches\":\"$BENCHES\",\"ngpu\":$NGPU,\"cpus\":$CPUS,\"mem_gb\":$MEM_GB,\"tb2_n\":$TB2_N,\"swe_n\":$SWE_N,\"asi_par\":$ASI_PAR,\"hle_n\":$HLE_N,\"tb2_attempts\":\"$TB2_ATTEMPTS\"}" > "$AIO_LOG/sizing.json"

# 1) vLLM + keepalive (writes $APEX/work/endpoints/$ENDPOINT_NAME.env when healthy)
bash $REPO/slurm/serve_model.sbatch > "$AIO_LOG/serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill $LANE_PIDS 2>/dev/null; kill $SERVE_PID 2>/dev/null; wait $SERVE_PID 2>/dev/null' EXIT
trap 'echo "[aio] SIGTERM"; exit 143' TERM
export EP_WAIT_SEC=3600

# 2) benchmark lanes (each is the normal runner, run as a plain script inside this allocation)
LANE_PIDS=""
lane() {  # <name> <cmd...>
  local name=$1; shift
  ( "$@" ) > "$AIO_LOG/$name.log" 2>&1 &
  LANE_PIDS="$LANE_PIDS $!"; echo "[aio] lane $name pid $! -> $AIO_LOG/$name.log"
}
if has tb2; then
  lane tb2 bash -c 'for a in '"$TB2_ATTEMPTS"'; do RUN='"$RUN"' ATTEMPT=$a SMOKE='"$SMOKE"' N_CONCURRENT='"$TB2_N"' bash '"$REPO"'/slurm/eval_tb2.sbatch; [ "'"$SMOKE"'" = 1 ] && break; done'
fi
if has swepro; then
  lane swepro bash -c 'for r in '"$SWE_REPS"'; do RUN='"$RUN"' REP=$r SMOKE='"$SMOKE"' N_CONCURRENT='"$SWE_N"' bash '"$REPO"'/slurm/eval_swepro.sbatch; [ "'"$SMOKE"'" = 1 ] && break; done'
fi
has asi    && lane asi env RUN=$RUN SMOKE=$SMOKE LEVELS=$ASI_LEVELS PARALLEL=$ASI_PAR bash $REPO/slurm/eval_asi.sbatch
has hle    && lane hle env RUN=$RUN SMOKE=$SMOKE HLE_N=$HLE_N bash $REPO/slurm/eval_hle.sbatch

rc=0
for p in $LANE_PIDS; do
  while kill -0 $p 2>/dev/null; do
    kill -0 $SERVE_PID 2>/dev/null || { echo "[aio] vLLM serve process died; see $AIO_LOG/serve.log"; rc=1; break 2; }
    sleep 60
  done
  wait $p || rc=1
done
echo "[aio] lanes finished rc=$rc"; grep -h "\[eval\] done\|generation done\|agent phase done" "$AIO_LOG"/*.log
exit $rc
