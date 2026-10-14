#!/bin/bash
# FIRST JOB TO RUN: 25-step throughput / peak-memory probe on synthetic packed data (no real data needed).
#   cases: (a) 1 GPU @32K   (b) 4 GPUs @32K   (c) 1 GPU @64K (E-SWE/mixed sizing)   [+ (d) SDPA vs FA if COMPARE_ATTN=1]
# Submit (login node):  cd $APEX/repo && sbatch slurm/train_throughput_test.sbatch
# Results: grep THROUGHPUT_SUMMARY in the .out file; table at the end; per-step metrics in
#          $APEX/runs/_throughput/<case>_<jobid>/logs/metrics.jsonl
#SBATCH --job-name=apex-lora-tp
#SBATCH --account=torch_pr_147_courant
#SBATCH --constraint=h200|b200
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1   # Torch 要求：多任务作业必须声明每任务 GPU 数，否则报 "Requested node configuration is not available"
#SBATCH --cpus-per-task=16
#SBATCH --mem=600G
#SBATCH --time=02:00:00
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --output=/scratch/eh3323/apex/logs/train/%x-%j.out

set -uo pipefail
source /scratch/eh3323/apex/env.sh
source $APEX/envs/train/bin/activate
cd $APEX/repo
export MASTER_ADDR=127.0.0.1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/$USER-triton-$SLURM_JOB_ID
mkdir -p $APEX/logs/train
hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch,transformers,peft,fla;print('torch',torch.__version__,torch.version.cuda,'tf',transformers.__version__,'peft',peft.__version__,'fla',fla.__version__)"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -l 10 > $APEX/logs/train/gpuutil-$SLURM_JOB_ID.csv &
SMI=$!

CFG=configs/train/throughput_test.yaml
run_case () {  # name ngpus extra-args...
  local name=$1 n=$2; shift 2
  export MASTER_PORT=$((20000 + (SLURM_JOB_ID + n + ${#name}) % 20000))
  echo "=================== CASE $name (ngpus=$n) $* ==================="
  local t0=$(date +%s)
  srun --ntasks=$n --cpus-per-task=16 --cpu-bind=none --kill-on-bad-exit=1 --export=ALL \
    bash -c 'export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS; \
             exec python -u -m src.train.sft_lora --config '"$CFG"' --set output.run_id='"${name}_${SLURM_JOB_ID}"' '"$*"
  echo "[case $name] rc=$? wall=$(( $(date +%s) - t0 ))s"
}

run_case tp_1gpu_32k 1
# 多卡 case 只在分到足够 GPU 时运行；用 1 卡提交时（更容易排到）自动跳过
[ "${SLURM_NTASKS:-1}" -ge 4 ] && run_case tp_4gpu_32k 4
[ "${SLURM_NTASKS:-1}" -eq 2 ] && run_case tp_2gpu_32k 2
run_case tp_1gpu_64k 1 --set data.max_seq_len=65536 --set train.global_batch_tokens=65536 \
                       --set data.synthetic.max_len=32768 --set train.max_steps=15
if [ "${COMPARE_ATTN:-0}" = "1" ]; then
  run_case tp_1gpu_32k_sdpa 1 --set model.attn_backend=sdpa --set train.max_steps=15
fi
kill $SMI 2>/dev/null

echo "=================== SUMMARY ==================="
python - <<PY
import json,glob,os
rows=[]
for f in sorted(glob.glob(os.path.expandvars("\$APEX/runs/_throughput/*_${SLURM_JOB_ID}/logs/throughput.json"))):
    d=json.load(open(f)); d["case"]=f.split("/")[-3]; rows.append(d)
print(f"{'case':28s} {'gpus':>4s} {'seq':>6s} {'tok/s':>9s} {'tok/s/gpu':>9s} {'peakGB':>7s}  attn / gdn / experts")
for d in rows:
    if not d: continue
    print(f"{d['case']:28s} {d['world_size']:4d} {d['max_seq_len']:6d} {d['median_tokens_per_s']:9.0f} {d['median_tokens_per_s_per_gpu']:9.0f} {d['max_peak_mem_gb']:7.1f}  {d['attention']} / {d['gdn']} / {d['experts']}")
if rows and rows[0]:
    tps=max(r.get('median_tokens_per_s_per_gpu',0) for r in rows if r)
    for tok,name in [(15e6,'one expert epoch (15M tok)'),(60e6,'mixed epoch (60M tok)')]:
        print(f"  {name}: ~{tok/tps/3600:.2f} GPU-hours at {tps:.0f} tok/s/gpu")
PY
