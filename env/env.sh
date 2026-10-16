# source /scratch/eh3323/apex/env.sh  -- 所有缓存放 /scratch，避免 $HOME 文件数配额(30k,已用96%)
export APEX=/scratch/eh3323/apex
export HF_HOME=/scratch/eh3323/.huggingface          # 沿用已有 HF 登录
export HF_HUB_CACHE=$APEX/cache/hf
export HF_XET_HIGH_PERFORMANCE=1
export UV_CACHE_DIR=$APEX/cache/uv
export PIP_CACHE_DIR=$APEX/cache/pip
export XDG_CACHE_HOME=$APEX/cache/xdg
export APPTAINER_CACHEDIR=$APEX/cache/apptainer
export TMPDIR=$APEX/cache/tmp
export PATH=$HOME/.local/bin:$PATH
# 只用 uv 自带的 Python（带头文件，Triton 编译 JIT 时要用；系统 python3.12 没有 Python.h）
export UV_PYTHON_INSTALL_DIR=$APEX/cache/uv-python
export UV_PYTHON_PREFERENCE=only-managed
# apptainer 构建镜像的临时目录必须放在本地盘：/scratch 是 NFS，解包时会因 nfs4_dacl xattr 报错
export APPTAINER_TMPDIR=/tmp/$USER-apptainer; mkdir -p $APPTAINER_TMPDIR
export FLASHINFER_WORKSPACE_BASE=$APEX/cache   # flashinfer JIT cache -> /scratch
# The JIT/data dirs below default to $HOME (30k file quota): move them all to /scratch
export TRITON_CACHE_DIR=$APEX/cache/triton      # Triton kernel cache (default ~/.triton; 1,053 new files in the last 24h)
export CUDA_CACHE_PATH=$APEX/cache/nv           # CUDA JIT cache (default ~/.nv)
export XDG_DATA_HOME=$APEX/cache/xdg-data       # opencode session data (default ~/.local/share); 12 parallel ASI agents would quickly fill the quota
export XDG_STATE_HOME=$APEX/cache/xdg-state
