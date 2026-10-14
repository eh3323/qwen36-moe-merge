"""Model loading + fast-path kernel wiring for Qwen3.6-35B-A3B (qwen3_5_moe) LoRA training.

Fast paths (each verified at startup and logged into metadata):
  * MoE experts: transformers `experts_implementation="grouped_mm"` (torch.nn.functional.grouped_mm over the
    fused 3-D expert tensors; no Python loop over 256 experts). Experts are frozen -> only dX is computed.
  * Gated DeltaNet (30/40 layers): FLA Triton `chunk_gated_delta_rule` (varlen via cu_seqlens) and FLA Triton
    `causal_conv1d` (varlen) patched into the HF module. The HF reference torch path is >10x slower and its
    torch conv1d fallback would leak 3 tokens across packed samples.
  * Full attention (10/40 layers, head_dim 256, GQA 16/2): custom attention fn registered as "apex_varlen":
    flash-attn varlen (pip flash_attn or kernels-hub FA3/FA2) if it passes a numeric self-test, else exact
    per-sample SDPA (flash/efficient/cuDNN backends only; MATH disabled so we never fall into O(T^2) memory).
  * Loss: chunked + checkpointed lm_head/CE over assistant positions only (never materializes T x 248K logits).
"""
from __future__ import annotations

import math
import re
from typing import Callable

import torch
import torch.nn.functional as F

ATTN_NAME = "apex_varlen"
_FA_FUNC: Callable | None = None
_FA_NAME: str = "sdpa_segments"
KERNEL_INFO: dict = {}


# ============================================================================ attention
def _sdpa_segments(query, key, value, cu_list, scaling):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)
    outs = []
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        for s, e in zip(cu_list[:-1], cu_list[1:]):
            outs.append(F.scaled_dot_product_attention(
                query[:, :, s:e], key[:, :, s:e], value[:, :, s:e], is_causal=True, scale=scaling))
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
    return out.transpose(1, 2).contiguous()  # [B, T, H, D]


def apex_varlen_attention(module, query, key, value, attention_mask, dropout: float = 0.0,
                          scaling: float | None = None, **kwargs):
    """query [B,H,T,D], key/value [B,Hkv,T,D] -> ([B,T,H,D], None). Packed samples isolated via cu_seqlens."""
    B, H, T, D = query.shape
    cu = kwargs.get("cu_seq_lens_q")
    cu_cpu = kwargs.get("cu_seqlens_cpu")
    if cu is None:
        cu = torch.tensor([0, T], dtype=torch.int32, device=query.device)
        cu_cpu = torch.tensor([0, T])
        maxlen = T
    else:
        assert B == 1, "packed batches must have batch size 1"
        maxlen = int(kwargs.get("max_length_q") or T)
    if _FA_FUNC is not None:
        q = query.transpose(1, 2).reshape(B * T, H, D)
        k = key.transpose(1, 2).reshape(B * T, key.shape[1], D)
        v = value.transpose(1, 2).reshape(B * T, value.shape[1], D)
        out = _FA_FUNC(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=maxlen, max_seqlen_k=maxlen,
                       softmax_scale=scaling, causal=True)
        if isinstance(out, tuple):
            out = out[0]
        return out.view(B, T, H, D), None
    cu_list = cu_cpu.tolist() if cu_cpu is not None else cu.tolist()
    return _sdpa_segments(query, key, value, cu_list, scaling), None


def _fa_candidates(pref: str):
    cands = []
    if pref in ("auto", "flash_attn"):
        def _pip():
            from flash_attn import flash_attn_varlen_func
            return flash_attn_varlen_func
        cands.append(("flash_attn(pip)", _pip))
    if pref in ("auto", "hub_fa3"):
        def _fa3():
            from kernels import get_kernel
            return get_kernel("kernels-community/flash-attn3", version=1).flash_attn_varlen_func
        cands.append(("hub:flash-attn3", _fa3))
    if pref in ("auto", "hub_fa2"):
        def _fa2():
            from kernels import get_kernel
            return get_kernel("kernels-community/flash-attn2", version=2).flash_attn_varlen_func
        cands.append(("hub:flash-attn2", _fa2))
    return cands


def _selftest_fa(fn, device) -> float:
    """Compare FA varlen vs SDPA-segments on a small packed GQA problem (hd=256), fwd+bwd."""
    torch.manual_seed(0)
    T, H, Hkv, D = 384, 16, 2, 256
    cu_list = [0, 100, 384]
    q = torch.randn(1, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, Hkv, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, Hkv, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    ref = _sdpa_segments(q, k, v, cu_list, D ** -0.5)
    cu = torch.tensor(cu_list, dtype=torch.int32, device=device)
    out = fn(q.transpose(1, 2).reshape(T, H, D), k.transpose(1, 2).reshape(T, Hkv, D),
             v.transpose(1, 2).reshape(T, Hkv, D), cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=284,
             max_seqlen_k=284, softmax_scale=D ** -0.5, causal=True)
    if isinstance(out, tuple):
        out = out[0]
    out = out.view(1, T, H, D)
    out.float().sum().backward()
    err = (out.float() - ref.float()).abs().max().item()
    if not math.isfinite(err) or q.grad is None or not torch.isfinite(q.grad).all():
        return float("inf")
    return err


def setup_attention(pref: str, device, log=print):
    global _FA_FUNC, _FA_NAME
    from transformers import AttentionInterface

    AttentionInterface.register(ATTN_NAME, apex_varlen_attention)
    try:  # never let HF build a dense 4D mask for our impl
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, flash_attention_mask

        ALL_MASK_ATTENTION_FUNCTIONS.register(ATTN_NAME, flash_attention_mask)
    except Exception:
        pass
    _FA_FUNC, _FA_NAME = None, "sdpa_segments"
    if pref != "sdpa":
        for name, loader in _fa_candidates(pref):
            try:
                fn = loader()
                err = _selftest_fa(fn, device)
                if err < 5e-2:
                    _FA_FUNC, _FA_NAME = fn, name
                    log(f"[attn] using {name} (selftest max|diff|={err:.2e})")
                    break
                log(f"[attn] {name} failed selftest (err={err})")
            except Exception as e:
                log(f"[attn] {name} unavailable: {type(e).__name__}: {str(e)[:200]}")
    if _FA_FUNC is None:
        log("[attn] using exact per-sample SDPA (flash/cudnn/efficient backends)")
    KERNEL_INFO["attention"] = _FA_NAME


# ============================================================================ Gated DeltaNet
def patch_gated_deltanet(backend: str = "fla", log=print):
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M

    if backend == "torch":
        log("[gdn] WARNING: using HF reference torch GDN path (very slow, conv leaks across packed samples)")
        KERNEL_INFO["gdn"] = "torch_reference"
        return
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk
    try:
        from fla.modules.conv.causal_conv1d import causal_conv1d as fla_conv
    except ImportError:
        from fla.modules import causal_conv1d as fla_conv  # older layouts

    def chunk_fn(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
                 use_qk_l2norm_in_kernel=False, cu_seqlens=None, cu_seqlens_cpu=None, **_):
        return fla_chunk(query, key, value, g=g, beta=beta, initial_state=initial_state,
                         output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                         cu_seqlens=None if cu_seqlens is None else cu_seqlens.long(),
                         cu_seqlens_cpu=None if cu_seqlens_cpu is None else cu_seqlens_cpu.long())

    def conv_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        # HF layout [B, D, T] -> FLA layout [B, T, D]
        cu = kwargs.get("cu_seq_lens_q")
        cu_cpu = kwargs.get("cu_seqlens_cpu")
        x = hidden_states.transpose(1, 2).contiguous()
        y = fla_conv(x, weight=weight, bias=bias, activation=activation,
                     cu_seqlens=None if cu is None else cu.long(),
                     cu_seqlens_cpu=None if cu_cpu is None else cu_cpu.long())
        if isinstance(y, tuple):
            y = y[0]
        return y.transpose(1, 2)

    M.torch_chunk_gated_delta_rule = chunk_fn
    M.causal_conv1d_fn = conv_fn
    KERNEL_INFO["gdn"] = "fla_triton(chunk_gated_delta_rule+causal_conv1d, varlen)"
    log("[gdn] patched HF Qwen3.5-MoE GatedDeltaNet -> FLA Triton kernels (varlen)")


def selftest_gdn(device, log=print) -> dict:
    """Packed-vs-separate equivalence test of patched GDN pieces (catches cross-sample leakage)."""
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M

    torch.manual_seed(0)
    T1, T2, H, K, V, Dc = 80, 120, 4, 128, 128, 64
    T = T1 + T2
    q = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, V, device=device, dtype=torch.bfloat16)
    g = -torch.rand(1, T, H, device=device, dtype=torch.float32)
    beta = torch.rand(1, T, H, device=device, dtype=torch.bfloat16)
    cu = torch.tensor([0, T1, T], dtype=torch.int32, device=device)
    o_packed, _ = M.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True,
                                                 cu_seqlens=cu, cu_seqlens_cpu=cu.cpu())
    o2, _ = M.torch_chunk_gated_delta_rule(q[:, T1:], k[:, T1:], v[:, T1:], g=g[:, T1:], beta=beta[:, T1:],
                                           use_qk_l2norm_in_kernel=True)
    err_gdn = (o_packed[:, T1:].float() - o2.float()).abs().max().item()
    x = torch.randn(1, Dc, T, device=device, dtype=torch.bfloat16)
    w = torch.randn(Dc, 4, device=device, dtype=torch.bfloat16)
    y_p = M.causal_conv1d_fn(x, w, None, activation="silu", cu_seq_lens_q=cu, cu_seqlens_cpu=cu.cpu())
    y_2 = M.causal_conv1d_fn(x[:, :, T1:].contiguous(), w, None, activation="silu")
    err_conv = (y_p[:, :, T1:].float() - y_2.float()).abs().max().item()
    res = {"gdn_packed_vs_separate_maxdiff": err_gdn, "conv_packed_vs_separate_maxdiff": err_conv}
    log(f"[gdn] selftest {res}")
    if err_gdn > 5e-2 or err_conv > 5e-2:
        raise RuntimeError(f"GDN varlen selftest failed: {res}")
    return res


# ============================================================================ model + LoRA
def build_target_regex(lcfg: dict) -> str:
    full = "|".join(re.escape(m) for m in lcfg.get("full_attn_targets", []))
    lin = "|".join(re.escape(m) for m in lcfg.get("linear_attn_targets", []))
    mlp = "|".join(re.escape(m) for m in lcfg.get("mlp_targets", []))
    alts = []
    if full:
        alts.append(rf"self_attn\.({full})")
    if lin:
        alts.append(rf"linear_attn\.({lin})")
    if mlp:
        alts.append(rf"mlp\.({mlp})")
    # anchored to the text decoder: excludes vision tower, MTP head, router (mlp.gate), routed experts
    return rf".*language_model\.layers\.\d+\.({'|'.join(alts)})"


def load_model(cfg: dict, device, log=print):
    import transformers

    mcfg = cfg["model"]
    dtype = getattr(torch, mcfg.get("dtype", "bfloat16"))
    cls = getattr(transformers, mcfg.get("class", "Qwen3_5MoeForConditionalGeneration"))
    kwargs = dict(dtype=dtype, device_map={"": device}, attn_implementation=ATTN_NAME)
    if mcfg.get("revision_check"):
        pass
    exp_impl = mcfg.get("experts_implementation", "grouped_mm")
    try:
        model = cls.from_pretrained(mcfg["path"], experts_implementation=exp_impl, **kwargs)
    except TypeError:
        log("[model] experts_implementation kwarg unsupported; setting on config after load")
        model = cls.from_pretrained(mcfg["path"], **kwargs)
    for c in (model.config, getattr(model.config, "text_config", None)):
        if c is not None:
            try:
                c._experts_implementation = exp_impl
            except Exception:
                pass
            c.use_cache = False
    KERNEL_INFO["experts"] = getattr(model.config.text_config, "_experts_implementation", exp_impl)
    KERNEL_INFO["attn_implementation_config"] = getattr(model.config.text_config, "_attn_implementation", None)
    log(f"[model] loaded {cls.__name__} experts_impl={KERNEL_INFO['experts']} "
        f"attn_impl={KERNEL_INFO['attn_implementation_config']}")
    if mcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def apply_lora(model, lcfg: dict, log=print):
    from peft import LoraConfig, get_peft_model

    regex = build_target_regex(lcfg)
    pcfg = LoraConfig(
        r=int(lcfg["r"]), lora_alpha=float(lcfg["alpha"]), lora_dropout=float(lcfg.get("dropout", 0.0)),
        target_modules=regex, bias="none", use_rslora=bool(lcfg.get("use_rslora", False)),
        init_lora_weights=True,
    )
    model = get_peft_model(model, pcfg)
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()  # fp32 master weights; compute runs under bf16 autocast
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    names = sorted({n.rsplit(".lora_", 1)[0] for n, _ in model.named_parameters() if "lora_A" in n})
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[lora] regex={regex}\n[lora] {len(names)} target modules, {n_tr/1e6:.2f}M trainable params")
    return model, names, regex


def get_text_parts(peft_model):
    base = peft_model.base_model.model  # Qwen3_5MoeForConditionalGeneration
    return base.model.language_model, base.lm_head


# ============================================================================ loss
def _ce_chunk(h, y, w):
    logits = F.linear(h, w).float()
    return F.cross_entropy(logits, y, reduction="sum")


def chunked_ce_sum(hidden: torch.Tensor, labels: torch.Tensor, lm_head_weight: torch.Tensor,
                   chunk: int = 4096) -> tuple[torch.Tensor, int]:
    """hidden [T,H], labels [T] (-100 ignore). Returns (sum CE over targets, #targets)."""
    sel = labels >= 0
    idx = sel.nonzero(as_tuple=True)[0]
    n = idx.numel()
    if n == 0:
        return hidden.sum() * 0.0, 0
    h = hidden.index_select(0, idx)
    y = labels.index_select(0, idx)
    total = hidden.new_zeros((), dtype=torch.float32)
    from torch.utils.checkpoint import checkpoint

    for s in range(0, n, chunk):
        total = total + checkpoint(_ce_chunk, h[s : s + chunk], y[s : s + chunk], lm_head_weight,
                                   use_reentrant=False)
    return total, n


def forward_pack(text_model, lm_head, input_ids, labels, position_ids, cu, cu_cpu, ce_chunk: int):
    maxlen = int((cu_cpu[1:] - cu_cpu[:-1]).max())
    out = text_model(
        input_ids=input_ids[None],
        position_ids=position_ids[None],
        attention_mask={"full_attention": None, "linear_attention": None},
        use_cache=False,
        cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=maxlen, max_length_k=maxlen,
        cu_seqlens_cpu=cu_cpu,
    )
    hidden = out.last_hidden_state[0]
    return chunked_ce_sum(hidden, labels, lm_head.weight, ce_chunk)
