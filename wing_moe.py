#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

# -----------------------------------------------------------------------------
# wing / MoE — MXFP4 two-stage, aiter + FlyDSL stage2.
#
# What I actually cared about when tuning this (not marketing copy):
#   - block_m isn't "one number fits all". E=33 with fat tokens wants different
#     blocking than E=257 where experts are sparse and you bleed work on padding.
#   - ksplit>1 path is the BF16/cktile escape hatch — no activation fp4 quant on
#     that branch; fighting that in code is pointless, the graph already picked it.
#   - The fused quant kernel wants stable output buffers; I cache by (M,N,sorted_len,topk)
#     so I'm not malloc-storming the runner every call.
#   - I patch cfg_2stages once: load CSV if cold, then overlay my rows. Idempotent.
# -----------------------------------------------------------------------------

import os
import torch
import triton
from task import input_t, output_t

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import get_2stage_cfgs, get_padded_M, get_inter_dim
import aiter.fused_moe as _wing_moe_core
import aiter.ops.flydsl.moe_kernels as _wing_fly_registry
from aiter.ops.triton._triton_kernels.quant.fused_mxfp4_quant import (
    _fused_dynamic_mxfp4_quant_moe_sort_kernel,
)

# --- persistent host pools (same lifetime semantics as original v186) ---
_wing_persistent = {}
_wing_q_persistent = {}
_wing_cfg_merged = False


def _wing_merge_aiter_tables():
    """Load default fMOE CSV once, then slam our per-shape overrides on top."""
    global _wing_cfg_merged
    if _wing_cfg_merged:
        return
    _wing_cfg_merged = True

    if _wing_moe_core.cfg_2stages is None:
        import pandas as pd
        from aiter.jit.core import AITER_CONFIGS

        tune_path = AITER_CONFIGS.AITER_CONFIG_FMOE_FILE
        if os.path.exists(tune_path):
            idx = [
                "cu_num", "token", "model_dim", "inter_dim", "expert", "topk",
                "act_type", "dtype", "q_dtype_a", "q_dtype_w", "q_type",
                "use_g1u1", "doweight_stage1",
            ]
            frame = pd.read_csv(tune_path)
            if "_tag" in frame.columns:
                frame = frame[frame["_tag"].fillna("") == ""]
            _wing_moe_core.cfg_2stages = frame.set_index(idx).to_dict("index")
        else:
            _wing_moe_core.cfg_2stages = {}

    # wing: last writer wins on keys — intentional, I trust the rows below more than stale CSV luck.
    _wing_moe_core.cfg_2stages.update(_WING_PER_SHAPE)


def _wing_tune_key(tokens, inter_dim, experts):
    # wing: tuple has to match aiter's index shape exactly or you get silent misses.
    return (
        256, tokens, 7168, inter_dim, experts, 9,
        "ActivationType.Silu", "torch.bfloat16",
        "torch.float4_e2m1fn_x2", "torch.float4_e2m1fn_x2",
        "QuantType.per_1x32", True, False,
    )


_WING_CK_STAGE1 = (
    "moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
)
_WING_FLY_STAGE2 = "flydsl_moe2_afp4_wfp4_bf16_t16x128x128_atomic"

_wing_fly_registry._KERNEL_PARAMS[_WING_FLY_STAGE2] = {
    "stage": 2,
    "a_dtype": "fp4",
    "b_dtype": "fp4",
    "out_dtype": "bf16",
    "tile_m": 16,
    "tile_n": 128,
    "tile_k": 128,
    "mode": "atomic",
    "MPerBlock": 16,
}

# wing: table below is the whole "why this file exists" — numbers are from sweeps, not vibes.
_WING_PER_SHAPE = {
    _wing_tune_key(16, 256, 257): {
        "block_m": 16, "ksplit": 2, "kernelName1": "", "kernelName2": "", "run_1stage": False,
    },
    _wing_tune_key(128, 256, 257): {
        "block_m": 32, "ksplit": 0,
        "kernelName1": _WING_CK_STAGE1, "kernelName2": _WING_FLY_STAGE2, "run_1stage": False,
    },
    _wing_tune_key(512, 256, 257): {
        "block_m": 32, "ksplit": 0,
        "kernelName1": _WING_CK_STAGE1, "kernelName2": _WING_FLY_STAGE2, "run_1stage": False,
    },
    _wing_tune_key(16, 512, 33): {
        "block_m": 32, "ksplit": 2, "kernelName1": "", "kernelName2": "", "run_1stage": False,
    },
    _wing_tune_key(128, 512, 33): {
        "block_m": 32, "ksplit": 0,
        "kernelName1": _WING_CK_STAGE1, "kernelName2": _WING_FLY_STAGE2, "run_1stage": False,
    },
    _wing_tune_key(512, 512, 33): {
        "block_m": 64, "ksplit": 0,
        "kernelName1": _WING_CK_STAGE1, "kernelName2": _WING_FLY_STAGE2, "run_1stage": False,
    },
    _wing_tune_key(512, 2048, 33): {
        "block_m": 32, "ksplit": 0,
        "kernelName1": _WING_CK_STAGE1, "kernelName2": _WING_FLY_STAGE2, "run_1stage": False,
    },
}


def _wing_alloc_sort_workspace(num_tokens, num_experts, top_k, model_dim, block_m, device):
    key = (num_tokens, num_experts, top_k, model_dim, block_m)
    if key not in _wing_persistent:
        max_pad = num_tokens * top_k + num_experts * block_m - top_k
        max_blk = (max_pad + block_m - 1) // block_m
        _wing_persistent[key] = {
            "sid": torch.empty(max_pad, dtype=dtypes.i32, device=device),
            "sw": torch.empty(max_pad, dtype=dtypes.fp32, device=device),
            "se": torch.empty(max_blk, dtype=dtypes.i32, device=device),
            "nv": torch.empty(2, dtype=dtypes.i32, device=device),
            "out": torch.empty((num_tokens, model_dim), dtype=torch.bfloat16, device=device),
            "a2": torch.empty((num_tokens, top_k, 0), dtype=torch.bfloat16, device=device),
        }
    return _wing_persistent[key]


def _wing_alloc_mid_hidden(num_tokens, top_k, inter_dim, device):
    key = ("a2", num_tokens, top_k, inter_dim)
    if key not in _wing_persistent:
        _wing_persistent[key] = torch.empty(
            (num_tokens, top_k, inter_dim), dtype=torch.bfloat16, device=device
        )
    return _wing_persistent[key]


def _wing_run_sorted_quant(x, sorted_ids, num_valid_ids, token_num, top_k, block_m, device):
    # wing: this kernel is ugly-fast; I don't touch the grid math — it's already balanced for our M/N.
    rows, cols = x.shape
    qbs = 32
    blk_mx = 128
    blk_m, blk_n = 32, 8
    blk_m_u32, blk_n_u32 = 16, 4

    scale_n = triton.cdiv(cols, qbs)
    sorted_len = sorted_ids.shape[0]

    qkey = (rows, cols, sorted_len, top_k)
    if qkey not in _wing_q_persistent:
        _wing_q_persistent[qkey] = {
            "fp4": torch.empty((rows, cols // 2), dtype=torch.uint8, device=device),
            "bs": torch.empty(
                (triton.cdiv(sorted_len, blk_m), triton.cdiv(scale_n, blk_n),
                 blk_n_u32, blk_m_u32, 4),
                dtype=torch.uint8,
                device=device,
            ),
        }
    bucket = _wing_q_persistent[qkey]

    num_pid = triton.cdiv(rows, blk_mx) * scale_n + triton.cdiv(sorted_len, blk_m) * triton.cdiv(scale_n, blk_n)

    _fused_dynamic_mxfp4_quant_moe_sort_kernel[(num_pid,)](
        x, bucket["fp4"], sorted_ids, num_valid_ids, bucket["bs"],
        rows, cols, scale_n,
        *x.stride(), *bucket["fp4"].stride(), *bucket["bs"].stride(),
        token_num=token_num, M_i=rows, N_i=scale_n,
        MXFP4_QUANT_BLOCK_SIZE=qbs, BLOCK_SIZE_Mx=blk_mx,
        BLOCK_SIZE_M=blk_m // 2, BLOCK_SIZE_N=blk_n // 2,
        TOPK=top_k,
    )

    return (
        bucket["fp4"].view(dtypes.fp4x2),
        bucket["bs"].view(dtypes.fp8_e8m0).view(-1, scale_n),
    )


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, _w1r, _w2r, _w1sr, _w2sr,
        w1, w2, w1s, w2s,
        topk_weights, topk_ids, config,
    ) = data

    _wing_merge_aiter_tables()

    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    device = hidden_states.device
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    num_experts, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    padded_m = get_padded_M(num_tokens)

    meta = get_2stage_cfgs(
        padded_m, model_dim, inter_dim, num_experts, top_k,
        torch.bfloat16, dtypes.fp4x2, dtypes.fp4x2,
        QuantType.per_1x32, True, ActivationType.Silu,
        False, hidden_pad, intermediate_pad, True,
    )
    block_m = int(meta.block_m)

    workspace = _wing_alloc_sort_workspace(num_tokens, num_experts, top_k, model_dim, block_m, device)
    aiter.moe_sorting_fwd(
        topk_ids, topk_weights,
        workspace["sid"], workspace["sw"], workspace["se"], workspace["nv"], workspace["out"],
        num_experts, block_m, None, None, 0,
    )

    w1_sc = w1s.view(dtypes.fp8_e8m0)
    w2_sc = w2s.view(dtypes.fp8_e8m0)
    mid = _wing_alloc_mid_hidden(num_tokens, top_k, inter_dim, device)

    if meta.ksplit > 1:
        # wing: BF16 path — don't fight it, just feed clean tensors.
        a1 = hidden_states.to(torch.bfloat16)
        a2 = meta.stage1(
            a1, w1, w2, workspace["sid"], workspace["se"], workspace["nv"], mid, top_k,
            block_m=block_m, a1_scale=None, w1_scale=w1_sc, sorted_weights=None,
        )
        meta.stage2(
            a2, w1, w2, workspace["sid"], workspace["se"], workspace["nv"], workspace["out"], top_k,
            w2_scale=w2_sc, a2_scale=None, block_m=block_m, sorted_weights=workspace["sw"],
        )
    else:
        a1, a1_sc = _wing_run_sorted_quant(
            hidden_states, workspace["sid"], workspace["nv"], num_tokens, 1, block_m, device,
        )
        a2 = meta.stage1(
            a1, w1, w2, workspace["sid"], workspace["se"], workspace["nv"], mid, top_k,
            block_m=block_m, a1_scale=a1_sc, w1_scale=w1_sc, sorted_weights=None,
        )
        flat = a2.view(-1, inter_dim)
        a2_q, a2_sc = _wing_run_sorted_quant(
            flat, workspace["sid"], workspace["nv"], num_tokens, top_k, block_m, device,
        )
        a2_q = a2_q.view(num_tokens, top_k, -1)
        meta.stage2(
            a2_q, w1, w2, workspace["sid"], workspace["se"], workspace["nv"], workspace["out"], top_k,
            w2_scale=w2_sc, a2_scale=a2_sc, block_m=block_m, sorted_weights=workspace["sw"],
        )

    return workspace["out"]
