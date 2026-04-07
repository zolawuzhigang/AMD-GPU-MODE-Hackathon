#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

# MXFP4 矩阵乘法核函数 - AMD MI355X 优化版本

from task import input_t, output_t
import torch
import triton
import triton.language as tl


# ==================== 量化相关核函数 ====================

@triton.jit
def _mxfp4_quant_in_reg(
    x_bf16,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """将 BF16 数据块量化为 MXFP4 格式"""
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr = 32
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_K // MXFP4_QUANT_BLOCK_SIZE

    # 转换为 FP32 并计算缩放因子
    x_fp32 = x_bf16.to(tl.float32).reshape(
        BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE
    )

    amax = tl.max(tl.abs(x_fp32), axis=-1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    log2_amax = ((amax >> 23) & 0xFF).to(tl.int32) - 127

    scale_e8m0_unbiased_i = log2_amax - 2
    scale_e8m0_unbiased_i = tl.minimum(
        tl.maximum(scale_e8m0_unbiased_i, -127), 127
    )
    bs_e8m0 = scale_e8m0_unbiased_i.to(tl.uint8) + 127

    # 计算硬件缩放因子
    hw_scale_bits = (scale_e8m0_unbiased_i.to(tl.int32) + 127).to(tl.uint32) << 23
    hw_scale = hw_scale_bits.to(tl.float32, bitcast=True)

    # 广播缩放因子到每个元素对
    hw_scale_flat = tl.broadcast_to(
        hw_scale, (BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)
    )
    hw_scale_flat = hw_scale_flat.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K)
    hw_scale_pairs = hw_scale_flat.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // 2, 2)
    hw_scale_even, _ = tl.split(hw_scale_pairs)
    hw_scale_pair = hw_scale_even.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // 2)

    # 将 BF16 数据打包为 uint32 以便硬件指令处理
    x_u16 = x_bf16.to(tl.uint16, bitcast=True).reshape(
        BLOCK_SIZE_M, BLOCK_SIZE_K // 2, 2
    )
    lo_u16, hi_u16 = tl.split(x_u16)
    x_u32 = lo_u16.to(tl.uint32) | (hi_u16.to(tl.uint32) << 16)
    x_u32 = x_u32.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // 2)

    # 执行硬件 FP4 转换指令
    fp4_u32 = tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp4_bf16 $0, $1, $2",
        "=v, v, v",
        [x_u32, hw_scale_pair],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    x_fp4 = (fp4_u32 & 0xFF).to(tl.uint8)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // 2)

    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)


@triton.jit
def _standalone_quant_kernel(
    a_ptr, a_fp4_ptr, a_scale_ptr,
    M, K,
    stride_am, stride_ak,
    stride_qm, stride_qk,
    stride_sm, stride_sk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """独立的量化核函数"""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    m_mask = offs_m[:, None] < M
    k_mask = offs_k[None, :] < K

    a_bf16 = tl.load(a_ptrs, mask=m_mask & k_mask, other=0.0)
    a_fp4, a_scales = _mxfp4_quant_in_reg(a_bf16, BLOCK_SIZE_M, BLOCK_SIZE_K)

    HALF_K: tl.constexpr = BLOCK_SIZE_K // 2
    offs_qk = pid_k * HALF_K + tl.arange(0, HALF_K)
    q_ptrs = a_fp4_ptr + offs_m[:, None] * stride_qm + offs_qk[None, :] * stride_qk
    tl.store(q_ptrs, a_fp4, mask=m_mask & (offs_qk[None, :] < (K // 2)))

    SCALE_K: tl.constexpr = BLOCK_SIZE_K // 32
    offs_sk = pid_k * SCALE_K + tl.arange(0, SCALE_K)
    s_ptrs = a_scale_ptr + offs_m[:, None] * stride_sm + offs_sk[None, :] * stride_sk
    tl.store(s_ptrs, a_scales, mask=m_mask & (offs_sk[None, :] < (K // 32)))


# ==================== 融合量化 GEMM 核函数 ====================

@triton.heuristics({
    "EVEN_K": lambda args: (
        args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0
        and args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0
        and args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0
    ),
})
@triton.jit
def _fused_quant_gemm_preshuffle_kernel(
    a_ptr, b_ptr, c_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_ck, stride_cm, stride_cn,
    stride_bsn, stride_bsk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """融合量化和 GEMM 的核函数"""

    # 约束检查
    tl.assume(stride_am > 0); tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0); tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0); tl.assume(stride_cn > 0)
    tl.assume(stride_bsn > 0); tl.assume(stride_bsk > 0)

    SCALE_GROUP_SIZE: tl.constexpr = 32
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # 计算 PID
    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT

    # 根据是否使用 split-k 选择不同的调度策略
    if NUM_KSPLIT == 1:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0); tl.assume(pid_n >= 0); tl.assume(pid_k >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)

        # 计算 A 矩阵的指针
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_ak = pid_k * SPLITK_BLOCK_SIZE + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)

        # 计算 B 矩阵的指针（预洗牌格式）
        offs_k_shuffle_arr = tl.arange(0, (BLOCK_SIZE_K // 2) * 16)
        offs_k_shuffle = pid_k * (SPLITK_BLOCK_SIZE // 2) * 16 + offs_k_shuffle_arr
        offs_bn = (pid_n * (BLOCK_SIZE_N // 16) + tl.arange(0, BLOCK_SIZE_N // 16)) % (N // 16)
        b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k_shuffle[None, :] * stride_bk)

        # 计算 B 缩放因子的指针
        offs_bsn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // 32) * 32
        offs_ks = pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE) * 32 + tl.arange(
            0, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32
        )
        b_scale_ptrs = (
            b_scales_ptr + offs_bsn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # K 维度循环
        for k_iter in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            if EVEN_K:
                a_bf16 = tl.load(a_ptrs)
                b_scales_raw = tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
                b_raw = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                k_offset = (k_iter - pid_k * num_k_iter) * BLOCK_SIZE_K
                a_bf16 = tl.load(
                    a_ptrs,
                    mask=tl.arange(0, BLOCK_SIZE_K)[None, :] < (2 * K - pid_k * SPLITK_BLOCK_SIZE - k_offset),
                    other=0.0,
                )
                b_scales_raw = tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
                b_raw = tl.load(
                    b_ptrs,
                    mask=offs_k_shuffle_arr[None, :] < (
                        (K - (pid_k * (SPLITK_BLOCK_SIZE // 2) + (k_iter - pid_k * num_k_iter) * (BLOCK_SIZE_K // 2))) * 16
                    ),
                    other=0,
                    cache_modifier=cache_modifier,
                )

            # 量化 A 矩阵
            a_fp4, a_scales = _mxfp4_quant_in_reg(a_bf16, BLOCK_SIZE_M, BLOCK_SIZE_K)

            # 反洗牌 B 的缩放因子
            b_scales = (
                b_scales_raw
                .reshape(BLOCK_SIZE_N // 32, BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1)
                .permute(0, 5, 3, 1, 4, 2, 6)
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            )

            # 反洗牌 B 的数据
            b = (
                b_raw.reshape(1, BLOCK_SIZE_N // 16, BLOCK_SIZE_K // 64, 2, 16, 16)
                .permute(0, 1, 4, 2, 3, 5)
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // 2)
                .trans(1, 0)
            )

            # 执行缩放点积
            accumulator = tl.dot_scaled(
                a_fp4, a_scales, "e2m1", b, b_scales, "e2m1", accumulator
            )

            # 更新指针
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 2) * 16 * stride_bk
            b_scale_ptrs += BLOCK_SIZE_K * stride_bsk

        # 存储结果
        c = accumulator.to(c_ptr.type.element_ty)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :] + pid_k * stride_ck
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


# ==================== Reduce 核函数 ====================

@triton.jit
def _reduce_kernel(
    c_in_ptr, c_out_ptr, M, N,
    stride_c_in_k, stride_c_in_m, stride_c_in_n,
    stride_c_out_m, stride_c_out_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr, MAX_KSPLIT: tl.constexpr,
):
    """将多个部分结果累加为最终结果"""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    base_ptrs = c_in_ptr + offs_m[:, None] * stride_c_in_m + offs_n[None, :] * stride_c_in_n
    acc = tl.load(base_ptrs).to(tl.float32)

    for ks in tl.static_range(1, MAX_KSPLIT):
        if ks < ACTUAL_KSPLIT:
            acc += tl.load(base_ptrs + ks * stride_c_in_k).to(tl.float32)

    c = acc.to(c_out_ptr.type.element_ty)
    c_out_ptrs = c_out_ptr + offs_m[:, None] * stride_c_out_m + offs_n[None, :] * stride_c_out_n
    tl.store(c_out_ptrs, c)


# ==================== 仅 GEMM 核函数（输入已量化） ====================

@triton.heuristics({
    "EVEN_K": lambda args: (
        args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0
        and args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0
        and args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0
    ),
})
@triton.jit
def _gemm_only_preshuffle_kernel(
    a_fp4_ptr, a_scale_ptr, b_ptr, c_ptr, b_scales_ptr,
    M, N, K,
    stride_qm, stride_qk,
    stride_sm, stride_sk,
    stride_bn, stride_bk,
    stride_ck, stride_cm, stride_cn,
    stride_bsn, stride_bsk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """仅执行 GEMM 计算（输入已经是 FP4 格式）"""

    tl.assume(stride_qm > 0); tl.assume(stride_qk > 0)
    tl.assume(stride_sm > 0); tl.assume(stride_sk > 0)
    tl.assume(stride_bn > 0); tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0); tl.assume(stride_cn > 0)
    tl.assume(stride_bsn > 0); tl.assume(stride_bsk > 0)

    SCALE_GROUP_SIZE: tl.constexpr = 32
    HALF_BK: tl.constexpr = BLOCK_SIZE_K // 2
    SCALE_BK: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_SIZE

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT

    if NUM_KSPLIT == 1:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0); tl.assume(pid_n >= 0); tl.assume(pid_k >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE // 2, HALF_BK)

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_aqk = pid_k * (SPLITK_BLOCK_SIZE // 2) + tl.arange(0, HALF_BK)
        a_fp4_ptrs = a_fp4_ptr + (offs_am[:, None] * stride_qm + offs_aqk[None, :] * stride_qk)

        offs_ask = pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE) + tl.arange(0, SCALE_BK)
        a_scale_ptrs = a_scale_ptr + (offs_am[:, None] * stride_sm + offs_ask[None, :] * stride_sk)

        offs_k_shuffle_arr = tl.arange(0, HALF_BK * 16)
        offs_k_shuffle = pid_k * (SPLITK_BLOCK_SIZE // 2) * 16 + offs_k_shuffle_arr
        offs_bn = (pid_n * (BLOCK_SIZE_N // 16) + tl.arange(0, BLOCK_SIZE_N // 16)) % (N // 16)
        b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k_shuffle[None, :] * stride_bk)

        offs_bsn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // 32) * 32
        offs_ks = pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE) * 32 + tl.arange(0, SCALE_BK * 32)
        b_scale_ptrs = b_scales_ptr + offs_bsn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_iter in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            if EVEN_K:
                a_fp4 = tl.load(a_fp4_ptrs, cache_modifier=cache_modifier)
                a_scales = tl.load(a_scale_ptrs, cache_modifier=cache_modifier)
            else:
                k_off = (k_iter - pid_k * num_k_iter) * HALF_BK
                k_remain = K - (pid_k * (SPLITK_BLOCK_SIZE // 2) + k_off)
                a_fp4 = tl.load(
                    a_fp4_ptrs,
                    mask=tl.arange(0, HALF_BK)[None, :] < k_remain,
                    other=0,
                    cache_modifier=cache_modifier,
                )
                s_remain = (
                    (2 * K) // SCALE_GROUP_SIZE
                    - (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE) + (k_iter - pid_k * num_k_iter) * SCALE_BK)
                )
                a_scales = tl.load(
                    a_scale_ptrs,
                    mask=tl.arange(0, SCALE_BK)[None, :] < s_remain,
                    other=0,
                    cache_modifier=cache_modifier,
                )

            b_scales = (
                tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
                .reshape(BLOCK_SIZE_N // 32, SCALE_BK // 8, 4, 16, 2, 2, 1)
                .permute(0, 5, 3, 1, 4, 2, 6)
                .reshape(BLOCK_SIZE_N, SCALE_BK)
            )

            if EVEN_K:
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                b = tl.load(
                    b_ptrs,
                    mask=offs_k_shuffle_arr[None, :] < (
                        (K - (pid_k * (SPLITK_BLOCK_SIZE // 2) + (k_iter - pid_k * num_k_iter) * HALF_BK)) * 16
                    ),
                    other=0,
                    cache_modifier=cache_modifier,
                )

            b = (
                b.reshape(1, BLOCK_SIZE_N // 16, BLOCK_SIZE_K // 64, 2, 16, 16)
                .permute(0, 1, 4, 2, 3, 5)
                .reshape(BLOCK_SIZE_N, HALF_BK)
                .trans(1, 0)
            )

            accumulator = tl.dot_scaled(a_fp4, a_scales, "e2m1", b, b_scales, "e2m1", accumulator)

            a_fp4_ptrs += HALF_BK * stride_qk
            a_scale_ptrs += SCALE_BK * stride_sk
            b_ptrs += HALF_BK * 16 * stride_bk
            b_scale_ptrs += BLOCK_SIZE_K * stride_bsk

        c = accumulator.to(c_ptr.type.element_ty)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :] + pid_k * stride_ck
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


# ==================== SplitK 参数计算 ====================

def get_splitk(K, BLOCK_SIZE_K, NUM_KSPLIT):
    """计算 SplitK 的最优参数"""
    SPLITK_BLOCK_SIZE = triton.cdiv((2 * triton.cdiv(K, NUM_KSPLIT)), BLOCK_SIZE_K) * BLOCK_SIZE_K

    while NUM_KSPLIT > 1 and BLOCK_SIZE_K > 16:
        if (
            K % (SPLITK_BLOCK_SIZE // 2) == 0
            and SPLITK_BLOCK_SIZE % BLOCK_SIZE_K == 0
            and K % (BLOCK_SIZE_K // 2) == 0
        ):
            break
        elif K % (SPLITK_BLOCK_SIZE // 2) != 0 and NUM_KSPLIT > 1:
            NUM_KSPLIT = NUM_KSPLIT // 2
        elif SPLITK_BLOCK_SIZE % BLOCK_SIZE_K != 0:
            if NUM_KSPLIT > 1:
                NUM_KSPLIT = NUM_KSPLIT // 2
            elif BLOCK_SIZE_K > 16:
                BLOCK_SIZE_K = BLOCK_SIZE_K // 2
        elif K % (BLOCK_SIZE_K // 2) != 0 and BLOCK_SIZE_K > 16:
            BLOCK_SIZE_K = BLOCK_SIZE_K // 2
        else:
            break
        SPLITK_BLOCK_SIZE = triton.cdiv((2 * triton.cdiv(K, NUM_KSPLIT)), BLOCK_SIZE_K) * BLOCK_SIZE_K

    NUM_KSPLIT = triton.cdiv(K, (SPLITK_BLOCK_SIZE // 2))
    return SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT


# ==================== 调优配置 ====================

TUNE_CONFIGS = {
    (4, 2880, 512): {
        'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256,
        'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2,
        'waves_per_eu': 1, 'matrix_instr_nonkdim': 16,
        'cache_modifier': '.cg', 'NUM_KSPLIT': 1
    },
    (16, 2112, 7168): {
        'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 512,
        'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2,
        'waves_per_eu': 3, 'matrix_instr_nonkdim': 16,
        'cache_modifier': '.cg', 'NUM_KSPLIT': 14
    },
    (32, 4096, 512): {
        'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256,
        'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3,
        'waves_per_eu': 3, 'matrix_instr_nonkdim': 16,
        'cache_modifier': '.cg', 'NUM_KSPLIT': 1
    },
    (32, 2880, 512): {
        'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256,
        'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2,
        'waves_per_eu': 2, 'matrix_instr_nonkdim': 16,
        'cache_modifier': None, 'NUM_KSPLIT': 1
    },
    (64, 7168, 2048): {
        'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 512,
        'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2,
        'waves_per_eu': 2, 'matrix_instr_nonkdim': 16,
        'cache_modifier': '.cg', 'NUM_KSPLIT': 1
    },
    (256, 3072, 1536): {
        'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 512,
        'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2,
        'waves_per_eu': 2, 'matrix_instr_nonkdim': 16,
        'cache_modifier': None, 'NUM_KSPLIT': 1
    },
}

DEFAULT_TUNE_CONFIG = {
    'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256,
    'GROUP_SIZE_M': 1, 'num_warps': 2, 'num_stages': 2,
    'waves_per_eu': 0, 'matrix_instr_nonkdim': 16,
    'cache_modifier': '.cg', 'NUM_KSPLIT': 1
}

GEMM_TUNE_CONFIGS = {
    (32, 4096, 512): {
        'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256,
        'GROUP_SIZE_M': 4, 'num_warps': 4, 'num_stages': 2,
        'waves_per_eu': 0, 'matrix_instr_nonkdim': 16,
        'cache_modifier': '.cg', 'NUM_KSPLIT': 1
    },
    (32, 2880, 512): {
        'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256,
        'GROUP_SIZE_M': 4, 'num_warps': 4, 'num_stages': 2,
        'waves_per_eu': 0, 'matrix_instr_nonkdim': 16,
        'cache_modifier': '.cg', 'NUM_KSPLIT': 1
    },
    (64, 7168, 2048): {
        'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 512,
        'GROUP_SIZE_M': 4, 'num_warps': 4, 'num_stages': 2,
        'waves_per_eu': 0, 'matrix_instr_nonkdim': 16,
        'cache_modifier': '.cg', 'NUM_KSPLIT': 2
    },
}

GEMM_DEFAULT_CONFIG = {
    'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256,
    'GROUP_SIZE_M': 4, 'num_warps': 4, 'num_stages': 2,
    'waves_per_eu': 0, 'matrix_instr_nonkdim': 16,
    'cache_modifier': '.cg', 'NUM_KSPLIT': 1
}


# ==================== 缓存和辅助函数 ====================

_buf_cache = {}
_config_cache = {}
_launch_cache = {}


def _get_buffers(m, n, num_ksplit, device):
    """获取或创建输出缓冲区"""
    key = (m, n, num_ksplit)
    if key not in _buf_cache:
        y = torch.empty((m, n), dtype=torch.bfloat16, device=device)
        y_pp = torch.empty((num_ksplit, m, n), dtype=torch.float32, device=device) if num_ksplit > 1 else None
        _buf_cache[key] = (y, y_pp)
    return _buf_cache[key]


def _get_config(m, n, k):
    """获取指定形状的调优配置"""
    key = (m, n, k)
    if key not in _config_cache:
        config = TUNE_CONFIGS.get(key, DEFAULT_TUNE_CONFIG).copy()
        K_packed = k // 2

        if config["NUM_KSPLIT"] > 1:
            SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
                K_packed, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
            )
            config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
            config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
            config["NUM_KSPLIT"] = NUM_KSPLIT
        else:
            config["SPLITK_BLOCK_SIZE"] = 2 * K_packed
            config["NUM_KSPLIT"] = 1

        if config["BLOCK_SIZE_K"] >= 2 * K_packed:
            config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K_packed)
            config["SPLITK_BLOCK_SIZE"] = 2 * K_packed
            config["NUM_KSPLIT"] = 1

        config["BLOCK_SIZE_N"] = max(config["BLOCK_SIZE_N"], 32)
        _config_cache[key] = config
    return _config_cache[key]


def _prepare_b_views(B_shuffle, B_scale_sh, n, k_packed):
    """准备权重矩阵的视图"""
    b_reshaped = B_shuffle.view(torch.uint8).reshape(n // 16, k_packed * 16)
    b_scale_uint8 = B_scale_sh.view(torch.uint8)
    return b_reshaped, b_scale_uint8


def _build_launch_params(m, n, k, device):
    """预先计算所有启动参数"""
    config = _get_config(m, n, k)
    K_packed = k // 2
    ks = config["NUM_KSPLIT"]

    y, y_pp = _get_buffers(m, n, ks, device)
    grid = (ks * triton.cdiv(m, config["BLOCK_SIZE_M"]) * triton.cdiv(n, config["BLOCK_SIZE_N"]),)

    if ks == 1:
        c_stride_k, c_stride_m, c_stride_n = 0, y.stride(0), y.stride(1)
    else:
        c_stride_k, c_stride_m, c_stride_n = y_pp.stride(0), y_pp.stride(1), y_pp.stride(2)

    params = {
        'config': config,
        'K_packed': K_packed,
        'grid': grid,
        'ks': ks,
        'c_stride_k': c_stride_k,
        'c_stride_m': c_stride_m,
        'c_stride_n': c_stride_n,
    }

    if ks > 1:
        params['reduce_grid'] = (triton.cdiv(m, 16), triton.cdiv(n, 64))
        params['actual_ksplit'] = triton.cdiv(K_packed, (config["SPLITK_BLOCK_SIZE"] // 2))
        params['max_ksplit'] = triton.next_power_of_2(ks)

    return params


# ==================== 主计算函数 ====================

def fused_quant_gemm(A_bf16, B_shuffle, B_scale_sh, m, n, k):
    """融合量化和 GEMM 计算"""
    key = (m, n, k)
    if key not in _launch_cache:
        _launch_cache[key] = _build_launch_params(m, n, k, A_bf16.device)

    p = _launch_cache[key]
    y, y_pp = _get_buffers(m, n, p['ks'], A_bf16.device)

    b_reshaped, b_scale_uint8 = _prepare_b_views(B_shuffle, B_scale_sh, n, p['K_packed'])

    _fused_quant_gemm_preshuffle_kernel[p['grid']](
        A_bf16, b_reshaped,
        y if p['ks'] == 1 else y_pp,
        b_scale_uint8,
        m, n, p['K_packed'],
        A_bf16.stride(0), A_bf16.stride(1),
        b_reshaped.stride(0), b_reshaped.stride(1),
        p['c_stride_k'], p['c_stride_m'], p['c_stride_n'],
        b_scale_uint8.stride(0), b_scale_uint8.stride(1),
        **p['config'],
    )

    if p['ks'] > 1:
        _reduce_kernel[p['reduce_grid']](
            y_pp, y, m, n,
            y_pp.stride(0), y_pp.stride(1), y_pp.stride(2),
            y.stride(0), y.stride(1),
            16, 64,
            p['actual_ksplit'], p['max_ksplit'],
        )

    return y


def separate_quant_gemm(A_bf16, B_shuffle, B_scale_sh, m, n, k):
    """分离量化和 GEMM 计算"""
    K_packed = k // 2
    K_bf16 = k

    QUANT_BM = 16
    QUANT_BK = 256

    A_fp4 = torch.empty((m, K_packed), dtype=torch.uint8, device=A_bf16.device)
    A_scale = torch.empty((m, K_bf16 // 32), dtype=torch.uint8, device=A_bf16.device)

    grid_quant = (triton.cdiv(m, QUANT_BM), triton.cdiv(K_bf16, QUANT_BK))
    _standalone_quant_kernel[grid_quant](
        A_bf16, A_fp4, A_scale,
        m, K_bf16,
        A_bf16.stride(0), A_bf16.stride(1),
        A_fp4.stride(0), A_fp4.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        QUANT_BM, QUANT_BK,
    )

    config = GEMM_TUNE_CONFIGS.get((m, n, k), GEMM_DEFAULT_CONFIG).copy()

    if config["NUM_KSPLIT"] > 1:
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
            K_packed, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
        )
        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K_packed
        config["NUM_KSPLIT"] = 1

    if config["BLOCK_SIZE_K"] >= 2 * K_packed:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K_packed)
        config["SPLITK_BLOCK_SIZE"] = 2 * K_packed
        config["NUM_KSPLIT"] = 1

    config["BLOCK_SIZE_N"] = max(config["BLOCK_SIZE_N"], 32)

    y = torch.empty((m, n), dtype=torch.bfloat16, device=A_bf16.device)

    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty((config["NUM_KSPLIT"], m, n), dtype=torch.float32, device=A_bf16.device)
    else:
        y_pp = None

    b_reshaped, b_scale_uint8 = _prepare_b_views(B_shuffle, B_scale_sh, n, K_packed)

    grid = lambda META: (
        META["NUM_KSPLIT"]
        * triton.cdiv(m, META["BLOCK_SIZE_M"])
        * triton.cdiv(n, META["BLOCK_SIZE_N"]),
    )

    _gemm_only_preshuffle_kernel[grid](
        A_fp4, A_scale,
        b_reshaped,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        b_scale_uint8,
        m, n, K_packed,
        A_fp4.stride(0), A_fp4.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        b_reshaped.stride(0), b_reshaped.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        b_scale_uint8.stride(0), b_scale_uint8.stride(1),
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 16
        REDUCE_BLOCK_SIZE_N = 64
        ACTUAL_KSPLIT = triton.cdiv(K_packed, (config["SPLITK_BLOCK_SIZE"] // 2))
        grid_reduce = (triton.cdiv(m, REDUCE_BLOCK_SIZE_M), triton.cdiv(n, REDUCE_BLOCK_SIZE_N))

        _reduce_kernel[grid_reduce](
            y_pp, y, m, n,
            y_pp.stride(0), y_pp.stride(1), y_pp.stride(2),
            y.stride(0), y.stride(1),
            REDUCE_BLOCK_SIZE_M, REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT, triton.next_power_of_2(config["NUM_KSPLIT"]),
        )

    return y


def custom_kernel(data: input_t) -> output_t:
    """主入口函数"""
    A = data[0]
    return fused_quant_gemm(A, data[3], data[4], A.shape[0], data[1].shape[0], A.shape[1])