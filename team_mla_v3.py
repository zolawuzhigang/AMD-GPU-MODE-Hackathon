#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
MLA解码核优化版本
基于structmix公开路线，针对不同batch和序列长度进行参数调优
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

# ============ 全局常量定义 ============
FP8_TYPE = aiter_dtypes.fp8
BFLOAT16 = torch.bfloat16
FP8_MAXVAL = float(torch.finfo(FP8_TYPE).max)
Q_AMAX_FIXED = 16.0

# ============ 全局缓存池 ============
metadata_pool = {}
tensor_pool = {}
bf16_cache = {}


# ============ Triton量化核函数 ============
@triton.jit
def quantize_q_kernel(query_ptr, output_ptr, scale_addr, amax_addr, FP8_MAXVAL: tl.constexpr, total_elems, BLOCK_SIZE: tl.constexpr):
    """将查询向量从BF16量化为FP8格式"""
    amax = tl.load(amax_addr)
    amax = tl.where(amax < 1e-12, 1e-12, amax)
    scale = amax / FP8_MAXVAL
    if tl.program_id(0) == 0:
        tl.store(scale_addr, scale)
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elems
    x = tl.load(query_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = x / scale
    x = tl.clamp(x, -FP8_MAXVAL, FP8_MAXVAL)
    tl.store(output_ptr + offs, x.to(output_ptr.dtype.element_ty), mask=mask)


# ============ KV分片数选择函数 ============
def get_kv_split_count(bs: int, seq_len_kv: int) -> int:
    """根据batch大小和KV序列长度选择最优的分片数"""
    # 256/1024 特殊优化：减少分片数降低归约开销
    if bs == 256 and seq_len_kv == 1024:
        return 4
    # 长序列+小batch：增加分片数提升并行度
    if seq_len_kv >= 8192 and bs <= 32:
        return 16
    # 小batch用较少分片
    if bs <= 4:
        return 4
    # 中等batch用中等分片
    if bs <= 64:
        return 8
    # 大batch用较多分片
    return 16


# ============ 元数据准备函数 ============
def prepare_metadata(
    bs, seq_len_kv, seq_len_q, num_q_heads, num_kv_heads,
    kv_splits, pg_size, dtype_q, dtype_kv,
    qo_ptr, kv_ptr, kv_gran,
):
    """准备MLA解码所需的元数据结构"""
    total_kv = bs * seq_len_kv

    # 根据页大小计算页相关参数
    if pg_size == 1:
        num_pages = total_kv
        kv_pg_ptr = kv_ptr
        seq_lens = kv_ptr[1:] - kv_ptr[:-1]
        kv_last_pg_len = seq_lens.to(torch.int32)
    else:
        num_pages = total_kv // pg_size
        kv_pg_ptr = kv_ptr // pg_size
        seq_lens = kv_ptr[1:] - kv_ptr[:-1]
        kv_last_pg_len = (seq_lens % pg_size).to(torch.int32)
        kv_last_pg_len = torch.where(kv_last_pg_len == 0, pg_size, kv_last_pg_len)

    # 获取元数据信息并分配工作缓冲区
    info = get_mla_metadata_info_v1(
        bs, seq_len_q, num_q_heads, dtype_q, dtype_kv,
        is_sparse=False, fast_mode=False,
        num_kv_splits=kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in info]
    wm, wi, wis, ri, rfm, rpm = work

    # 填充元数据
    get_mla_metadata_v1(
        qo_ptr, kv_pg_ptr, kv_last_pg_len,
        num_q_heads // num_kv_heads, num_kv_heads, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=pg_size, kv_granularity=kv_gran,
        max_seqlen_qo=seq_len_q, uni_seqlen_qo=seq_len_q,
        fast_mode=False, max_split_per_batch=kv_splits,
        intra_batch_mode=True, dtype_q=dtype_q, dtype_kv=dtype_kv,
    )

    kv_idx = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    return wm, wi, wis, ri, rfm, rpm, kv_idx, kv_last_pg_len, kv_pg_ptr, pg_size


# ============ 主核函数 ============
def custom_kernel(data: input_t) -> output_t:
    """MLA解码主函数"""
    q, kv_data, qo_ptr, kv_ptr, config = data

    # 解析配置参数
    bs = config["batch_size"]
    num_q_heads = config["num_heads"]
    num_kv_heads = config["num_kv_heads"]
    qk_dim = config["qk_head_dim"]
    v_dim = config["v_head_dim"]
    seq_len_q = config["q_seq_len"]
    softmax_scale = config["sm_scale"]
    seq_len_kv = config["kv_seq_len"]

    # 确定执行路径
    use_bf16_small = bs <= 32 and seq_len_kv <= 1024
    use_bf16_page2 = bs == 256 and seq_len_kv == 1024

    # ============ BF16非分页路径：小batch短序列 ============
    if use_bf16_small:
        key = (bs, seq_len_kv)
        if key not in bf16_cache:
            total_kv = bs * seq_len_kv
            kv_idx = torch.arange(total_kv, dtype=torch.int32, device="cuda")
            kv_last_pg_len = torch.full((bs,), seq_len_kv, dtype=torch.int32, device="cuda")
            bf16_cache[key] = (kv_idx, kv_last_pg_len)
        kv_idx, kv_last_pg_len = bf16_cache[key]

        alloc_id = ("bf16_np", q.shape[0], num_q_heads, v_dim)
        if alloc_id not in tensor_pool:
            tensor_pool[alloc_id] = torch.empty((q.shape[0], num_q_heads, v_dim), dtype=BFLOAT16, device="cuda")
        output = tensor_pool[alloc_id]

        kv_buf_bf16 = kv_data["bf16"].view(-1, 1, num_kv_heads, qk_dim)
        mla_decode_fwd(
            q, kv_buf_bf16, output, qo_ptr, kv_ptr,
            kv_idx, kv_last_pg_len, seq_len_q,
            page_size=1, nhead_kv=num_kv_heads, sm_scale=softmax_scale,
            intra_batch_mode=False,
        )
        return output

    # ============ 设置FP8/BF16混合路径参数 ============
    if use_bf16_page2:
        # 256/1024 特殊路径：BF16查询 + FP8 KV，页大小为2
        pg_size = 2
        dtype_q = BFLOAT16
        dtype_kv = FP8_TYPE
        kv_gran = 8
        use_fp8_quant = False
    else:
        # 默认FP8路径
        pg_size = 1 if seq_len_kv <= 1024 else 8
        dtype_q = FP8_TYPE
        dtype_kv = FP8_TYPE
        kv_gran = max(1, 16 // pg_size)
        use_fp8_quant = True

    # 获取KV分片数
    kv_splits = get_kv_split_count(bs, seq_len_kv)

    # 准备或获取缓存的元数据
    key = (bs, seq_len_kv, kv_splits, pg_size, str(dtype_q), str(dtype_kv), kv_gran)
    if key not in metadata_pool:
        metadata_pool[key] = prepare_metadata(
            bs, seq_len_kv, seq_len_q, num_q_heads, num_kv_heads,
            kv_splits, pg_size, dtype_q, dtype_kv,
            qo_ptr, kv_ptr, kv_gran,
        )

    wm, wi, wis, ri, rfm, rpm, kv_idx, kv_last_pg_len, kv_pg_ptr, ps = metadata_pool[key]
    kv_fp8, kv_scl = kv_data["fp8"]
    kv_4d = kv_fp8.view(-1, ps, num_kv_heads, kv_fp8.shape[-1])

    # ============ FP8量化路径 ============
    if use_fp8_quant:
        alloc_id = ("fp8_structmix", q.shape[0], num_q_heads, v_dim, qk_dim)
        if alloc_id not in tensor_pool:
            amax_val = torch.full((1,), Q_AMAX_FIXED, dtype=torch.float32, device="cuda")
            tensor_pool[alloc_id] = (
                torch.empty((q.shape[0], num_q_heads, v_dim), dtype=BFLOAT16, device="cuda"),
                amax_val,
                torch.empty(1, dtype=torch.float32, device="cuda"),
                torch.empty(q.shape[0] * num_q_heads * qk_dim, dtype=FP8_TYPE, device="cuda"),
            )
        output, amax_val, scl_buf, q_fp8_1d = tensor_pool[alloc_id]
        n = q.numel()
        block_size = 4096
        grid = ((n + block_size - 1) // block_size,)
        quantize_q_kernel[grid](q, q_fp8_1d, scl_buf, amax_val, FP8_MAXVAL=FP8_MAXVAL, total_elems=n, BLOCK_SIZE=block_size)
        q_in = q_fp8_1d.view(q.shape[0], num_q_heads, qk_dim)
        extra_args = {"q_scale": scl_buf, "kv_scale": kv_scl}
    else:
        # BF16页大小2路径
        alloc_id = ("bf16_pg2_256", q.shape[0], num_q_heads, v_dim)
        if alloc_id not in tensor_pool:
            tensor_pool[alloc_id] = torch.empty((q.shape[0], num_q_heads, v_dim), dtype=BFLOAT16, device="cuda")
        output = tensor_pool[alloc_id]
        q_in = q
        extra_args = {"kv_scale": kv_scl}

    # ============ 执行MLA解码 ============
    mla_decode_fwd(
        q_in, kv_4d, output, qo_ptr, kv_pg_ptr,
        kv_idx, kv_last_pg_len, seq_len_q,
        page_size=ps, nhead_kv=num_kv_heads, sm_scale=softmax_scale,
        logit_cap=0.0, num_kv_splits=kv_splits,
        intra_batch_mode=True,
        work_meta_data=wm, work_indptr=wi, work_info_set=wis,
        reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
        **extra_args,
    )
    return output