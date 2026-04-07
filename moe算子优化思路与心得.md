# MoE MXFP4 算子优化心得

## 背景

参加AMD GPU MODE Hackathon的MoE MXFP4优化赛道，目标是在MI355X上实现高性能的Mixture-of-Experts推理算子。本文总结优化过程中的技术决策与经验教训。

## 问题分析

MoE的核心计算流程：
1. **Expert Routing**: 根据gating网络选择top-k experts
2. **Token Sorting**: 将tokens按expert分组
3. **Expert Computation**: gate_up GEMM → SwiGLU → down GEMM
4. **Weighted Reduction**: 按routing weights合并输出

优化难点：
- **不规则访存**: tokens被路由到不同experts，访问模式分散
- **Padding开销**: expert负载不均导致的计算浪费
- **量化精度**: MXFP4带来的精度损失需要控制在可接受范围
- **Kernel选择**: 不同shape最优kernel不同，难以一概而论

## 关键优化点

### 1. block_m参数调优

`block_m`是moe_sorting的核心参数，决定了token分组的粒度。

**核心认知**：不存在"万能"的block_m值。

- **E=257场景**: 256个routed experts，专家数量多，每个expert处理的tokens较少。较小的block_m虽然launch次数多，但padding浪费少。实测bs=512时block_m=32效果优于64。

- **E=33场景**: 32个routed experts，每个expert负载更重。大batch时block_m=64能减少kernel launch开销，收益大于额外padding。

- **inter_dim影响**: d=2048时，每个token计算量更大，对memory bandwidth要求更高，需要重新评估block_m。

**经验值**（仅供参考，实际需benchmark验证）：

| Shape | block_m |
|-------|---------|
| bs=16, E=257 | 16 |
| bs=128, E=257 | 32 |
| bs=512, E=257 | 32 |
| bs=512, E=33, d=512 | 64 |
| bs=512, E=33, d=2048 | 32 |

### 2. 两阶段流水线选择

底层优化框架提供两条路径：

**ksplit > 1 路径**（cktile_moe）:
- Activation保持BF16，不做FP4量化
- 适合小batch场景，避免量化开销
- 当M较小时，量化kernel的时间占比过高

**ksplit = 0 路径**（CK 2-stage）:
- Activation量化为FP4
- 利用MXFP4的2x带宽优势
- 适合大batch，带宽瓶颈场景

**决策逻辑**：当M × top_k足够大时，FP4路径才有收益。实测bs=16时用ksplit=2，bs≥128时用ksplit=0。

### 3. 持久化Buffer池

每次调用都malloc会导致：
- 内存碎片化
- 潜在的同步开销
- 难以预测的延迟抖动

**解决方案**：用模块级字典缓存tensor，key为shape tuple。

```python
_mem_pool = {}

def alloc_buffer(shape_key, factory):
    if shape_key not in _mem_pool:
        _mem_pool[shape_key] = factory()
    return _mem_pool[shape_key]
```

注意：不同shape的请求会创建不同buffer，需权衡内存占用和命中效率。

### 4. FlyDSL Stage2配置

Stage2（down GEMM + reduction）可选FlyDSL kernel。关键参数：

- **tile_k=128**: 相比默认64，更好的MAC利用率
- **tile_m=16**: 对于reduction操作，小tile减少竞争
- **atomic模式**: 直接原子加，避免二次同步

### 5. Stage1 Kernel选择

CK框架提供多个预编译kernel：

- **M128 kernel**: `256x128x128x128`，适合标准场景
- **M32 kernel**: `256x32x128x128`，适合expert稀疏场景

E=257时expert分布更稀疏，M32 kernel的wavefront利用率更高。但需要实测验证，不可凭直觉。

## 踩坑记录

### 坑1：use_non_temporal_load误用

最初认为non_temporal_load能提升大batch性能，但实际测试导致超时。

**原因**：该参数需要kernel支持，且对数据访问模式有要求。盲目添加可能破坏缓存局部性。

**教训**：优化参数需要逐个验证，不可堆砌。

### 坑2：配置表key格式错误

底层框架的配置查找使用13元组作为key，任何字段格式不匹配都会静默失败。

```python
# 错误示例
"ActivationType.Silu"  # 正确
"silu"                 # 错误，静默miss
```

**教训**：打印实际使用的key，与CSV对比验证。

### 坑3：Rate Limit策略

Leaderboard每小时只能提交一次，测试模式和正式模式分开计数。

**教训**：
- 先用test模式验证正确性
- 确认无误后再提交leaderboard
- 预留时间buffer，避免截止前匆忙

## 性能数据

最终提交成绩（MI355X）：

| Shape | 延迟 (µs) |
|-------|-----------|
| bs=16/E=257/d=256 | 90.8 |
| bs=128/E=257/d=256 | 104 |
| bs=512/E=257/d=256 | 140 |
| bs=16/E=33/d=512 | 62.0 |
| bs=128/E=33/d=512 | 90.0 |
| bs=512/E=33/d=512 | 105 |
| bs=512/E=33/d=2048 | 185 |

## 总结

MoE优化的核心不是"一招鲜"，而是针对每个shape找到最合适的配置组合：

1. **理解数据流**：知道每个阶段的瓶颈在哪里
2. **逐参数调优**：改动一个参数，跑benchmark，记录结果
3. **建立intuition**：为什么这个shape用这个配置更好
4. **保持简洁**：过度优化往往是优化的反面

心得：最有效的优化往往来自对问题的深刻理解，而非技巧的堆砌。