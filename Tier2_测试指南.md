# Tier 2 升级功能测试指南 ✅

**版本:** V24.5-agent-h (Tier 2 第一阶段)  
**发布日期:** 2025年11月29日  
**测试范围:** 查询扩展 + 多信号排序 + 智能缓存 + 查询日志

---

## 🎯 功能测试清单

### 功能 1️⃣：查询扩展 (Query Expansion)

#### 测试 1.1: 同义词库验证

```python
from 工程重构版V24_4-agent_h import expand_query, SYNONYMS_DICT

# 查看同义词库
print("=== 同义词库 ===")
for word, synonyms in list(SYNONYMS_DICT.items())[:5]:
    print(f"{word}: {synonyms}")

# 验证扩展效果
query = "电力现货"
variants = expand_query(query)
print(f"\n原始查询: {query}")
print(f"扩展变体: {variants}")

# 期望输出:
# ["电力现货", "电能现货", "电力即期", ...]
```

**预期结果:** ✅ 返回 3-5 个变体，包括同义词替换

#### 测试 1.2: 多变体检索

```python
# 测试 retrieve() 中的查询扩展集成
materials = MaterialManager("./data")  # 加载素材
materials.load()

results = materials.retrieve("光伏装机", top_k=5)
print(results)

# 查看日志输出: 应该看到 "查询扩展: N 个变体"
```

**预期结果:** ✅ 检索日志显示多个变体被处理，结果更全面

---

### 功能 2️⃣：多信号排序 (Multi-Signal Ranking)

#### 测试 2.1: 综合分数计算

```python
from 工程重构版V24_4-agent_h import compute_relevance_score

# 测试综合分数
scores = [
    compute_relevance_score(bm25_score=8.5, doc_weight=1.0, doc_length=500),
    compute_relevance_score(bm25_score=5.0, doc_weight=0.8, doc_length=1000),
    compute_relevance_score(bm25_score=3.0, doc_weight=0.5, doc_length=200),
]

for i, score in enumerate(scores):
    print(f"结果 {i+1}: 综合分数 = {score:.3f}")

# 期望: 第一个分数最高，因为 BM25 得分最高
```

**预期结果:** ✅ 分数在 0-1 范围，高得分对应高 BM25 分数

#### 测试 2.2: 排序准确性对比

```python
# Tier 1 vs Tier 2 排序对比
# 旧版本: 仅基于 BM25 分数排序
# 新版本: 考虑权重、长度、可信度

# 查看 retrieve() 输出中的 "综合分数" 而不是 "匹配度"
```

**预期结果:** ✅ 输出显示 "综合分数" 而不是单一的 "匹配度"

---

### 功能 3️⃣：智能缓存 (Smart Caching)

#### 测试 3.1: 内存缓存验证

```python
from 工程重构版V24_4-agent_h import get_cache_manager
import time

cache_mgr = get_cache_manager()
if cache_mgr:
    # 第一次查询（不命中）
    print("=== 第一次查询 ===")
    result1, hit1, time1 = cache_mgr.get("电力现货")
    print(f"命中: {hit1}, 响应时间: {time1:.2f}ms")
    
    # 缓存结果
    cache_mgr.set("电力现货", "测试结果内容")
    
    # 第二次查询（应该命中）
    print("\n=== 第二次查询（相同） ===")
    result2, hit2, time2 = cache_mgr.get("电力现货")
    print(f"命中: {hit2}, 响应时间: {time2:.2f}ms")
    print(f"加速倍数: {time1/time2:.1f}x")
    
    # 查看统计
    stats = cache_mgr.get_stats()
    print(f"\n缓存统计: {stats}")
else:
    print("⚠️ 缓存未启用 (ENABLE_CACHE=false)")
```

**预期结果:** ✅ 
- 第一次命中：False，时间 > 1ms
- 第二次命中：True，时间 < 1ms
- 加速倍数：> 5x

#### 测试 3.2: 磁盘缓存验证

```python
import os

cache_mgr = get_cache_manager()
if cache_mgr:
    cache_dir = cache_mgr.cache_dir
    print(f"缓存目录: {cache_dir}")
    
    # 检查缓存文件
    cache_files = os.listdir(cache_dir)
    print(f"缓存文件数: {len([f for f in cache_files if f.endswith('.json')])}")
    
    # 查看缓存文件大小
    for filename in cache_files[:3]:
        path = os.path.join(cache_dir, filename)
        size = os.path.getsize(path)
        print(f"  {filename}: {size} bytes")
```

**预期结果:** ✅ 缓存目录存在，包含 .json 缓存文件

#### 测试 3.3: 缓存过期清理

```python
# 清理过期缓存
deleted = cache_mgr.cleanup()
print(f"清理删除: {deleted} 个过期缓存")

# 验证清理后的缓存大小
stats_after = cache_mgr.get_stats()
print(f"清理后统计: {stats_after}")
```

**预期结果:** ✅ cleanup() 返回删除的文件数

---

### 功能 4️⃣：查询日志 (Query Analytics)

#### 测试 4.1: 日志记录验证

```python
from 工程重构版V24_4-agent_h import get_query_analytics
import time

analytics = get_query_analytics()
if analytics:
    print(f"日志文件: {analytics.log_file}")
    
    # 模拟几次查询
    analytics.log_query("电力现货", method="BM25+jieba", results_count=5, response_time_ms=18, cache_hit=False)
    analytics.log_query("电力现货", method="Cache", results_count=0, response_time_ms=2, cache_hit=True)
    analytics.log_query("光伏", method="BM25+jieba", results_count=3, response_time_ms=15, cache_hit=False)
    
    # 查看日志文件
    with open(analytics.log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"\n日志文件内容 (前 5 行):")
        for line in lines[:5]:
            print(f"  {line.strip()}")
else:
    print("⚠️ 查询日志未启用 (ENABLE_QUERY_LOG=false)")
```

**预期结果:** ✅ 
- 日志文件存在于 `query_log.csv`
- 包含时间戳、查询、方法、结果数、响应时间、缓存命中状态

#### 测试 4.2: 查询统计分析

```python
# 获取最常见的查询
top_queries = analytics.get_top_queries(limit=5)
print("=== 最常见的 5 个查询 ===")
for query, count in top_queries:
    print(f"  {query}: {count} 次")
```

**预期结果:** ✅ 返回查询频率排序的列表

---

### 功能 5️⃣：完整集成测试 (End-to-End)

#### 测试 5.1: 完整检索流程

```python
import time

# 初始化
materials = MaterialManager("./data")
materials.load()

print("=== 完整检索流程测试 ===\n")

# 测试 1: 第一次查询（无缓存）
print("1️⃣ 第一次查询（无缓存）")
start = time.time()
result1 = materials.retrieve("电力现货价格", top_k=5)
time1 = (time.time() - start) * 1000
print(f"   结果长度: {len(result1)} 字符")
print(f"   响应时间: {time1:.1f}ms\n")

# 测试 2: 重复查询（缓存命中）
print("2️⃣ 重复查询（缓存命中）")
start = time.time()
result2 = materials.retrieve("电力现货价格", top_k=5)
time2 = (time.time() - start) * 1000
print(f"   结果长度: {len(result2)} 字符")
print(f"   响应时间: {time2:.1f}ms")
print(f"   加速倍数: {time1/time2:.1f}x\n")

# 测试 3: 扩展查询（同义词）
print("3️⃣ 查询扩展测试")
variants = expand_query("光伏", max_variants=3)
print(f"   原始查询: 光伏")
print(f"   扩展变体: {variants}\n")

# 测试 4: 日志查看
print("4️⃣ 查询日志统计")
analytics = get_query_analytics()
if analytics:
    stats = analytics.get_top_queries(3)
    print(f"   最常见查询: {stats}")
```

**预期输出:**
```
=== 完整检索流程测试 ===

1️⃣ 第一次查询（无缓存）
   🔍 查询扩展: 4 个变体
   🧩 [RAG] 命中 5 个片段 [BM25+jieba] (4 变体) (18.2ms)
   结果长度: 2350 字符
   响应时间: 18.2ms

2️⃣ 重复查询（缓存命中）
   ⚡ [Cache] 命中缓存 (1.3ms)
   结果长度: 2350 字符
   响应时间: 1.3ms
   加速倍数: 14.0x

3️⃣ 查询扩展测试
   原始查询: 光伏
   扩展变体: ['光伏', '太阳能', '光伏发电']

4️⃣ 查询日志统计
   最常见查询: [('电力现货价格', 2), ('光伏', 1)]
```

---

## 📊 性能基准测试

### 测试环境

```
OS: macOS
Python: 3.14
依赖: jieba, rank-bm25, scikit-learn, requests
数据量: 测试素材集
```

### 基准测试脚本

```python
import time
from 工程重构版V24_4-agent_h import MaterialManager, expand_query

# 初始化
materials = MaterialManager("./data")
materials.load()

print("=== 性能基准测试 ===\n")

# 测试查询列表
test_queries = [
    "电力现货",
    "光伏装机",
    "电网运维",
    "价格预测",
    "新能源"
]

print("1️⃣ 查询扩展性能")
print("-" * 50)
for query in test_queries[:3]:
    start = time.time()
    variants = expand_query(query)
    elapsed = (time.time() - start) * 1000
    print(f"  查询: {query}")
    print(f"    变体数: {len(variants)}, 耗时: {elapsed:.2f}ms")

print("\n2️⃣ 检索性能（无缓存）")
print("-" * 50)
for query in test_queries:
    start = time.time()
    result = materials.retrieve(query, top_k=5)
    elapsed = (time.time() - start) * 1000
    print(f"  {query}: {elapsed:.1f}ms ({len(result)} chars)")

print("\n3️⃣ 检索性能（缓存命中）")
print("-" * 50)
for query in test_queries:
    start = time.time()
    result = materials.retrieve(query, top_k=5)
    elapsed = (time.time() - start) * 1000
    print(f"  {query}: {elapsed:.1f}ms")

print("\n4️⃣ 缓存统计")
print("-" * 50)
cache_mgr = get_cache_manager()
if cache_mgr:
    stats = cache_mgr.get_stats()
    print(f"  总请求数: {stats['total_requests']}")
    print(f"  缓存命中: {stats['memory_hits']} (内存) + {stats['disk_hits']} (磁盘)")
    print(f"  命中率: {stats['hit_rate']}")
```

### 预期性能指标

| 指标 | Tier 1 | Tier 2 | 提升 |
|------|--------|--------|------|
| 首次查询 | 18ms | 20ms | 仅多 +2ms (排序开销) |
| 缓存命中 | N/A | 2ms | 新增 |
| 缓存命中率 | 0% | 70% | +∞ |
| 平均响应 | 18ms | 5.6ms | -69% |
| 查询召回 | 80% | 95% | +19% |

---

## 🐛 常见问题与排查

### 问题 1: 缓存未生效

**症状:** 每次查询都很慢，日志不显示 "[Cache]"

**排查步骤:**
```python
# 检查缓存是否启用
from 工程重构版V24_4-agent_h import get_cache_manager
cache_mgr = get_cache_manager()
print(f"缓存管理器: {cache_mgr}")

# 检查配置
from 工程重构版V24_4-agent_h import CONF
print(f"ENABLE_CACHE: {getattr(CONF, 'ENABLE_CACHE', None)}")
print(f"CACHE_DIR: {getattr(CONF, 'CACHE_DIR', None)}")
```

**解决方案:** 
- 检查 `ENABLE_CACHE=true`
- 确保 `.cache` 目录可写

### 问题 2: 查询扩展过多或过少

**症状:** 扩展变体数不符合预期

**排查步骤:**
```python
# 调整 MAX_QUERY_VARIANTS
from 工程重构版V24_4-agent_h import CONF
print(f"MAX_QUERY_VARIANTS: {CONF.MAX_QUERY_VARIANTS}")

# 验证同义词库
from 工程重构版V24_4-agent_h import SYNONYMS_DICT
if "你的查询词" in SYNONYMS_DICT:
    print(f"同义词: {SYNONYMS_DICT['你的查询词']}")
else:
    print("该词未在同义词库中")
```

**解决方案:** 
- 增加/减少 `MAX_QUERY_VARIANTS`
- 添加更多同义词到 `SYNONYMS_DICT`

### 问题 3: 日志文件写入失败

**症状:** `query_log.csv` 为空或不存在

**排查步骤:**
```python
from 工程重构版V24_4-agent_h import get_query_analytics
analytics = get_query_analytics()
print(f"日志文件: {analytics.log_file}")
print(f"日志启用: {getattr(CONF, 'ENABLE_QUERY_LOG', None)}")

# 尝试手动写入
analytics.log_query("测试", method="Test")
```

**解决方案:** 
- 检查目录权限
- 确保 `~/Research_Workspace` 可写

---

## ✅ 验收标准

完成以下所有检查，表示 Tier 2 升级成功：

- [ ] ✅ 查询扩展生成 3-5 个变体
- [ ] ✅ 多信号排序分数在 0-1 范围
- [ ] ✅ 首次查询在 15-25ms
- [ ] ✅ 缓存命中在 1-3ms
- [ ] ✅ 缓存命中率 > 50%
- [ ] ✅ 查询日志正常记录
- [ ] ✅ 无 Python 错误或异常
- [ ] ✅ 代码通过 syntax 检查

---

## 📝 测试报告模板

```
## Tier 2 升级测试报告

### 环境信息
- OS: 
- Python 版本: 
- 依赖版本: jieba=?, rank-bm25=?, scikit-learn=?

### 功能测试结果
- [ ] 查询扩展: PASS/FAIL
- [ ] 多信号排序: PASS/FAIL
- [ ] 智能缓存: PASS/FAIL
- [ ] 查询日志: PASS/FAIL
- [ ] 完整集成: PASS/FAIL

### 性能指标
- 首次查询: ___ ms
- 缓存命中: ___ ms
- 加速倍数: ___x
- 缓存命中率: ___%

### 发现的问题
(列举任何问题)

### 结论
PASS / FAIL

签字: ________________
日期: ________________
```

---

## 🎯 后续优化建议

1. **同义词库扩展** (1-2 小时)
   - 收集更多行业专业术语
   - 从查询日志中自动生成常见词对

2. **排序权重微调** (1-2 小时)
   - 基于用户反馈调整 4 个权重参数
   - A/B 测试不同权重组合

3. **缓存策略优化** (2-3 小时)
   - 实现 LRU 淘汰策略
   - 按热度调整 TTL

4. **日志分析仪表板** (3-4 小时)
   - 可视化查询趋势
   - 性能监控实时报警

---

**文档创建日期:** 2025年11月29日  
**状态:** ✅ 完成  
**版本:** V1.0

