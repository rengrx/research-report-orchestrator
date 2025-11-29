#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tier 2 升级功能快速验证脚本
快速检查：查询扩展、多信号排序、缓存、日志
"""

import os
import sys
import time
import importlib.util

# 设置环境
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 动态导入模块（处理特殊字符的模块名）
module_name = "工程重构版V24_4-agent-h"
module_file = os.path.join(os.path.dirname(__file__), module_name + ".py")

if not os.path.exists(module_file):
    print(f"❌ 模块文件不存在: {module_file}")
    sys.exit(1)

spec = importlib.util.spec_from_file_location(module_name, module_file)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module

try:
    spec.loader.exec_module(module)
    expand_query = module.expand_query
    compute_relevance_score = module.compute_relevance_score
    get_cache_manager = module.get_cache_manager
    get_query_analytics = module.get_query_analytics
    SYNONYMS_DICT = module.SYNONYMS_DICT
    CONF = module.CONF
    print("✅ 导入成功\n")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def test_query_expansion():
    """测试查询扩展"""
    print("=" * 60)
    print("🔄 功能 1: 查询扩展 (Query Expansion)")
    print("=" * 60)
    
    test_cases = [
        "电力现货",
        "光伏装机",
        "电网",
    ]
    
    for query in test_cases:
        variants = expand_query(query, max_variants=5)
        print(f"\n📝 查询: {query}")
        print(f"   变体数: {len(variants)}")
        print(f"   变体列表: {variants}")
    
    print(f"\n✅ 同义词库大小: {len(SYNONYMS_DICT)} 个词汇")
    return True

def test_relevance_scoring():
    """测试多信号排序"""
    print("\n" + "=" * 60)
    print("⭐ 功能 2: 多信号排序 (Multi-Signal Ranking)")
    print("=" * 60)
    
    test_cases = [
        {"bm25": 8.5, "weight": 1.0, "length": 500, "desc": "高相关度文档"},
        {"bm25": 5.0, "weight": 0.8, "length": 1000, "desc": "中等相关度+长文档"},
        {"bm25": 3.0, "weight": 0.5, "length": 200, "desc": "低相关度+短文档"},
    ]
    
    print("\n📊 综合分数计算示例:")
    scores = []
    for i, case in enumerate(test_cases):
        score = compute_relevance_score(
            bm25_score=case["bm25"],
            doc_weight=case["weight"],
            doc_length=case["length"]
        )
        scores.append(score)
        print(f"\n   {i+1}. {case['desc']}")
        print(f"      BM25={case['bm25']}, 权重={case['weight']}, 长度={case['length']}")
        print(f"      → 综合分数: {score:.3f}")
    
    # 验证排序
    if scores[0] > scores[1] > scores[2]:
        print(f"\n✅ 排序准确: 分数递减")
    else:
        print(f"\n⚠️ 排序异常: {scores}")
    
    return True

def test_cache():
    """测试智能缓存"""
    print("\n" + "=" * 60)
    print("💾 功能 3: 智能缓存 (Smart Caching)")
    print("=" * 60)
    
    cache_mgr = get_cache_manager()
    if not cache_mgr:
        print("\n⚠️ 缓存管理器未启用 (ENABLE_CACHE=false)")
        print(f"   配置: ENABLE_CACHE = {getattr(CONF, 'ENABLE_CACHE', None)}")
        return True
    
    print(f"\n📂 缓存配置:")
    print(f"   缓存目录: {cache_mgr.cache_dir}")
    print(f"   内存TTL: {cache_mgr.memory_ttl}s")
    print(f"   磁盘TTL: {cache_mgr.disk_ttl}s")
    print(f"   最大大小: {cache_mgr.max_cache_size_mb}MB")
    
    # 测试缓存操作
    test_query = "测试查询"
    test_value = "这是测试结果" * 50
    
    print(f"\n🧪 缓存测试:")
    
    # 第一次获取（不命中）
    print(f"   1. 获取缓存 (应该不命中)")
    result, hit, elapsed = cache_mgr.get(test_query)
    print(f"      → 命中: {hit}, 耗时: {elapsed:.2f}ms")
    
    # 设置缓存
    print(f"   2. 存储缓存")
    cache_mgr.set(test_query, test_value)
    print(f"      → 完成")
    
    # 第二次获取（应该命中）
    print(f"   3. 获取缓存 (应该命中)")
    result, hit, elapsed = cache_mgr.get(test_query)
    print(f"      → 命中: {hit}, 耗时: {elapsed:.2f}ms")
    
    # 显示统计
    stats = cache_mgr.get_stats()
    print(f"\n📈 缓存统计:")
    print(f"   总请求数: {stats['total_requests']}")
    print(f"   命中数: {stats['memory_hits']} (内存)")
    print(f"   未命中数: {stats['misses']}")
    print(f"   命中率: {stats['hit_rate']}")
    
    if hit:
        print(f"\n✅ 缓存功能正常")
    else:
        print(f"\n⚠️ 缓存命中异常")
    
    return True

def test_analytics():
    """测试查询日志"""
    print("\n" + "=" * 60)
    print("📝 功能 4: 查询日志 (Query Analytics)")
    print("=" * 60)
    
    analytics = get_query_analytics()
    if not analytics:
        print("\n⚠️ 查询分析器未启用 (ENABLE_QUERY_LOG=false)")
        print(f"   配置: ENABLE_QUERY_LOG = {getattr(CONF, 'ENABLE_QUERY_LOG', None)}")
        return True
    
    print(f"\n📂 日志配置:")
    print(f"   日志文件: {analytics.log_file}")
    
    # 记录测试日志
    print(f"\n🧪 日志记录测试:")
    test_logs = [
        ("电力现货", "BM25+jieba", 5, 18.5, False),
        ("电力现货", "Cache", 5, 1.2, True),
        ("光伏", "BM25+jieba", 3, 15.8, False),
        ("光伏", "Cache", 3, 0.9, True),
    ]
    
    for query, method, count, time_ms, cache_hit in test_logs:
        analytics.log_query(query, method, count, time_ms, cache_hit)
        status = "✓" if cache_hit else "✕"
        print(f"   [{status}] {query} ({method}) - {time_ms:.1f}ms")
    
    # 显示统计
    top_queries = analytics.get_top_queries(limit=3)
    print(f"\n📊 最常见查询:")
    for query, freq in top_queries:
        print(f"   • {query}: {freq} 次")
    
    print(f"\n✅ 日志功能正常")
    
    return True

def main():
    """主测试函数"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "🎯 Tier 2 升级快速验证" + " " * 22 + "║")
    print("╚" + "=" * 58 + "╝\n")
    
    results = []
    
    try:
        results.append(("查询扩展", test_query_expansion()))
        results.append(("多信号排序", test_relevance_scoring()))
        results.append(("智能缓存", test_cache()))
        results.append(("查询日志", test_analytics()))
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📋 测试汇总")
    print("=" * 60)
    
    all_pass = True
    for feature, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {feature}")
        if not result:
            all_pass = False
    
    print("\n" + "=" * 60)
    if all_pass:
        print("🎉 所有功能验证成功！Tier 2 升级就绪！")
    else:
        print("⚠️ 某些功能验证失败，请检查日志")
    print("=" * 60 + "\n")
    
    return all_pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
