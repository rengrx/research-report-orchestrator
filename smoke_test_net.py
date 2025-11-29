import os
import json
import hashlib
import time
import requests
import logging

# --- 模拟配置 (请确保与你真实环境一致) ---
# 1. 获取 API Key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 2. 配置代理 (如果有)
# 如果你需要代理，请取消注释并填入，否则保持为 None (模拟 Config.PROXIES_CLOUD)
PROXIES = None 
# PROXIES = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"} 

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_search_robust(query):
    logging.info(f"🧪 开始网络冒烟测试: Query='{query}'")
    
    # 1. 检查 Key
    if not TAVILY_API_KEY:
        logging.error("❌ 未检测到 TAVILY_API_KEY！请先 export TAVILY_API_KEY=Tvly-xxxx")
        return

    # 2. 计算 MD5 (验证哈希逻辑)
    query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
    cache_dir = os.path.expanduser("~/mineru/workflow/search_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{query_hash}.json")
    logging.info(f"📂 预期缓存路径: {cache_path}")

    # 3. 清理旧缓存 (为了测试真实网络请求)
    if os.path.exists(cache_path):
        os.remove(cache_path)
        logging.info("🧹 已清除旧缓存，强制发起网络请求...")

    # 4. 发起请求 (验证代理和 Tavily 连通性)
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "max_results": 2
    }

    try:
        start_time = time.time()
        logging.info("🌐 发送请求到 Tavily...")
        
        # 模拟 retry 逻辑中的一次请求
        response = requests.post(
            url, 
            json=payload, 
            headers={"Content-Type": "application/json"},
            proxies=PROXIES, # 测试代理
            timeout=15
        )
        
        if response.status_code == 200:
            duration = time.time() - start_time
            data = response.json()
            answer = data.get("answer", "无摘要")
            logging.info(f"✅ 网络请求成功! (耗时: {duration:.2f}s)")
            logging.info(f"📄 返回摘要: {answer[:50]}...")
            
            # 5. 写入缓存 (验证写权限)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            logging.info("💾 缓存写入成功")
            
            # 6. 二次读取 (验证缓存命中逻辑)
            if os.path.exists(cache_path):
                logging.info("🔍 验证缓存文件存在: 是")
            else:
                logging.error("❌ 缓存文件未生成！")
                
        elif response.status_code == 403:
            logging.error("❌ API Key 无效 (403 Forbidden)")
        else:
            logging.error(f"❌ 请求失败: Status {response.status_code} - {response.text}")

    except Exception as e:
        logging.error(f"❌ 网络连接异常 (可能是代理配置错误): {e}")

if __name__ == "__main__":
    # 测试一个永远不会变的热点词，或者带时间戳的词以确保结果新鲜
    test_search_robust("DeepSeek-R1 technical report analysis")