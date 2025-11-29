以下为针对脚本 `工程重构版V24_4-agent-h.py` 执行报告的风格指南：

*   **语调与基调**
    *   保持绝对客观的工程交付标准，剔除所有情绪化表达与营销修饰。
    *   聚焦代码执行效能、重构差异及 Agent 行为逻辑，确保信息高密度。

*   **叙事结构**
    *   **环境锚定**：首段明确 `.venv` 解释器路径及依赖库版本，确保环境可复现。
    *   **数据驱动**：优先展示 V24_4 版本的运行时长（Runtime）、内存占用（Memory Footprint）及 Agent 决策准确率。
    *   **逻辑递进**：从执行日志分析到代码重构逻辑（Refactoring Logic），最后推导优化结论。

*   **数据规范**
    *   所有数据必须溯源至本地 Log 文件或 Console Output，严禁估算。
    *   **表格配置**：仅包含一张《重构前后性能对比表》，关键字段：Version, Latency (ms), Throughput (OPS), Success Rate。

*   **语言准则**
    *   零冗余，禁止重复表述同一概念。
    *   量化描述，禁用“快速”、“高效”等模糊形容词，改用“耗时降低 15%”、“吞吐量提升 200%”。
    *   强制使用行业术语：Interpreter Path, Heuristic Algorithm, Dependency Injection, Stack Trace。

*   **图表约束**
    *   仅保留一张核心图：**Agent 收敛曲线图**或**资源消耗趋势图**。
    *   图表描述需在 20 字以内，明确坐标轴定义（如：X=Time, Y=CPU Usage），无多余装饰元素。