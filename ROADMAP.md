# MLIR Pipeline Searcher 细化路线图

## 1) Pass 建模细化
- **全局 pass 分类**：把 pass 分成 conversion / canonical / cleanup / target-specific 四类，在搜索时可加不同权重。
- **模式粒度**：从 dialect 级扩展到 op 级，再扩展到 trait/type 条件组合（例如仅对 `ElementwiseMappable + tensor` 生效）。
- **副作用建模**：记录 pass 是否引入循环、是否可能增大 IR（便于 A* 代价函数更真实）。

## 2) 搜索策略细化
- **多目标代价**：总代价 = pass 成本 + IR 膨胀惩罚 + target 不匹配惩罚。
- **阶段约束**：引入 stage（high/mid/low），避免过早执行 `*-to-llvm`。
- **可解释性输出**：每一步输出“为什么可用/为什么被拒绝”，便于毕设展示。

## 3) Knowledge Base 构建细化
- **自动导入链路**：TableGen -> 结构化 pass metadata -> 校验 -> 写入 knowledge base。
- **数据质量控制**：给每个导入 pass 打置信度分数（regex 提取低、LLM+规则交叉验证高）。
- **版本化**：按 MLIR 版本切分 KB（例如 llvm-project 17/18/19）。

## 4) 评测与可视化
- **benchmark 套件**：准备 10~20 个代表性 IR（tosa/linalg/vector/scf）。
- **指标**：求解成功率、平均 pipeline 长度、运行时间、非法节点下降曲线。
- **可视化**：把状态图导出为 dot/mermaid，用于论文图示。

## 5) 工程化建议
- 为 pass schema 提供 JSON Schema，防止导入脏数据。
- 增加单元测试：pattern 匹配、global pass、生效条件、A* 回归。
- 增加 CLI：`search`, `import-pass`, `validate-kb`, `benchmark`。
