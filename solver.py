from definition import MLIRType, Operation, CompilationTarget, MLIRPass
from solver_def import PipelineSearcher

# 新增：从文件解析 MLIR
from mlir_parser import MLIRParser
import os

# ==============================================================================
# 从 test.mlir 加载并解析 MLIR，然后搜索 lowering pipeline
# ==============================================================================
from mock_kb import build_mock_kb
if __name__ == "__main__":
    # 解析 test.mlir（位于同一目录）
    base_dir = os.path.dirname(__file__)
    mlir_path = os.path.join(base_dir, "test.mlir")

    parser = MLIRParser()
    try:
        parsed = parser.parse_file(mlir_path)
        start_ops = parsed.get("ops", set())
        start_types = parsed.get("types", set())
    except Exception as e:
        print(f"[Error] 无法解析 {mlir_path}: {e}")
        raise

    # 目标限制: 只允许 LLVM
    target = CompilationTarget()
    for d in ["arith", "linalg", "scf", "affine", "cf", "func", "tosa", "builtin"]:
        target.mark_dialect_illegal(d)
    target.mark_type_illegal("tensor")
    target.mark_type_illegal("memref")

    # 获取分离的 KB
    kb = build_mock_kb()

    searcher = PipelineSearcher(kb)
    pipeline = searcher.search(start_ops, start_types, target)

    print("\n[Generated Pipeline]:")
    print(" -> ".join(pipeline) if pipeline else "Fail")