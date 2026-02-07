import heapq
from typing import List, Set, Optional, FrozenSet
# 导入你在 definition.py 中定义的类
from definition import Operation, CompilationTarget, MLIRPass
from mlir_parser import MLIRParser

# ==============================================================================
# 1. 搜索状态 (Search State)
# ==============================================================================

class CompilationState:
    """
    [The Snapshot] 搜索树的一个节点。
    表示当前代码中包含的 Operation 集合。
    """
    def __init__(self, ops: Set[Operation]):
        # 使用 frozenset 确保状态是不可变且可哈希的
        self.ops: FrozenSet[Operation] = frozenset(ops)

    def __hash__(self):
        return hash(self.ops)

    def __eq__(self, other):
        if not isinstance(other, CompilationState):
            return False
        return self.ops == other.ops

    def __lt__(self, other):
        # 仅用于优先队列打破平局，逻辑不重要，保证确定性即可
        # 这里我们倾向于 Ops 数量更少的状态（意味着更精简）
        return len(self.ops) < len(other.ops)

    def __repr__(self):
        # 简单的字符串表示，方便调试
        op_strs = [op.full_name for op in self.ops]
        return f"State({sorted(op_strs)})"

    def get_illegal_ops(self, target: CompilationTarget) -> List[Operation]:
        """
        根据 Target 的定义，找出当前状态中所有非法的 Op。
        这是计算启发函数 h(n) 的基础。
        """
        return [op for op in self.ops if not target.is_legal(op)]

    def is_solved(self, target: CompilationTarget) -> bool:
        """检查是否达成目标（即没有 Illegal Ops）"""
        return len(self.get_illegal_ops(target)) == 0


# ==============================================================================
# 2. 知识库 (Knowledge Base)
# ==============================================================================

class KnowledgeBase:
    """
    管理所有注册的 Pass 以及状态转移逻辑。
    """
    def __init__(self):
        self.passes: List[MLIRPass] = []

    def register_pass(self, p: MLIRPass):
        self.passes.append(p)

    def get_valid_moves(self, state: CompilationState) -> List[MLIRPass]:
        """
        返回所有适用于当前 State 的 Pass。
        复用 MLIRPass.is_applicable 方法。
        """
        # 优化策略：我们只应该尝试那些能处理当前 "Illegal Ops" 的 Pass。
        # 如果一个 Pass 的输入全是合法的 Op，通常没必要跑它（除非是优化 Pass，目前暂不考虑）。
        return [p for p in self.passes if p.is_applicable(state.ops)]

    def apply_pass(self, state: CompilationState, p: MLIRPass) -> CompilationState:
        """
        [Core Logic] 模拟 Pass 执行后的状态变化。
        New State = (Old State - Handled Ops) + Generated Ops
        """
        current_ops = set(state.ops)
        new_ops = set()
        
        # 1. 移除被此 Pass 处理掉的 Ops
        for op in current_ops:
            handled = False
            
            # 检查: 该 Pass 是否声明处理整个 Dialect
            if op.dialect in p.src_dialects:
                handled = True
            # 检查: 该 Pass 是否声明处理特定 Op
            elif op in p.src_ops:
                handled = True
            
            if not handled:
                new_ops.add(op) # 没被处理，保留
        
        # 2. 添加此 Pass 产生的新 Ops
        # 注意：definition.py 中 MLIRPass 也有 tgt_dialects。
        # 但 State 需要具体的 Operation 对象。在 Mock/Search 阶段，
        # 我们主要依赖 pass.tgt_ops 来模拟产生的新操作。
        for gen_op in p.tgt_ops:
            new_ops.add(gen_op)
            
        return CompilationState(new_ops)


# ==============================================================================
# 3. 搜索算法 (A*)
# ==============================================================================

class PipelineSearcher:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def heuristic(self, state: CompilationState, target: CompilationTarget) -> float:
        """
        h(n): 估算距离目标的代价。
        策略：剩余 Illegal Op 的数量。
        """
        illegal_ops = state.get_illegal_ops(target)
        return float(len(illegal_ops))

    def search(self, start_ops: Set[Operation], target: CompilationTarget) -> Optional[List[str]]:
        """
        执行 A* 搜索。
        Args:
            start_ops: 初始的 Operation 集合
            target: 定义合法性的目标 (CompilationTarget)
        Returns:
            List[str]: 找到的 Pass Pipeline (Pass 名称列表) 或 None
        """
        start_state = CompilationState(start_ops)
        
        # 初始启发值
        start_h = self.heuristic(start_state, target)
        
        # 优先队列: (f_score, g_score, state, path)
        # f = g + h
        queue = [(start_h, 0.0, start_state, [])]
        
        visited = set()
        visited.add(start_state)

        print(f"🔎 [Solver] Start Search.")
        print(f"   Initial State: {start_state}")
        print(f"   Target Constraints: {target}")

        steps = 0
        while queue:
            f, g, current_state, path = heapq.heappop(queue)
            steps += 1

            # 1. 检查是否达成目标
            if current_state.is_solved(target):
                print(f"✅ [Solver] Solution Found in {steps} steps! Cost: {g}")
                print(f"   Final State: {current_state}")
                return [p.name for p in path]

            # 2. 扩展节点 (Expand)
            valid_passes = self.kb.get_valid_moves(current_state)
            
            for p in valid_passes:
                next_state = self.kb.apply_pass(current_state, p)
                
                if next_state in visited:
                    continue
                
                # A* 计算
                new_g = g + p.cost
                new_h = self.heuristic(next_state, target)
                new_f = new_g + new_h
                
                # 记录路径
                new_path = path + [p]
                
                visited.add(next_state)
                heapq.heappush(queue, (new_f, new_g, next_state, new_path))
        
        print("❌ [Solver] Exhausted search space. No solution found.")
        return None


# ==============================================================================
# 4. Mock 测试 (演示如何使用上述类)
# ==============================================================================

def test_integration_with_parser():
    """
    测试 1: 集成 MLIRParser，演示从文本到 Pipeline 的全过程。
    """
    print("\n=== Test 1: Integration with MLIRParser ===")
    
    # 模拟一段 MLIR 代码
    mlir_code = """
func.func private @callee(%arg0: memref<f32>) -> memref<f32> {
  %2 = arith.constant 2.0 : f32
  memref.load %arg0[] {tag = "call_and_store_before::enter_callee"} : memref<f32>
  memref.store %2, %arg0[] {tag_name = "callee"} : memref<f32>
  memref.load %arg0[] {tag = "exit_callee"} : memref<f32>
  return %arg0 : memref<f32>
}
    """
    
    # 1. 解析
    parser = MLIRParser()
    parsed_dict = parser.parse_content(mlir_code)
    
    # 2. 扁平化: Dict[str, Set[Op]] -> Set[Op]
    start_ops = set()
    for ops in parsed_dict.values():
        start_ops.update(ops)
    
    # 3. 设置目标 (Target)
    target = CompilationTarget()
    # 标记非 LLVM 为 Illegal (简化版逻辑)
    target.mark_dialect_illegal("func")
    target.mark_dialect_illegal("arith")
    target.mark_dialect_illegal("linalg")
    target.mark_dialect_illegal("scf")
    # target.mark_dialect_legal("llvm") # 默认其它如果没标记为 illegal，且不在 legal 列表... 
    # (依赖 CompilationTarget 具体的 is_legal 实现逻辑，这里假设未标记 Illegal 且没 Legal Set 限制时会有问题，
    # 建议配合 definition.py 里的逻辑：只要 dialect 在 illegal_dialects 里就是非法)

    # 4. 构建 KB (模拟 Full Conversion)
    kb = KnowledgeBase()
    
    p1 = MLIRPass("convert-linalg-to-loops")
    p1.add_source("linalg") # 处理所有 linalg
    p1.add_target(Operation("scf", "for"))
    kb.register_pass(p1)

    p2 = MLIRPass("convert-scf-to-cf")
    p2.add_source("scf")
    p2.add_target(Operation("cf", "br"))
    kb.register_pass(p2)

    p3 = MLIRPass("arith-to-llvm")
    p3.add_source("arith")
    p3.add_target(Operation("llvm", "add"))
    kb.register_pass(p3)

    p4 = MLIRPass("convert-cf-to-llvm")
    p4.add_source("cf")
    p4.add_target(Operation("llvm", "br"))
    kb.register_pass(p4)
    
    p5 = MLIRPass("func-to-llvm")
    p5.add_source("func")
    p5.add_target(Operation("llvm", "func"))
    kb.register_pass(p5)

    # 5. 搜索
    searcher = PipelineSearcher(kb)
    pipeline = searcher.search(start_ops, target)
    if pipeline:
        print("Pipeline:", " -> ".join(pipeline))


def test_partial_conversion():
    """
    测试 2: 演示 Partial Conversion (部分转换)。
    场景：Vector Dialect。
    - Pass A 只处理 vector.transfer_read
    - Pass B 只处理 vector.transpose
    """
    print("\n=== Test 2: Partial Conversion (Vector Dialect) ===")

    # 1. 初始 Ops: 包含 vector.transfer_read 和 vector.transpose
    start_ops = {
        Operation("vector", "transfer_read"),
        Operation("vector", "transpose"),
        Operation("func", "return")
    }

    # 2. 目标
    target = CompilationTarget()
    target.mark_dialect_illegal("vector") # 整个 vector 都不想要
    
    # 3. 构建 KB
    kb = KnowledgeBase()

    # Pass A: 这是一个 Partial Pass
    # 它只声明处理 vector.transfer_read，而不处理整个 vector dialect
    p_transfer = MLIRPass("convert-vector-transfer-to-scf")
    p_transfer.add_source(Operation("vector", "transfer_read")) 
    p_transfer.add_target(Operation("scf", "for"))
    kb.register_pass(p_transfer)

    # Pass B: 这是另一个 Partial Pass
    p_transpose = MLIRPass("vector-transpose-to-llvm") # 假设直接降级
    p_transpose.add_source(Operation("vector", "transpose"))
    p_transpose.add_target(Operation("llvm", "matrix_intrinsics")) # 模拟
    kb.register_pass(p_transpose)

    # 辅助 Pass
    p_scf = MLIRPass("scf-to-cf")
    p_scf.add_source("scf")
    p_scf.add_target(Operation("cf", "br"))
    kb.register_pass(p_scf)

    # 4. 搜索
    searcher = PipelineSearcher(kb)
    pipeline = searcher.search(start_ops, target)
    
    if pipeline:
        print("Pipeline:", " -> ".join(pipeline))
        # 预期：算法必须同时选出 p_transfer 和 p_transpose 才能消除所有的 vector op
    else:
        print("Failed to find pipeline!")

if __name__ == "__main__":
    test_integration_with_parser()
    test_partial_conversion()