import heapq
from typing import List, Set, Optional, FrozenSet
# 导入你在 definition.py 中定义的类
from definition import MLIRType, Operation, CompilationTarget, MLIRPass
from mlir_parser import MLIRParser

# ==============================================================================
# 1. 搜索状态 (Search State)
# ==============================================================================

class CompilationState:
    """
    [The Snapshot] 搜索树的一个节点。
    表示当前代码中包含的 Operation 集合。
    """
    def __init__(self, ops: Set[Operation], types: Set[MLIRType]):
        # 使用 frozenset 确保状态是不可变且可哈希的
        self.ops: FrozenSet[Operation] = frozenset(ops)
        self.types: FrozenSet[MLIRType] = frozenset(types)

    def __hash__(self):
        return hash((self.ops, self.types))

    def __eq__(self, other):
        if not isinstance(other, CompilationState):
            return False
        return self.ops == other.ops and self.types == other.types

    def __lt__(self, other):
        # 仅用于优先队列打破平局，逻辑不重要，保证确定性即可
        # 这里我们倾向于 Ops 数量更少的状态（意味着更精简）
        return len(self.ops) < len(other.ops)

    def __repr__(self):
        return f"State(Ops={len(self.ops)}, Types={list(self.types)})"

    def get_illegal_items(self, target: CompilationTarget) -> int:
        """
        根据 Target 的定义，计算 heuristic 的代价 (Ops + Types)
        这是计算启发函数 h(n) 的基础。
        """
        ill_ops = [op for op in self.ops if not target.is_legal_op(op)]
        ill_types = [t for t in self.types if not target.is_legal_type(t)] # [新增]
        return len(ill_ops) + len(ill_types)

    def is_solved(self, target: CompilationTarget) -> bool:
        return self.get_illegal_items(target) == 0


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
        return [p for p in self.passes if p.is_applicable(state.ops, state.types)]

    def apply_pass(self, state: CompilationState, p: MLIRPass) -> CompilationState:
        # 1. Update Ops
        current_ops = set(state.ops)
        new_ops = set()
        for op in current_ops:
            handled = False
            if op.dialect in p.src_dialects or op in p.src_ops:
                handled = True
            if not handled:
                new_ops.add(op)
        for gen_op in p.tgt_ops:
            new_ops.add(gen_op)

        # 2. Update Types [新增]
        current_types = set(state.types)
        new_types = set()
        
        # 逻辑：如果 Pass 声明处理某种 Type (如 tensor)，我们假设它把所有 tensor 都转化了
        # (对于 Global Pass 如 bufferization 成立)
        for t in current_types:
            if t not in p.src_types:
                new_types.add(t)
        
        # 添加引入的新 Type
        for t in p.tgt_types:
            new_types.add(t)

        return CompilationState(new_ops, new_types)


# ==============================================================================
# 3. 搜索算法 (A*)
# ==============================================================================

class PipelineSearcher:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def heuristic(self, state: CompilationState, target: CompilationTarget) -> float:
        """
        h(n): 估算距离目标的代价。
        策略：剩余 Illegal Op + Illegal Types 的数量。
        """
        return float(state.get_illegal_items(target))

    def search(self, start_ops: Set[Operation], start_types: Set[MLIRType], target: CompilationTarget) -> Optional[List[str]]:
        """
        执行 A* 搜索。
        Args:
            start_ops: 初始的 Operation 集合
            target: 定义合法性的目标 (CompilationTarget)
        Returns:
            List[str]: 找到的 Pass Pipeline (Pass 名称列表) 或 None
        """
        start_state = CompilationState(start_ops, start_types)
        
        # 初始启发值
        start_h = self.heuristic(start_state, target)
        
        # 优先队列: (f_score, g_score, state, path)
        # f = g + h
        queue = [(start_h, 0.0, start_state, [])]
        
        visited = set()
        visited.add(start_state)

        print(f"[Solver] Start Search.")
        print(f"   Initial State: {start_state}")
        print(f"   Target Constraints: {target}")

        steps = 0
        while queue:
            f, g, current_state, path = heapq.heappop(queue)
            steps += 1

            # 1. 检查是否达成目标
            if current_state.is_solved(target):
                print(f"[Solver] Solution Found in {steps} steps! Cost: {g}")
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
        
        print("[Solver] Exhausted search space. No solution found.")
        return None


# ==============================================================================
# 4. Mock 测试 (演示如何使用上述类)
# ==============================================================================

if __name__ == "__main__":
    # 1. Target: LLVM Dialect Only, No Tensors
    target = CompilationTarget()
    target.mark_dialect_illegal("arith")
    target.mark_type_illegal("tensor")
    target.mark_type_illegal("memref")

    # 2. Ops & Types
    start_ops = {Operation("arith", "constant")} # %0 = arith.constant ... : tensor<...>
    start_types = {MLIRType("tensor")}

    # 3. KB
    kb = KnowledgeBase()

    # Pass: Bufferization
    # 它的作用是将 Tensor 类型转化为 MemRef 类型
    p_buf = MLIRPass("one-shot-bufferize")
    p_buf.add_source(MLIRType("tensor")) # 处理 tensor
    p_buf.add_target(MLIRType("memref")) # 产生 memref
    kb.register_pass(p_buf)

    # Pass: Memref to LLVM
    p_mem = MLIRPass("finalize-memref-to-llvm")
    p_mem.add_source(MLIRType("memref")) # 处理 memref
    # 它可能也会产生 llvm dialect 的 op，此处省略
    kb.register_pass(p_mem)
    
    # Pass: Arith to LLVM
    p_arith = MLIRPass("arith-to-llvm")
    p_arith.add_source("arith")
    kb.register_pass(p_arith)

    # 4. Search
    searcher = PipelineSearcher(kb)
    pipeline = searcher.search(start_ops, start_types, target)

    # 预期路径: one-shot-bufferize (消 tensor) -> finalize-memref-to-llvm (消 memref) -> arith-to-llvm (消 op)
    print(" -> ".join(pipeline) if pipeline else "Fail")
