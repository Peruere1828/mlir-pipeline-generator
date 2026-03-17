import heapq
from typing import List, Set, Optional, FrozenSet
from definition import MLIRType, Operation, CompilationTarget, MLIRPass

class CompilationState:
    def __init__(self, ops: Set[Operation], types: Set[MLIRType]):
        self.ops: FrozenSet[Operation] = frozenset(ops)
        self.types: FrozenSet[MLIRType] = frozenset(types)

    def __hash__(self):
        return hash((self.ops, self.types))

    def __eq__(self, other):
        if not isinstance(other, CompilationState): return False
        return self.ops == other.ops and self.types == other.types

    def __lt__(self, other): return len(self.ops) < len(other.ops)

    def __repr__(self):
        return f"State(Ops={list(self.ops)})"

    def get_illegal_items(self, target: CompilationTarget) -> int:
        ill_ops = [op for op in self.ops if not target.is_legal_op(op)]
        ill_types = [t for t in self.types if not target.is_legal_type(t)]
        return len(ill_ops) + len(ill_types)

    def is_solved(self, target: CompilationTarget) -> bool:
        return self.get_illegal_items(target) == 0


class KnowledgeBase:
    def __init__(self):
        self.passes: List[MLIRPass] = []

    def register_pass(self, p: MLIRPass):
        self.passes.append(p)

    def get_valid_moves(self, state: CompilationState) -> List[MLIRPass]:
        return [p for p in self.passes if p.is_applicable(state.ops, state.types)]

    def apply_pass(self, state: CompilationState, p: MLIRPass) -> CompilationState:
        # 0. 先应用全局变换（例如 cse / canonicalize）
        working_ops = set(state.ops)
        working_types = set(state.types)
        for transform in p.global_transforms:
            if transform.applicable(working_ops, working_types):
                working_ops, working_types = transform.apply(working_ops, working_types)

        new_ops = set()
        new_types = set(working_types)

        # 1. 执行全局类型转换 (Type Conversion)
        type_changed_map = {}
        for src_t, tgt_t in p.type_conversions.items():
            if src_t in new_types:
                new_types.remove(src_t)
                new_types.add(tgt_t)
                type_changed_map[src_t] = tgt_t

        # 2. 对每个 Operation 进行 Pattern 匹配重写
        for op in working_ops:
            matched_pattern = None
            # 找到第一个命中的规则
            for pattern in p.patterns:
                if pattern.match(op, working_ops, working_types):
                    matched_pattern = pattern
                    break

            if matched_pattern:
                # 规则触发：根据规则生成 opB
                generated_ops = matched_pattern.apply(op)
                for gen_op in generated_ops:
                    # 如果刚才发生了类型转换，需要把生成的 Op 的类型也同步更新
                    # 例如把 arith.add(tensor) 变成了 llvm.add(memref)
                    if type_changed_map:
                        updated_types = {type_changed_map.get(t, t) for t in gen_op.operand_types}
                        gen_op.operand_types = updated_types
                    new_ops.add(gen_op)
            else:
                # 没有规则触发该 Op，原样保留
                # 但要注意：如果全局类型改变了，且该未被重写的 Op 使用了旧类型，我们需要给它换上新类型
                if type_changed_map and any(t in type_changed_map for t in op.operand_types):
                    updated_types = {type_changed_map.get(t, t) for t in op.operand_types}
                    new_ops.add(Operation(op.dialect, op.name, op.traits, updated_types))
                else:
                    new_ops.add(op)

        return CompilationState(new_ops, new_types)


class PipelineSearcher:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def heuristic(self, state: CompilationState, target: CompilationTarget) -> float:
        return float(state.get_illegal_items(target))

    def search(self, start_ops: Set[Operation], start_types: Set[MLIRType], target: CompilationTarget) -> Optional[List[str]]:
        start_state = CompilationState(start_ops, start_types)
        queue = [(self.heuristic(start_state, target), 0.0, start_state, [])]
        visited = {start_state}

        print(f"[Solver] Initial State: {start_state}")
        
        steps = 0
        while queue:
            f, g, current_state, path = heapq.heappop(queue)
            steps += 1

            if current_state.is_solved(target):
                print(f"[Solver] Solution Found in {steps} steps!")
                print(f"   Final State: {current_state}")
                return [p.name for p in path]

            for p in self.kb.get_valid_moves(current_state):
                next_state = self.kb.apply_pass(current_state, p)
                if next_state in visited:
                    continue
                
                visited.add(next_state)
                new_g = g + p.cost
                new_f = new_g + self.heuristic(next_state, target)
                heapq.heappush(queue, (new_f, new_g, next_state, path + [p]))
        
        print("[Solver] No solution found.")
        return None