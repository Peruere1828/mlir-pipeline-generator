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
        new_ops = set()
        new_types = set(state.types)
        
        # 1. 更新全局 Types
        for t in p.src_types:
            if t in new_types: new_types.remove(t)
        for t in p.tgt_types:
            new_types.add(t)

        # 2. 更新 Ops
        for op in state.ops:
            handled = False
            
            # 判断该 Op 是否被作为 Source "吃掉" (Consumed)
            if op.dialect in p.src_dialects:
                if op.dialect in p.op_type_requirements:
                    if p.op_type_requirements[op.dialect] in op.operand_types:
                        handled = True
                else:
                    handled = True

            if p.src_traits.intersection(op.traits):
                if "trait_target" in p.op_type_requirements:
                    if p.op_type_requirements["trait_target"] in op.operand_types:
                        handled = True
                else:
                    handled = True

            # Also consider explicit src_ops
            if op in p.src_ops:
                handled = True

            if not handled:
                # 如果没被吃掉，检查它的 Type 是否被转换了（比如 tensor -> memref）
                current_op_types = set(op.operand_types)
                changed = False
                for st in p.src_types:
                    if st in current_op_types:
                        current_op_types.remove(st)
                        changed = True
                if changed:
                    for tt in p.tgt_types:
                        current_op_types.add(tt)
                    # 生成 Type 被修改后的新 Op
                    new_ops.add(Operation(op.dialect, op.name, op.traits, current_op_types))
                else:
                    # 原样保留
                    new_ops.add(op)

        # 3. 添加 Pass 生成的新 Ops
        for gen_op in p.tgt_ops:
            new_ops.add(gen_op)
            
        # 简单模拟：如果 Pass 声明会生成某 dialect，粗略塞一个 generic op 进去代表状态
        for gen_dialect in p.tgt_dialects:
            new_ops.add(Operation(gen_dialect, "generic", operand_types=new_types))

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