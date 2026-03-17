from typing import Optional, Set, List, Callable, Dict, Union, Tuple

class MLIRType:
    def __init__(self, name: str):
        self.name = name  # e.g., 'tensor', 'memref', 'vector'

    def __repr__(self):
        return f"Type({self.name})"

    def __eq__(self, other):
        return isinstance(other, MLIRType) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Operation:
    """
    表示一个 MLIR Operation 类型。
    例如: 对于 'linalg.generic', dialect='linalg', name='generic'
    """
    def __init__(self, dialect: str, name: str,
                 traits: Optional[Set[str]] = None,
                 operand_types: Optional[Set[MLIRType]] = None):
        self.dialect = dialect
        self.name = name  # 仅保留后缀，例如 'matmul'
        self.traits = traits if traits else set()
        self.operand_types = operand_types if operand_types else set()

    @property
    def full_name(self) -> str:
        """返回全名，如 'linalg.matmul'"""
        return f"{self.dialect}.{self.name}"

    def __repr__(self):
        types_str = ",".join(t.name for t in self.operand_types)
        traits_str = f"[{','.join(self.traits)}]" if self.traits else ""
        return f"<{self.full_name}{traits_str}({types_str})>"

    def __eq__(self, other):
        """现在判断两个 Op 是否相同，要求dialect、opname、optypes相同"""
        if not isinstance(other, Operation):
            return False
        return (self.dialect == other.dialect and
                self.name == other.name and
                self.operand_types == other.operand_types)

    def __hash__(self):
        """
        实现 Hash，以便可以放入 Set 中去重。
        """
        return hash((self.dialect, self.name, frozenset(self.operand_types)))


class CompilationTarget:
    """
    对应 MLIR 中的 `ConversionTarget` 概念。
    它定义了当前编译阶段的"合法性标准"。

    Logic:
    1. Open World Assumption: 默认所有 Dialect 和 Op 都是 Legal 的。
    2. 如果 mark_dialect_illegal(d): 该 Dialect 下的所有 Op 变为 Illegal。
    3. 如果 mark_op_illegal(op): 仅该 Op 变为 Illegal (覆盖默认)。
    4. (未来扩展) mark_op_legal(op): 可以覆盖 Dialect 的 illegal 设置 (Specific overrides generic)。
    """
    def __init__(self):
        self._illegal_dialects: Set[str] = set()
        self._illegal_ops: Set[Operation] = set()

        # 预留：有些 Op 即使 dialect illegal 也是合法的 (White-listing)
        self._legal_ops: Set[Operation] = set()

        self._illegal_types: Set[MLIRType] = set()

    def mark_dialect_illegal(self, dialect_name: str):
        """声明某个 Dialect 非法（需要被 Lowering）"""
        self._illegal_dialects.add(dialect_name)

    def mark_op_illegal(self, op: Operation):
        """声明某个具体 Op 非法"""
        self._illegal_ops.add(op)

    def mark_op_legal(self, op: Operation):
        """声明某个具体 Op 合法 (用于覆盖 Dialect 级别的 Illegal)"""
        self._legal_ops.add(op)

    def mark_type_illegal(self, type_name: str):
        """声明某种类型是非法的 (例如 tensor 在 llvm target 中非法)"""
        self._illegal_types.add(MLIRType(type_name))

    def is_legal_op(self, op: Operation) -> bool:
        """
        判断某个 Op 在当前 State 下是否合法。
        判定优先级: Explicit Legal Op > Explicit Illegal Op > Explicit Illegal Dialect > Default Legal
        """
        # 1. 白名单优先 (Specific Overrides)
        if op in self._legal_ops:
            return True

        # 2. 黑名单：具体 Op
        if op in self._illegal_ops:
            return False

        # 3. 黑名单：Dialect
        if op.dialect in self._illegal_dialects:
            return False

        # 4. 默认合法
        return True

    def is_legal_type(self, type_obj: MLIRType) -> bool:
        return type_obj not in self._illegal_types

    def __repr__(self):
        return f"<Target: illegal_dialects={len(self._illegal_dialects)}, illegal_ops={len(self._illegal_ops)}, illegal_types={len(self._illegal_types)}>"


class RewritePattern:
    """
    表示一个从 1个源Op 到 N个目标Op 的转换规则。
    src_dialect 支持通配符 "*"，用于表达全局应用 pass（例如 canonicalize/cse）。
    """
    def __init__(self,
                 src_dialect: str,
                 src_name: Optional[str] = None,
                 generated_targets: Optional[List[Tuple[str, str]]] = None,
                 condition: Optional[Callable[['Operation', Set['Operation'], Set['MLIRType']], bool]] = None):

        self.src_dialect = src_dialect
        self.src_name = src_name
        self.generated_targets = generated_targets if generated_targets is not None else []
        self.condition = condition

    def match(self, op: 'Operation', current_ops: Set['Operation'], current_types: Set['MLIRType']) -> bool:
        if self.src_dialect != "*" and op.dialect != self.src_dialect:
            return False
        if self.src_name and op.name != self.src_name:
            return False
        if self.condition and not self.condition(op, current_ops, current_types):
            return False
        return True

    def apply(self, op: 'Operation') -> List['Operation']:
        """应用转换：生成一组新的 Operation"""
        results = []
        for d_name, o_name in self.generated_targets:
            # 新生成的 Op 暂时继承原 Op 的 traits 和 operand_types (在宏观特征集模型下足够用)
            results.append(Operation(d_name, o_name, op.traits, set(op.operand_types)))
        return results


class GlobalTransform:
    """描述对整个 IR 状态生效的 Pass 动作。"""

    def __init__(
        self,
        name: str,
        is_applicable: Callable[[Set['Operation'], Set['MLIRType']], bool],
        transform: Callable[[Set['Operation'], Set['MLIRType']], Tuple[Set['Operation'], Set['MLIRType']]],
    ):
        self.name = name
        self._is_applicable = is_applicable
        self._transform = transform

    def applicable(self, ops: Set['Operation'], types: Set['MLIRType']) -> bool:
        return self._is_applicable(ops, types)

    def apply(self, ops: Set['Operation'], types: Set['MLIRType']) -> Tuple[Set['Operation'], Set['MLIRType']]:
        return self._transform(ops, types)


class MLIRPass:
    """
    一个 Pass 本质上就是一组 Pattern 规则的集合，外加一些全局的类型转换策略。
    """
    def __init__(self, name: str, cost: float = 1.0):
        self.name = name
        self.cost = cost

        # 存放 opA -> opB 的规则集
        self.patterns: List[RewritePattern] = []

        # 存放全局类型转换 (例如 bufferization 中的 tensor -> memref)
        self.type_conversions: Dict[MLIRType, MLIRType] = {}

        # 存放全局转换 (例如 cse / canonicalize 这样的全局应用 pass)
        self.global_transforms: List[GlobalTransform] = []

    def add_pattern(self, pattern: RewritePattern):
        self.patterns.append(pattern)

    def add_type_conversion(self, src_type: Union[str, MLIRType], tgt_type: Union[str, MLIRType]):
        if isinstance(src_type, str):
            src_type = MLIRType(src_type)
        if isinstance(tgt_type, str):
            tgt_type = MLIRType(tgt_type)
        self.type_conversions[src_type] = tgt_type

    def add_global_transform(self, transform: GlobalTransform):
        self.global_transforms.append(transform)

    def is_applicable(self, ops: Set['Operation'], types: Set['MLIRType']) -> bool:
        """
        只要有任何一个全局类型需要被转换，或者有任何一个 Op 能命中规则，就可以应用该 Pass。
        """
        for transform in self.global_transforms:
            if transform.applicable(ops, types):
                return True

        for src_t in self.type_conversions.keys():
            if src_t in types:
                return True

        for op in ops:
            for pattern in self.patterns:
                if pattern.match(op, ops, types):
                    return True
        return False

    def __repr__(self):
        return f"Pass({self.name})"
