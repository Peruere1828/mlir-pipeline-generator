from typing import Optional, Set, Union

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
    def __init__(self, dialect: str, name: str, attributes: Optional[dict] = None):
        self.dialect = dialect
        self.name = name  # 仅保留后缀，例如 'matmul'
        self.attributes = attributes if attributes else {} # 预留字段，暂时为空
    
    @property
    def full_name(self) -> str:
        """返回全名，如 'linalg.matmul'"""
        return f"{self.dialect}.{self.name}"

    def __repr__(self):
        return f"<Op: {self.full_name}>"

    def __eq__(self, other):
        """
        定义相等性：如果 Dialect 和 Name 都相同，则认为是同一个 Op 类型。
        忽略 attributes 的差异（因为我们只关心 Op 类型是否存在）。
        """
        if not isinstance(other, Operation):
            return False
        return self.dialect == other.dialect and self.name == other.name

    def __hash__(self):
        """
        实现 Hash，以便可以放入 Set 中去重。
        """
        return hash((self.dialect, self.name))
    
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
        return f"<Target: illegal_dialects={len(self._illegal_dialects)}, illegal_ops={len(self._illegal_ops)}>, illegal_types={len(self._illegal_types)}>"
    
class MLIRPass:
    """
    表示一个 Conversion Pass。
    它定义了它能处理什么 (Source/Illegal) 以及它会生成什么 (Target/Legal)。
    """
    def __init__(self, name: str, cost: float = 1.0):
        self.name = name
        self.cost = cost
        
        # 该 Pass 能够处理（消除）的来源
        self.src_dialects: Set[str] = set()
        self.src_ops: Set[Operation] = set()
        
        # 该 Pass 会引入（生成）的目标
        self.tgt_dialects: Set[str] = set()
        self.tgt_ops: Set[Operation] = set()

        # [新增] 该 Pass 能消除的类型 (例如 bufferize 消除 tensor)
        self.src_types: Set[MLIRType] = set()
        # [新增] 该 Pass 引入的类型 (例如 bufferize 引入 memref)
        self.tgt_types: Set[MLIRType] = set()

    def add_source(self, item: Union[str, Operation, MLIRType]):
        """定义该 Pass 能'消化'什么"""
        if isinstance(item, str):
            self.src_dialects.add(item)
        elif isinstance(item, Operation):
            self.src_ops.add(item)
        elif isinstance(item, MLIRType):
            self.src_types.add(item)

    def add_target(self, item: Union[str, Operation, MLIRType]):
        """定义该 Pass 能'生产'什么"""
        if isinstance(item, str):
            self.tgt_dialects.add(item)
        elif isinstance(item, Operation):
            self.tgt_ops.add(item)
        elif isinstance(item, MLIRType):
            self.tgt_types.add(item)
            
    def is_applicable(self, current_ops: Set[Operation], current_types: Set[MLIRType]) -> bool:
        """
        判断该 Pass 是否适用于当前的 Op 集合。
        逻辑：如果 current_ops 里存在该 Pass 的 source (dialects 或 ops)，则适用。
        """
        for op in current_ops:
            # 检查 Dialect 匹配
            if op.dialect in self.src_dialects:
                return True
            # 检查 Op 匹配
            if op in self.src_ops:
                return True
            
        for t in current_types:
            if t in self.src_types:
                return True
        return False

    def __repr__(self):
        return f"Pass({self.name})"