import re
import os
from typing import Set, List, Dict, Tuple
from config import OFFICIAL_DIALECT_NAMESPACES
from definition import Operation
from collections import defaultdict

class MLIRParser:
    def __init__(self):
        self.whitelist = OFFICIAL_DIALECT_NAMESPACES

    def _remove_comments(self, content: str) -> str:
        """
        去除 MLIR 代码中的注释 (// ...)
        """
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            # 找到第一个 // 的位置，保留它之前的内容
            comment_start = line.find('//')
            if comment_start != -1:
                line = line[:comment_start]
            if line.strip(): # 只保留非空行
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    def parse_content(self, content: str) -> Dict[str, Set[Operation]]:
        """
        解析 MLIR 内容。
        
        Returns:
            Dict[str, Set[Operation]]: 
            Key 是 dialect 名称 (如 'linalg'), 
            Value 是该 Dialect 下出现的所有 Operation 对象集合。
        """
        cleaned_content = self._remove_comments(content)
        
        # 使用 defaultdict 简化字典初始化逻辑
        result = defaultdict(set)

        # 正则保持不变
        pattern = re.compile(r'\b([\w_]+)\.([\w_]+)\b')
        matches = pattern.findall(cleaned_content)

        for dialect, op_name in matches:
            # 1. 白名单过滤
            if dialect in self.whitelist:
                # 2. 创建 Operation 对象
                op = Operation(dialect=dialect, name=op_name)
                
                # 3. 存入结果 (由于是 set，自动去重)
                result[dialect].add(op)

        # 转换为普通 dict 返回 (可选，看个人喜好)
        return dict(result)

    def parse_file(self, file_path: str) -> Dict[str, Set[Operation]]:
        """
        读取文件并解析
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return self.parse_content(content)

# ==============================================================================
# Usage Example
# ==============================================================================
if __name__ == "__main__":
    # 模拟输入：包含重复的 linalg.generic 和不同的 arith op
    code = """
    func.func @test() {
        %0 = arith.constant 1 : i32
        %1 = arith.constant 2 : i32  // 重复的 arith.constant
        %2 = linalg.generic ...      // 第一次出现
        %3 = linalg.generic ...      // 第二次出现 (应该被去重)
        %4 = linalg.matmul ...       // 新的 linalg op
    }
    """
    
    parser = MLIRParser()
    parsed_data = parser.parse_content(code)
    
    print("=== Parsed Result ===")
    for dialect, ops in parsed_data.items():
        print(f"Dialect: [{dialect}]")
        for op in ops:
            # 这里调用的是 Operation.__repr__
            print(f"  └── {op} (Name: {op.name})")
            
    # 验证去重逻辑
    linalg_ops = parsed_data['linalg']
    print(f"\nTotal Linalg Ops Detected: {len(linalg_ops)}") 
    # 应该输出 2 (generic 和 matmul)，尽管 generic 在代码里出现了两次