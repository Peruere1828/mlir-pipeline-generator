import re
import os
from typing import Set, List, Dict, Tuple
from config import OFFICIAL_DIALECT_NAMESPACES
from definition import Operation, MLIRType
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

    def parse_content(self, content: str) -> Dict[str, object]:
        cleaned_content = self._remove_comments(content)
        
        # 1. Parse Ops (保持不变)
        found_ops_dict = defaultdict(set)
        op_pattern = re.compile(r'\b([\w_]+)\.([\w_]+)\b')
        op_matches = op_pattern.findall(cleaned_content)
        for dialect, op_name in op_matches:
            if dialect in self.whitelist:
                op = Operation(dialect=dialect, name=op_name)
                found_ops_dict[dialect].add(op)

        # 2. Parse Types [新增]
        # 简单的启发式搜索：查找 builtin 容器类型关键字
        # 匹配 pattern: tensor<, memref<, vector<
        found_types = set()
        type_keywords = ['tensor', 'memref', 'vector']
        
        for kw in type_keywords:
            # 搜索 "tensor<" 这种模式
            if re.search(rf'\b{kw}\s*<', cleaned_content):
                found_types.add(MLIRType(kw))

        # 3. 整合返回
        # 扁平化 Ops 集合
        all_ops = set()
        for ops in found_ops_dict.values():
            all_ops.update(ops)

        return {
            "ops": all_ops,
            "types": found_types
        }

    def parse_file(self, file_path: str) -> Dict[str, object]:
        """
        读取文件并解析
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return self.parse_content(content)