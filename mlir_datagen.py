import os
import re
import json
import shutil

# ================= 配置区域 =================

# MLIR 官方测试文件夹的绝对路径 (请修改这里)
MLIR_TEST_DIR = r"/home/ubuntuaaa/projects/llvm-project/mlir/test"

# 输出结果的文件夹绝对路径 (会自动创建)
TARGET_DIR = r"/home/ubuntuaaa/projects/mlir-pipeline-generator/testcases/official_cases"

# 是否要跳过包含 expected-error 的负面测试？ (建议 True，因为你关注的是正确的 pipeline)
SKIP_EXPECTED_ERROR = True

# ===========================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sanitize_filename(name):
    """将路径字符串转换为合法的文件名"""
    # 替换路径分隔符和非法字符
    name = name.replace(os.sep, '_').replace('/', '_').replace('\\', '_')
    return re.sub(r'[^\w\.\-]', '_', name)

from config import OFFICIAL_DIALECT_NAMESPACES

def extract_dialects(content):
    """
    使用正则提取内容中涉及的 dialect，并使用官方白名单过滤。
    """
    # 1. 提取所有形如 "word.word" 的潜在操作符
    # 这一步是为了捕获像 arith.constant, linalg.matmul 这样的结构
    # 正则解释：
    # [\w]+  : 匹配 Dialect 部分 (字母、数字、下划线)
    # \.     : 匹配点号
    # [\w]+  : 匹配 Op Name 部分
    potential_matches = re.findall(r'([\w]+)\.[\w]+', content)
    
    # 2. 过滤：只保留在白名单中的 Dialect
    # 使用 set 去重
    valid_dialects = set()
    for namespace in potential_matches:
        if namespace in OFFICIAL_DIALECT_NAMESPACES:
            valid_dialects.add(namespace)
            
    # 3. 排序并返回列表
    return sorted(list(valid_dialects))


def parse_run_commands(content):
    """
    解析 // RUN: 后的内容，处理 bash 换行符 \
    """
    solutions = []
    lines = content.split('\n')
    
    current_command = ""
    in_command = False
    
    for line in lines:
        stripped = line.strip()
        
        # 检查是否是 RUN 行
        if stripped.startswith('//') and 'RUN:' in stripped:
            # 提取 RUN: 之后的部分
            # 注意：有时是 // RUN:，有时是 //RUN:
            parts = stripped.split('RUN:', 1)
            if len(parts) > 1:
                cmd_part = parts[1].strip()
                
                # 如果上一行命令还没结束（有 \），则拼接
                if in_command:
                    current_command += " " + cmd_part
                else:
                    current_command = cmd_part
                
                in_command = True
        
        # 如果当前在处理命令，检查是否结束
        if in_command:
            # 如果当前行并不是以 // 开头的（可能是 RUN 命令换行到了下一行非注释区域？很少见）
            # 或者当前行就是刚刚提取的那行
            
            # 检查末尾是否有续行符 \
            if current_command.endswith('\\'):
                current_command = current_command[:-1].strip() # 去掉 \
                # 继续下一行寻找
                pass 
            else:
                # 命令结束
                solutions.append(current_command)
                current_command = ""
                in_command = False
                
    return solutions

def process_file(file_path, rel_path, meta_list):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"[Error] Reading {file_path}: {e}")
        return

    # 1. 拆分文件 (Split by // -----)
    # 官方 Split 标记通常是 // -----
    chunks = content.split('// -----')
    
    base_name = sanitize_filename(rel_path)
    base_name = base_name.replace('.mlir', '')

    for idx, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
            
        # 2. 过滤规则
        if SKIP_EXPECTED_ERROR and "expected-error" in chunk:
            continue

        # 3. 提取元数据
        dialects = extract_dialects(chunk)
        solutions = parse_run_commands(chunk)
        
        # 如果 Chunk 里没有 RUN 命令，尝试去 Header 找（也就是整个文件的第一部分）
        # 但如果是 Split 后的文件，通常每个部分是独立的。
        # 这里我们只保留该 Chunk 内部的 RUN，如果没有，可以留空，后续由你手动处理或用脚本统一跑一遍
        
        # 4. 生成新文件名
        case_id = f"{base_name}_case{idx}"
        new_filename = f"{case_id}.mlir"
        new_file_path = os.path.join(TARGET_DIR, new_filename)
        
        # 5. 写入新文件
        with open(new_file_path, 'w', encoding='utf-8') as out_f:
            out_f.write(chunk)
            
        # 6. 记录元数据
        meta_data = {
            "case_id": case_id,
            "case_path": new_file_path,
            "source_path": rel_path,
            "solution": solutions,
            "dialects": dialects,
            "raw_run_command_count": len(solutions)
        }
        meta_list.append(meta_data)

def main():
    print(f"Start processing from: {MLIR_TEST_DIR}")
    ensure_dir(TARGET_DIR)
    
    meta_list = []
    
    count = 0
    for root, dirs, files in os.walk(MLIR_TEST_DIR):
        for file in files:
            if file.endswith(".mlir"):
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, MLIR_TEST_DIR)
                
                process_file(abs_path, rel_path, meta_list)
                
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} source files...")

    # 保存 meta.json
    meta_path = os.path.join(TARGET_DIR, "meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_list, f, indent=2)
        
    print(f"\nDone! Processed {count} source files.")
    print(f"Total extracted cases: {len(meta_list)}")
    print(f"Metadata saved to: {meta_path}")

if __name__ == "__main__":
    main()
