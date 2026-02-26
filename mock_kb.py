from definition import MLIRType, Operation, MLIRPass, RewritePattern
from solver_def import KnowledgeBase 

def build_mock_kb() -> KnowledgeBase:
    kb = KnowledgeBase()

    p_tosa_linalg = MLIRPass("tosa-to-linalg")
    # 规则：匹配 tosa 方言的任意 op，转为 linalg.generic
    p_tosa_linalg.add_pattern(RewritePattern(
        src_dialect="tosa", 
        tgt_dialect="linalg", 
        tgt_name="generic"
    ))
    kb.register_pass(p_tosa_linalg)

    p_linalg_scf = MLIRPass("convert-linalg-to-loops")
    # 条件：只有当 IR 中出现 memref (即已经 bufferize 过) 才能将 linalg 转为 scf
    def needs_memref(op, current_ops, types):
        return MLIRType("memref") in types
        
    p_linalg_scf.add_pattern(RewritePattern(
        src_dialect="linalg", 
        tgt_dialect="scf", 
        tgt_name="for",
        condition=needs_memref
    ))
    kb.register_pass(p_linalg_scf)

    p_scf_cf = MLIRPass("convert-scf-to-cf")
    p_scf_cf.add_pattern(RewritePattern(src_dialect="scf", tgt_dialect="cf", tgt_name="br"))
    kb.register_pass(p_scf_cf)

    p_cf_llvm = MLIRPass("convert-cf-to-llvm")
    p_cf_llvm.add_pattern(RewritePattern(src_dialect="cf", tgt_dialect="llvm", tgt_name="br"))
    kb.register_pass(p_cf_llvm)

    # 例子 1: Arith 转换到 LLVM (同名转换，arith.add 变 llvm.add)
    # 这个规则代表：opA=arith.*, opB=llvm.*，不需要指定 tgt_name，它会自动继承 add。
    p_arith = MLIRPass("convert-arith-to-llvm")
    p_arith.add_pattern(RewritePattern(src_dialect="arith", tgt_dialect="llvm"))
    kb.register_pass(p_arith)

    # 例子 2: Tosa 转换到 Linalg
    # 这个规则代表：opA=tosa.*, opB=linalg.generic (因为我们不想去细究 tosa 是怎么变成 linalg 的，统一视为 generic)
    p_tosa = MLIRPass("tosa-to-linalg")
    p_tosa.add_pattern(RewritePattern(src_dialect="tosa", tgt_dialect="linalg", tgt_name="generic"))
    kb.register_pass(p_tosa)

    # 例子 3: 复杂的带条件判定 (条件 C)
    # Convert elementwise to linalg: 只有在 op 包含 tensor 类型且具有特定 trait 时触发
    p_elem = MLIRPass("convert-elementwise-to-linalg")
    def elem_condition(op, state_types):
        return MLIRType("tensor") in op.operand_types and "ElementwiseMappable" in op.traits
        
    p_elem.add_pattern(RewritePattern(
        src_dialect="arith", # 假设我们匹配 arith 方言
        tgt_dialect="linalg",
        tgt_name="generic",
        condition=elem_condition  # 引入条件 C
    ))
    kb.register_pass(p_elem)

    # 例子 4: 经典的类型转换 Pass (Bufferization)
    p_bufferize = MLIRPass("one-shot-bufferize")
    # Bufferize 不需要转换特定的 Op (粗略模型下)，而是改变全局类型 tensor -> memref
    p_bufferize.add_type_conversion("tensor", "memref")
    kb.register_pass(p_bufferize)

    def is_final_stage(op, current_ops, types):
        # 遍历当前环境中的所有 op
        for other_op in current_ops:
            # 如果还有除了 func、llvm（以及最外层可能存在的 builtin）之外的方言，就不允许收尾
            if other_op.dialect not in ["func", "llvm", "builtin"]:
                return False
        return True

    p_func_to_llvm = MLIRPass("convert-func-to-llvm")
    p_func_to_llvm.add_pattern(RewritePattern(
        src_dialect="func", 
        tgt_dialect="llvm", 
        tgt_name="func",
        condition=is_final_stage   # 【新增这里】：挂载收尾条件
    ))
    kb.register_pass(p_func_to_llvm)

    p_memref_llvm = MLIRPass("finalize-memref-to-llvm")
    # 消除 memref 类型，假装转为 llvm.ptr (这里用一个字符串表示底层的裸指针)
    p_memref_llvm.add_type_conversion("memref", "llvm_ptr")
    kb.register_pass(p_memref_llvm)

    return kb

# def build_mock_kb() -> KnowledgeBase:
#     kb = KnowledgeBase()
    
#     t_tensor = MLIRType("tensor")
#     t_memref = MLIRType("memref")

#     # ==========================================
#     # 1. 前端/高阶 Dialect (TOSA)
#     # ==========================================
#     p_tosa_to_arith = MLIRPass("tosa-to-arith")
#     p_tosa_to_arith.src_dialects.add("tosa")
#     p_tosa_to_arith.tgt_dialects.add("arith")
#     kb.register_pass(p_tosa_to_arith)

#     p_tosa_to_linalg = MLIRPass("tosa-to-linalg")
#     p_tosa_to_linalg.src_dialects.add("tosa")
#     p_tosa_to_linalg.tgt_dialects.add("linalg")
#     kb.register_pass(p_tosa_to_linalg)

#     p_tosa_to_scf = MLIRPass("tosa-to-scf")
#     p_tosa_to_scf.src_dialects.add("tosa")
#     p_tosa_to_scf.tgt_dialects.add("scf")
#     kb.register_pass(p_tosa_to_scf)

#     p_tosa_to_tensor = MLIRPass("tosa-to-tensor")
#     p_tosa_to_tensor.src_dialects.add("tosa")
#     p_tosa_to_tensor.tgt_dialects.add("tensor")
#     kb.register_pass(p_tosa_to_tensor)

#     # ==========================================
#     # 2. Tensor 与 Linalg 转换 
#     # ==========================================
#     p_tensor_to_linalg = MLIRPass("convert-tensor-to-linalg")
#     p_tensor_to_linalg.src_dialects.add("tensor")
#     p_tensor_to_linalg.tgt_dialects.add("linalg")
#     kb.register_pass(p_tensor_to_linalg)

#     p_elem_to_linalg = MLIRPass("convert-elementwise-to-linalg")
#     p_elem_to_linalg.src_traits.add("ElementwiseMappable")
#     p_elem_to_linalg.op_type_requirements["trait_target"] = t_tensor 
#     p_elem_to_linalg.tgt_ops.add(Operation("linalg", "generic", operand_types={t_tensor}))
#     kb.register_pass(p_elem_to_linalg)

#     p_linalg_to_std = MLIRPass("convert-linalg-to-std")
#     p_linalg_to_std.src_dialects.add("linalg")
#     p_linalg_to_std.tgt_dialects.add("std")
#     kb.register_pass(p_linalg_to_std)

#     # ==========================================
#     # 3. 内存与 Bufferization 
#     # ==========================================
#     p_bufferize = MLIRPass("one-shot-bufferize")
#     p_bufferize.src_types.add(t_tensor)
#     p_bufferize.tgt_types.add(t_memref)
#     kb.register_pass(p_bufferize)

#     p_buf_to_memref = MLIRPass("convert-bufferization-to-memref")
#     p_buf_to_memref.src_dialects.add("bufferization")
#     p_buf_to_memref.tgt_dialects.add("memref")
#     kb.register_pass(p_buf_to_memref)

#     # ==========================================
#     # 4. 结构化控制流 (SCF & Affine)
#     # ==========================================
#     p_linalg_to_scf = MLIRPass("convert-linalg-to-loops")
#     p_linalg_to_scf.src_dialects.add("linalg")
#     p_linalg_to_scf.op_type_requirements["linalg"] = t_memref 
#     p_linalg_to_scf.tgt_dialects.add("scf")
#     kb.register_pass(p_linalg_to_scf)

#     p_linalg_to_affine = MLIRPass("convert-linalg-to-affine-loops")
#     p_linalg_to_affine.src_dialects.add("linalg")
#     p_linalg_to_affine.op_type_requirements["linalg"] = t_memref
#     p_linalg_to_affine.tgt_dialects.add("affine")
#     kb.register_pass(p_linalg_to_affine)

#     p_lower_affine = MLIRPass("lower-affine")
#     p_lower_affine.src_dialects.add("affine")
#     p_lower_affine.tgt_dialects.add("scf")
#     p_lower_affine.tgt_dialects.add("arith")
#     kb.register_pass(p_lower_affine)

#     p_scf_to_cf = MLIRPass("convert-scf-to-cf")
#     p_scf_to_cf.src_dialects.add("scf")
#     p_scf_to_cf.tgt_dialects.add("cf")
#     kb.register_pass(p_scf_to_cf)

#     p_scf_to_openmp = MLIRPass("convert-scf-to-openmp")
#     p_scf_to_openmp.src_dialects.add("scf")
#     p_scf_to_openmp.tgt_dialects.add("omp")
#     kb.register_pass(p_scf_to_openmp)

#     p_lift_cf_to_scf = MLIRPass("lift-cf-to-scf")
#     p_lift_cf_to_scf.src_dialects.add("cf")
#     p_lift_cf_to_scf.tgt_dialects.add("scf")
#     kb.register_pass(p_lift_cf_to_scf)

#     # ==========================================
#     # 5. 基础计算 Dialects (Arith, Math, Complex)
#     # ==========================================
#     for src in ["arith", "math", "complex"]:
#         p = MLIRPass(f"convert-{src}-to-llvm")
#         p.src_dialects.add(src)
#         p.tgt_dialects.add("llvm")
#         kb.register_pass(p)

#     for src in ["arith", "math", "complex"]:
#         p = MLIRPass(f"convert-{src}-to-spirv")
#         p.src_dialects.add(src)
#         p.tgt_dialects.add("spirv")
#         kb.register_pass(p)

#     p_math_to_libm = MLIRPass("convert-math-to-libm")
#     p_math_to_libm.src_dialects.add("math")
#     p_math_to_libm.tgt_dialects.add("func") # call opaque
#     kb.register_pass(p_math_to_libm)

#     p_complex_to_libm = MLIRPass("convert-complex-to-libm")
#     p_complex_to_libm.src_dialects.add("complex")
#     p_complex_to_libm.tgt_dialects.add("func")
#     kb.register_pass(p_complex_to_libm)

#     p_arith_to_amdgpu = MLIRPass("convert-arith-to-amdgpu")
#     p_arith_to_amdgpu.src_dialects.add("arith")
#     p_arith_to_amdgpu.tgt_dialects.add("amdgpu")
#     kb.register_pass(p_arith_to_amdgpu)

#     # ==========================================
#     # 6. Vector 及其相关 Lowering
#     # ==========================================
#     vector_targets = ["llvm", "scf", "gpu", "spirv", "arm_sme", "amx", "xegpu"]
#     for tgt in vector_targets:
#         p = MLIRPass(f"convert-vector-to-{tgt.replace('_', '-')}")
#         p.src_dialects.add("vector")
#         p.tgt_dialects.add(tgt)
#         kb.register_pass(p)

#     # ==========================================
#     # 7. GPU, SPIR-V 及硬件相关
#     # ==========================================
#     gpu_targets = ["llvm", "nvvm", "rocdl", "spirv"]
#     for tgt in gpu_targets:
#         p = MLIRPass(f"convert-gpu-to-{tgt}")
#         p.src_dialects.add("gpu")
#         p.tgt_dialects.add(tgt)
#         kb.register_pass(p)

#     kb.register_pass(MLIRPass("convert-amdgpu-to-rocdl"))
#     kb.register_pass(MLIRPass("convert-nvgpu-to-nvvm"))
#     kb.register_pass(MLIRPass("convert-nvvm-to-llvm"))
#     kb.register_pass(MLIRPass("convert-spirv-to-llvm"))
#     kb.register_pass(MLIRPass("convert-xegpu-to-xevm"))
#     kb.register_pass(MLIRPass("convert-xevm-to-llvm"))
#     kb.register_pass(MLIRPass("convert-arm-sme-to-llvm"))

#     p_amdgpu_to_rocdl = kb.passes[-7]; p_amdgpu_to_rocdl.src_dialects.add("amdgpu"); p_amdgpu_to_rocdl.tgt_dialects.add("rocdl")
#     p_nvgpu_to_nvvm = kb.passes[-6]; p_nvgpu_to_nvvm.src_dialects.add("nvgpu"); p_nvgpu_to_nvvm.tgt_dialects.add("nvvm")
#     p_nvvm_to_llvm = kb.passes[-5]; p_nvvm_to_llvm.src_dialects.add("nvvm"); p_nvvm_to_llvm.tgt_dialects.add("llvm")
#     p_spirv_to_llvm = kb.passes[-4]; p_spirv_to_llvm.src_dialects.add("spirv"); p_spirv_to_llvm.tgt_dialects.add("llvm")
#     p_xegpu_xevm = kb.passes[-3]; p_xegpu_xevm.src_dialects.add("xegpu"); p_xegpu_xevm.tgt_dialects.add("xevm")
#     p_xevm_llvm = kb.passes[-2]; p_xevm_llvm.src_dialects.add("xevm"); p_xevm_llvm.tgt_dialects.add("llvm")
#     p_sme_llvm = kb.passes[-1]; p_sme_llvm.src_dialects.add("arm_sme"); p_sme_llvm.tgt_dialects.add("llvm")

#     p_scf_to_gpu = MLIRPass("convert-parallel-loops-to-gpu")
#     p_scf_to_gpu.src_dialects.add("scf")
#     p_scf_to_gpu.tgt_dialects.add("gpu")
#     kb.register_pass(p_scf_to_gpu)

#     # ==========================================
#     # 8. 其他核心生态 (Func, EmitC, Index, Async)
#     # ==========================================
#     p_func_to_llvm = MLIRPass("convert-func-to-llvm")
#     p_func_to_llvm.src_dialects.add("func")
#     p_func_to_llvm.tgt_dialects.add("llvm")
#     kb.register_pass(p_func_to_llvm)

#     p_cf_to_llvm = MLIRPass("convert-cf-to-llvm")
#     p_cf_to_llvm.src_dialects.add("cf")
#     p_cf_to_llvm.tgt_dialects.add("llvm")
#     kb.register_pass(p_cf_to_llvm)

#     p_memref_to_llvm = MLIRPass("finalize-memref-to-llvm")
#     p_memref_to_llvm.src_types.add(t_memref)
#     p_memref_to_llvm.src_dialects.add("memref")
#     p_memref_to_llvm.tgt_dialects.add("llvm")
#     kb.register_pass(p_memref_to_llvm)
    
#     p_index_to_llvm = MLIRPass("convert-index-to-llvm")
#     p_index_to_llvm.src_dialects.add("index")
#     p_index_to_llvm.tgt_dialects.add("llvm")
#     kb.register_pass(p_index_to_llvm)

#     p_ub_to_llvm = MLIRPass("convert-ub-to-llvm")
#     p_ub_to_llvm.src_dialects.add("ub")
#     p_ub_to_llvm.tgt_dialects.add("llvm")
#     kb.register_pass(p_ub_to_llvm)

#     p_async_to_llvm = MLIRPass("convert-async-to-llvm")
#     p_async_to_llvm.src_dialects.add("async")
#     p_async_to_llvm.tgt_dialects.add("llvm")
#     kb.register_pass(p_async_to_llvm)

#     # 统一收缩到 EmitC (C/C++ 代码生成)
#     for src in ["arith", "math", "func", "scf", "memref"]:
#         p = MLIRPass(f"convert-{src}-to-emitc")
#         p.src_dialects.add(src)
#         p.tgt_dialects.add("emitc")
#         kb.register_pass(p)

#     return kb