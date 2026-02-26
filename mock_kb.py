from definition import MLIRType, Operation, MLIRPass
from solver_def import KnowledgeBase 

def build_mock_kb() -> KnowledgeBase:
    kb = KnowledgeBase()
    
    t_tensor = MLIRType("tensor")
    t_memref = MLIRType("memref")

    # ==========================================
    # 1. 前端/高阶 Dialect (TOSA)
    # ==========================================
    p_tosa_to_arith = MLIRPass("tosa-to-arith")
    p_tosa_to_arith.src_dialects.add("tosa")
    p_tosa_to_arith.tgt_dialects.add("arith")
    kb.register_pass(p_tosa_to_arith)

    p_tosa_to_linalg = MLIRPass("tosa-to-linalg")
    p_tosa_to_linalg.src_dialects.add("tosa")
    p_tosa_to_linalg.tgt_dialects.add("linalg")
    kb.register_pass(p_tosa_to_linalg)

    p_tosa_to_scf = MLIRPass("tosa-to-scf")
    p_tosa_to_scf.src_dialects.add("tosa")
    p_tosa_to_scf.tgt_dialects.add("scf")
    kb.register_pass(p_tosa_to_scf)

    p_tosa_to_tensor = MLIRPass("tosa-to-tensor")
    p_tosa_to_tensor.src_dialects.add("tosa")
    p_tosa_to_tensor.tgt_dialects.add("tensor")
    kb.register_pass(p_tosa_to_tensor)

    # ==========================================
    # 2. Tensor 与 Linalg 转换 
    # ==========================================
    p_tensor_to_linalg = MLIRPass("convert-tensor-to-linalg")
    p_tensor_to_linalg.src_dialects.add("tensor")
    p_tensor_to_linalg.tgt_dialects.add("linalg")
    kb.register_pass(p_tensor_to_linalg)

    p_elem_to_linalg = MLIRPass("convert-elementwise-to-linalg")
    p_elem_to_linalg.src_traits.add("ElementwiseMappable")
    p_elem_to_linalg.op_type_requirements["trait_target"] = t_tensor 
    p_elem_to_linalg.tgt_ops.add(Operation("linalg", "generic", operand_types={t_tensor}))
    kb.register_pass(p_elem_to_linalg)

    p_linalg_to_std = MLIRPass("convert-linalg-to-std")
    p_linalg_to_std.src_dialects.add("linalg")
    p_linalg_to_std.tgt_dialects.add("std")
    kb.register_pass(p_linalg_to_std)

    # ==========================================
    # 3. 内存与 Bufferization 
    # ==========================================
    p_bufferize = MLIRPass("one-shot-bufferize")
    p_bufferize.src_types.add(t_tensor)
    p_bufferize.tgt_types.add(t_memref)
    kb.register_pass(p_bufferize)

    p_buf_to_memref = MLIRPass("convert-bufferization-to-memref")
    p_buf_to_memref.src_dialects.add("bufferization")
    p_buf_to_memref.tgt_dialects.add("memref")
    kb.register_pass(p_buf_to_memref)

    # ==========================================
    # 4. 结构化控制流 (SCF & Affine)
    # ==========================================
    p_linalg_to_scf = MLIRPass("convert-linalg-to-loops")
    p_linalg_to_scf.src_dialects.add("linalg")
    p_linalg_to_scf.op_type_requirements["linalg"] = t_memref 
    p_linalg_to_scf.tgt_dialects.add("scf")
    kb.register_pass(p_linalg_to_scf)

    p_linalg_to_affine = MLIRPass("convert-linalg-to-affine-loops")
    p_linalg_to_affine.src_dialects.add("linalg")
    p_linalg_to_affine.op_type_requirements["linalg"] = t_memref
    p_linalg_to_affine.tgt_dialects.add("affine")
    kb.register_pass(p_linalg_to_affine)

    p_lower_affine = MLIRPass("lower-affine")
    p_lower_affine.src_dialects.add("affine")
    p_lower_affine.tgt_dialects.add("scf")
    p_lower_affine.tgt_dialects.add("arith")
    kb.register_pass(p_lower_affine)

    p_scf_to_cf = MLIRPass("convert-scf-to-cf")
    p_scf_to_cf.src_dialects.add("scf")
    p_scf_to_cf.tgt_dialects.add("cf")
    kb.register_pass(p_scf_to_cf)

    p_scf_to_openmp = MLIRPass("convert-scf-to-openmp")
    p_scf_to_openmp.src_dialects.add("scf")
    p_scf_to_openmp.tgt_dialects.add("omp")
    kb.register_pass(p_scf_to_openmp)

    p_lift_cf_to_scf = MLIRPass("lift-cf-to-scf")
    p_lift_cf_to_scf.src_dialects.add("cf")
    p_lift_cf_to_scf.tgt_dialects.add("scf")
    kb.register_pass(p_lift_cf_to_scf)

    # ==========================================
    # 5. 基础计算 Dialects (Arith, Math, Complex)
    # ==========================================
    for src in ["arith", "math", "complex"]:
        p = MLIRPass(f"convert-{src}-to-llvm")
        p.src_dialects.add(src)
        p.tgt_dialects.add("llvm")
        kb.register_pass(p)

    for src in ["arith", "math", "complex"]:
        p = MLIRPass(f"convert-{src}-to-spirv")
        p.src_dialects.add(src)
        p.tgt_dialects.add("spirv")
        kb.register_pass(p)

    p_math_to_libm = MLIRPass("convert-math-to-libm")
    p_math_to_libm.src_dialects.add("math")
    p_math_to_libm.tgt_dialects.add("func") # call opaque
    kb.register_pass(p_math_to_libm)

    p_complex_to_libm = MLIRPass("convert-complex-to-libm")
    p_complex_to_libm.src_dialects.add("complex")
    p_complex_to_libm.tgt_dialects.add("func")
    kb.register_pass(p_complex_to_libm)

    p_arith_to_amdgpu = MLIRPass("convert-arith-to-amdgpu")
    p_arith_to_amdgpu.src_dialects.add("arith")
    p_arith_to_amdgpu.tgt_dialects.add("amdgpu")
    kb.register_pass(p_arith_to_amdgpu)

    # ==========================================
    # 6. Vector 及其相关 Lowering
    # ==========================================
    vector_targets = ["llvm", "scf", "gpu", "spirv", "arm_sme", "amx", "xegpu"]
    for tgt in vector_targets:
        p = MLIRPass(f"convert-vector-to-{tgt.replace('_', '-')}")
        p.src_dialects.add("vector")
        p.tgt_dialects.add(tgt)
        kb.register_pass(p)

    # ==========================================
    # 7. GPU, SPIR-V 及硬件相关
    # ==========================================
    gpu_targets = ["llvm", "nvvm", "rocdl", "spirv"]
    for tgt in gpu_targets:
        p = MLIRPass(f"convert-gpu-to-{tgt}")
        p.src_dialects.add("gpu")
        p.tgt_dialects.add(tgt)
        kb.register_pass(p)

    kb.register_pass(MLIRPass("convert-amdgpu-to-rocdl"))
    kb.register_pass(MLIRPass("convert-nvgpu-to-nvvm"))
    kb.register_pass(MLIRPass("convert-nvvm-to-llvm"))
    kb.register_pass(MLIRPass("convert-spirv-to-llvm"))
    kb.register_pass(MLIRPass("convert-xegpu-to-xevm"))
    kb.register_pass(MLIRPass("convert-xevm-to-llvm"))
    kb.register_pass(MLIRPass("convert-arm-sme-to-llvm"))

    p_amdgpu_to_rocdl = kb.passes[-7]; p_amdgpu_to_rocdl.src_dialects.add("amdgpu"); p_amdgpu_to_rocdl.tgt_dialects.add("rocdl")
    p_nvgpu_to_nvvm = kb.passes[-6]; p_nvgpu_to_nvvm.src_dialects.add("nvgpu"); p_nvgpu_to_nvvm.tgt_dialects.add("nvvm")
    p_nvvm_to_llvm = kb.passes[-5]; p_nvvm_to_llvm.src_dialects.add("nvvm"); p_nvvm_to_llvm.tgt_dialects.add("llvm")
    p_spirv_to_llvm = kb.passes[-4]; p_spirv_to_llvm.src_dialects.add("spirv"); p_spirv_to_llvm.tgt_dialects.add("llvm")
    p_xegpu_xevm = kb.passes[-3]; p_xegpu_xevm.src_dialects.add("xegpu"); p_xegpu_xevm.tgt_dialects.add("xevm")
    p_xevm_llvm = kb.passes[-2]; p_xevm_llvm.src_dialects.add("xevm"); p_xevm_llvm.tgt_dialects.add("llvm")
    p_sme_llvm = kb.passes[-1]; p_sme_llvm.src_dialects.add("arm_sme"); p_sme_llvm.tgt_dialects.add("llvm")

    p_scf_to_gpu = MLIRPass("convert-parallel-loops-to-gpu")
    p_scf_to_gpu.src_dialects.add("scf")
    p_scf_to_gpu.tgt_dialects.add("gpu")
    kb.register_pass(p_scf_to_gpu)

    # ==========================================
    # 8. 其他核心生态 (Func, EmitC, Index, Async)
    # ==========================================
    p_func_to_llvm = MLIRPass("convert-func-to-llvm")
    p_func_to_llvm.src_dialects.add("func")
    p_func_to_llvm.tgt_dialects.add("llvm")
    kb.register_pass(p_func_to_llvm)

    p_cf_to_llvm = MLIRPass("convert-cf-to-llvm")
    p_cf_to_llvm.src_dialects.add("cf")
    p_cf_to_llvm.tgt_dialects.add("llvm")
    kb.register_pass(p_cf_to_llvm)

    p_memref_to_llvm = MLIRPass("finalize-memref-to-llvm")
    p_memref_to_llvm.src_types.add(t_memref)
    p_memref_to_llvm.src_dialects.add("memref")
    p_memref_to_llvm.tgt_dialects.add("llvm")
    kb.register_pass(p_memref_to_llvm)
    
    p_index_to_llvm = MLIRPass("convert-index-to-llvm")
    p_index_to_llvm.src_dialects.add("index")
    p_index_to_llvm.tgt_dialects.add("llvm")
    kb.register_pass(p_index_to_llvm)

    p_ub_to_llvm = MLIRPass("convert-ub-to-llvm")
    p_ub_to_llvm.src_dialects.add("ub")
    p_ub_to_llvm.tgt_dialects.add("llvm")
    kb.register_pass(p_ub_to_llvm)

    p_async_to_llvm = MLIRPass("convert-async-to-llvm")
    p_async_to_llvm.src_dialects.add("async")
    p_async_to_llvm.tgt_dialects.add("llvm")
    kb.register_pass(p_async_to_llvm)

    # 统一收缩到 EmitC (C/C++ 代码生成)
    for src in ["arith", "math", "func", "scf", "memref"]:
        p = MLIRPass(f"convert-{src}-to-emitc")
        p.src_dialects.add(src)
        p.tgt_dialects.add("emitc")
        kb.register_pass(p)

    return kb