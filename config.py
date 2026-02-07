# ===========================================
# 官方 Dialect 白名单 (根据官方文档整理对应的 IR Namespace)
# ===========================================
OFFICIAL_DIALECT_NAMESPACES = {
    'acc',          # OpenACC
    'affine',
    'amdgpu',
    'amx',
    'arith',
    'arm_neon',
    'arm_sme',      # Doc: ArmSME
    'arm_sve',
    'async',
    'bufferization',
    'builtin',      # Builtin
    'cf',           # ControlFlow
    'complex',
    'dlti',
    'emitc',
    'func',
    'gpu',
    'index',
    'irdl',
    'linalg',
    'llvm',
    'math',
    'memref',
    'ml_program',
    'mpi',
    'nvgpu',
    'nvvm',
    'omp',          # OpenMP
    'pdl',
    'pdl_interp',
    'ptr',
    'quant',
    'rocdl',
    'scf',
    'shape',
    'shard',
    'smt',
    'sparse_tensor',
    'spirv',        # Doc: SPIR-V
    'tensor',
    'tosa',         # Doc: TOSA
    'transform',
    'ub',
    'vcix',
    'vector',
    'wasmssa',
    'x86vector',
    'xegpu',
    'xevm'
}