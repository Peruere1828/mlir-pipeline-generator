"""
Comprehensive Knowledge Base builder for MLIR Pipeline Searcher.

Uses a curated pass list for the linalg -> scf -> cf -> arith -> llvm lowering path.
"""

from __future__ import annotations
import os
import re
import json
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

from definition import MLIRType, Operation, MLIRPass, RewritePattern, GlobalTransform
from solver_def import KnowledgeBase


# --- Curated Pass Specification ---
# (cli_name, src_dialects, tgt_dialects, type_conversions, phase)
# Phase 1: frontend lowering into linalg
# Phase 2: linalg optimizations
# Phase 3-4: bufferization
# Phase 5-6: linalg/affine -> scf/arith
# Phase 7-9: scf/cf/arith -> llvm
# Phase 10-11: memref/func -> llvm
# Phase 12: cleanup

PASS_SPEC = [
    # Phase 1 — frontend -> linalg
    ("tosa-to-linalg",               {"tosa"}, {"linalg", "arith"}, [], 1),
    ("convert-elementwise-to-linalg", {"arith"}, {"linalg"}, [], 1),
    ("convert-tensor-to-linalg",     {"tensor"}, {"linalg"}, [], 1),
    ("linalg-generalize-named-ops",  {"linalg"}, {"linalg"}, [], 1),

    # Phase 2 — linalg optimization
    ("linalg-fold-unit-extent-dims",  {"linalg"}, {"linalg"}, [], 2),
    ("linalg-fuse-elementwise-ops",   {"linalg"}, {"linalg"}, [], 2),

    # Phase 3 — bufferization
    ("one-shot-bufferize",  {"tensor"}, {"memref"}, [("tensor","memref")], 3),

    # Phase 4 — cleanup after bufferization
    ("canonicalize", set(), set(), [], 4),
    ("cse",          set(), set(), [], 4),

    # Phase 5 — buffer deallocation & bufferization lowering
    ("buffer-deallocation-pipeline",    {"memref"}, {"memref"}, [], 5),
    ("convert-bufferization-to-memref", {"bufferization"}, {"memref"}, [], 5),

    # Phase 6 — linalg/affine/vector -> scf/arith
    ("convert-linalg-to-loops",  {"linalg"}, {"scf", "arith", "math", "func"}, [], 6),
    ("convert-linalg-to-affine-loops", {"linalg"}, {"affine"}, [], 6),
    ("lower-affine",             {"affine"}, {"scf", "arith", "math", "func"}, [], 6),
    ("convert-vector-to-scf",    {"vector"}, {"scf", "arith", "memref", "func"}, [], 6),

    # Phase 7 — scf -> cf
    ("convert-scf-to-cf", {"scf"}, {"cf"}, [], 7),

    # Phase 8 — arith/math/index -> llvm
    ("convert-arith-to-llvm", {"arith"}, {"llvm"}, [], 8),
    ("convert-math-to-llvm",  {"math"}, {"llvm"}, [], 8),
    ("convert-math-to-libm",  {"math"}, {"llvm"}, [], 8),
    ("convert-index-to-llvm", {"index"}, {"llvm"}, [], 8),

    # Phase 9 — cf -> llvm
    ("convert-cf-to-llvm",  {"cf"}, {"llvm"}, [], 9),

    # Phase 10 — memref/vector finalization
    ("finalize-memref-to-llvm", {"memref"}, {"llvm"}, [("memref","llvm_ptr")], 10),
    ("expand-strided-metadata", {"memref"}, {"memref"}, [], 10),
    ("convert-vector-to-llvm",  {"vector"}, {"llvm"}, [], 10),

    # Phase 11 — func/ub -> llvm
    ("convert-func-to-llvm", {"func"}, {"llvm"}, [], 11),
    ("convert-ub-to-llvm",   {"ub"}, {"llvm"}, [], 11),

    # Phase 12 — cleanup casts
    ("reconcile-unrealized-casts", set(), set(), [], 12),
]


# --- Special Passes with Global Transforms ---

def _canonicalize_pass() -> MLIRPass:
    p = MLIRPass("canonicalize", cost=0.1)
    p.add_global_transform(GlobalTransform(
        name="drop-unrealized-conversion-cast",
        is_applicable=lambda ops, types: any(
            op.dialect == "builtin" and "cast" in op.name for op in ops),
        transform=lambda ops, types: (
            {op for op in ops if not (op.dialect == "builtin" and "cast" in op.name)},
            set(types)),
    ))
    p.add_global_transform(GlobalTransform(
        name="drop-builtin-ops",
        is_applicable=lambda ops, types: any(op.dialect == "builtin" for op in ops),
        transform=lambda ops, types: (
            {op for op in ops if op.dialect != "builtin"}, set(types)),
    ))
    p.add_global_transform(GlobalTransform(
        name="drop-transform-ops",
        is_applicable=lambda ops, types: any(op.dialect == "transform" for op in ops),
        transform=lambda ops, types: (
            {op for op in ops if op.dialect != "transform"}, set(types)),
    ))
    p.add_global_transform(GlobalTransform(
        name="drop-bufferization-ops",
        is_applicable=lambda ops, types: any(op.dialect == "bufferization" for op in ops),
        transform=lambda ops, types: (
            {op for op in ops if op.dialect != "bufferization"}, set(types)),
    ))
    p.add_global_transform(GlobalTransform(
        name="drop-memref-dealloc",
        is_applicable=lambda ops, types: any(
            op.dialect == "memref" and op.name == "dealloc" for op in ops),
        transform=lambda ops, types: (
            {op for op in ops if not (op.dialect == "memref" and op.name == "dealloc")},
            set(types)),
    ))
    return p


def _one_shot_bufferize_pass() -> MLIRPass:
    p = MLIRPass("one-shot-bufferize", cost=0.3)
    p.add_type_conversion("tensor", "memref")
    p.add_pattern(RewritePattern(
        src_dialect="tensor",
        generated_targets=[("memref", "generic")],
    ))
    return p


def _reconcile_pass() -> MLIRPass:
    p = MLIRPass("reconcile-unrealized-casts", cost=0.1)
    p.add_global_transform(GlobalTransform(
        name="remove-unrealized-casts",
        is_applicable=lambda ops, types: any(
            op.dialect == "builtin" and op.name == "unrealized_conversion_cast" for op in ops),
        transform=lambda ops, types: (
            {op for op in ops if not (op.dialect == "builtin"
                                       and op.name == "unrealized_conversion_cast")},
            set(types)),
    ))
    return p


def _buffer_deallocation_pass() -> MLIRPass:
    p = MLIRPass("buffer-deallocation-pipeline", cost=0.2)
    p.add_global_transform(GlobalTransform(
        name="drop-memref-dealloc",
        is_applicable=lambda ops, types: any(
            op.dialect == "memref" and op.name == "dealloc" for op in ops),
        transform=lambda ops, types: (
            {op for op in ops if not (op.dialect == "memref" and op.name == "dealloc")},
            set(types)),
    ))
    return p


SPECIAL_PASSES = {
    "canonicalize": _canonicalize_pass,
    "one-shot-bufferize": _one_shot_bufferize_pass,
    "reconcile-unrealized-casts": _reconcile_pass,
    "buffer-deallocation-pipeline": _buffer_deallocation_pass,
}


# --- Stage-gating conditions ---

def _has_tensor(ops, types):
    return MLIRType("tensor") in types

def _no_tensor(ops, types):
    return MLIRType("tensor") not in types

def _no_high_level_dialects(ops, types):
    """arith-to-llvm: only scalar arith remains — no tensor/linalg/scf/affine/tosa ops needed first"""
    pre_arith = {"linalg", "affine", "tensor", "tosa", "bufferization", "scf"}
    return not any(op.dialect in pre_arith for op in ops)

def _no_non_llvm_ops(ops, types):
    return all(op.dialect in {"llvm", "func", "builtin"} for op in ops)

def _has_memref(ops, types):
    return MLIRType("memref") in types

# Pass-specific condition overrides
PASS_CONDITIONS = {
    "convert-elementwise-to-linalg": _has_tensor,
    "convert-tensor-to-linalg": _has_tensor,
    "convert-linalg-to-loops": _no_tensor,
    "convert-linalg-to-affine-loops": _no_tensor,
    "convert-arith-to-llvm": _no_high_level_dialects,
    "convert-func-to-llvm": _no_non_llvm_ops,
    "finalize-memref-to-llvm": _no_tensor,
    "expand-strided-metadata": _no_tensor,
}


def build_comprehensive_kb(llvm_root: str = "") -> KnowledgeBase:
    kb = KnowledgeBase()
    curated_names: set[str] = set()

    for name, src_d, tgt_d, type_convs, phase in PASS_SPEC:
        curated_names.add(name)
        cost = phase * 0.08 + 0.05
        condition = PASS_CONDITIONS.get(name)

        if name in SPECIAL_PASSES:
            p = SPECIAL_PASSES[name]()
            p.phase = phase
            p.cost = cost
        else:
            p = MLIRPass(name=name, cost=cost, phase=phase, condition=condition)
            for src_t, tgt_t in type_convs:
                p.add_type_conversion(src_t, tgt_t)
            for sd in src_d:
                targets = [(t, "generic") for t in tgt_d] if tgt_d else []
                if targets:
                    p.add_pattern(RewritePattern(
                        src_dialect=sd,
                        generated_targets=targets,
                    ))

        kb.register_pass(p)

    # Augment with auto-discovered passes (curated ones take precedence)
    if llvm_root:
        discovered = discover_passes(llvm_root)
        for p in discovered:
            if p.name not in curated_names:
                kb.register_pass(p)

    print(f"[KB] Registered {len(kb.passes)} passes ({len(curated_names)} curated)")
    return kb


# --- Auto-discovery of passes from TableGen files ---

# Target dialects relevant to our lowering path
RELEVANT_DIALECTS = {
    "arith", "linalg", "scf", "affine", "cf", "func", "tosa",
    "tensor", "memref", "bufferization", "math", "index", "ub",
    "vector", "llvm", "builtin",
}

# Map of pass name patterns to phases (for auto-assignment)
PHASE_MAP: Dict[str, int] = {
    "tosa": 1,
    "elementwise-to-linalg": 1,
    "tensor-to-linalg": 1,
    "linalg-generalize": 1,
    "linalg-fold": 2,
    "linalg-fuse": 2,
    "linalg-morph": 2,
    "linalg-inline": 2,
    "linalg-block-pack": 2,
    "depthwise-conv": 2,
    "fold-tensor-subset": 2,
    "one-shot-bufferize": 3,
    "bufferization": 3,
    "empty-tensor-to-alloc": 3,
    "canonicalize": 4,
    "cse": 4,
    "drop-equivalent": 4,
    "resolve-shaped-type": 4,
    "resolve-ranked": 4,
    "reify-result": 4,
    "buffer-deallocation": 5,
    "bufferization-to-memref": 5,
    "normalize-memrefs": 5,
    "fold-memref-alias": 5,
    "linalg-to-loops": 6,
    "linalg-to-affine": 6,
    "linalg-to-parallel": 6,
    "lower-affine": 6,
    "vector-to-scf": 6,
    "affine-data-copy": 6,
    "affine-loop-fusion": 6,
    "affine-super-vectorize": 6,
    "affine-loop-coalescing": 6,
    "affine-raise": 6,
    "affine-simplify": 6,
    "affine-expand-index-ops": 6,
    "affine-fold-memref": 6,
    "scf-to-cf": 7,
    "lift-cf-to-scf": 7,
    "scf-for-loop-canonicalization": 7,
    "scf-for-loop-peeling": 7,
    "scf-parallel-loop-tiling": 7,
    "arith-to-llvm": 8,
    "math-to-llvm": 8,
    "math-to-libm": 8,
    "math-to-funcs": 8,
    "index-to-llvm": 8,
    "arith-expand": 8,
    "arith-emulate": 8,
    "math-expand": 8,
    "math-sincos": 8,
    "cf-to-llvm": 9,
    "finalize-memref": 10,
    "memref-to-llvm": 10,
    "vector-to-llvm": 10,
    "expand-strided-metadata": 10,
    "expand-realloc": 10,
    "flatten-memref": 10,
    "memref-emulate": 10,
    "lower-vector-to-from": 10,
    "func-to-llvm": 11,
    "ub-to-llvm": 11,
    "reconcile-unrealized-casts": 12,
}


def _assign_phase(pass_name: str) -> int:
    pass_lower = pass_name.lower()
    for pattern, phase in PHASE_MAP.items():
        if pattern in pass_lower:
            return phase
    return 50  # unknown — very high phase (only tried as last resort)


def _is_relevant_pass(p: ImportedPass) -> bool:
    """Filter to passes on the linalg -> llvm lowering path."""
    all_dialects = set(p.source_dialects + p.target_dialects)
    # Must involve at least one dialect we care about
    if not (all_dialects & RELEVANT_DIALECTS):
        return False
    # Exclude non-lowering passes
    exclude_patterns = [
        # GPU / non-CPU targets
        "emitc", "spirv", "gpu-to", "-to-gpu", "-to-nvvm", "-to-rocdl",
        "-to-amdgpu", "-to-amx", "-to-arm", "-to-xevm", "-to-xegpu",
        "-to-openmp", "-to-mpi", "-to-shard", "-to-pdl", "-to-mlprogram",
        "nvgpu-to", "nvvm-to", "arm-sme",
        # Non-lowering / misc
        "apfloat", "convert-shape", "convert-openacc",
        "set-llvm-module", "map-memref-spirv", "memref-to-spirv",
        "convert-shard", "convert-linalg-to-std",
        # Avoid generic "convert-to-llvm" (matches everything)
        "-to-llvm",
    ]
    name_lower = p.name.lower()
    for ex in exclude_patterns:
        if ex in name_lower:
            return False
    return True


def discover_passes(llvm_root: str, use_ai: bool = False) -> List[MLIRPass]:
    """Auto-discover passes from TableGen files in the LLVM project."""
    from pass_importer import AIPassImporter, ImportedPass

    td_search_paths = [
        "mlir/include/mlir/Conversion/Passes.td",
        "mlir/include/mlir/Dialect/Linalg/Passes.td",
        "mlir/include/mlir/Dialect/SCF/Transforms/Passes.td",
        "mlir/include/mlir/Dialect/Affine/Transforms/Passes.td",
        "mlir/include/mlir/Dialect/Arith/Transforms/Passes.td",
        "mlir/include/mlir/Dialect/Math/Transforms/Passes.td",
        "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.td",
        "mlir/include/mlir/Dialect/Func/Transforms/Passes.td",
        "mlir/include/mlir/Dialect/Vector/Transforms/Passes.td",
        "mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.td",
        "mlir/include/mlir/Dialect/Tensor/Transforms/Passes.td",
    ]

    importer = AIPassImporter(model="deepseek-v4-flash")
    discovered: List[MLIRPass] = []
    seen = set()

    for rel_path in td_search_paths:
        full_path = os.path.join(llvm_root, rel_path)
        if not os.path.exists(full_path):
            continue

        try:
            imported = importer.import_from_td(full_path, use_ai=use_ai)
        except Exception as e:
            print(f"[Discover] Error reading {rel_path}: {e}")
            continue

        for p in imported:
            if not p.name or p.name in seen:
                continue
            if not _is_relevant_pass(p):
                continue

            seen.add(p.name)
            phase = _assign_phase(p.name)
            cost = phase * 0.08 + 0.05

            mlir_pass = MLIRPass(name=p.name, cost=cost, phase=phase)
            for sd in p.source_dialects:
                tgt_dialects = [t for t in p.target_dialects if t.lower() != sd]
                targets = (
                    [(t, "generic") for t in tgt_dialects] if tgt_dialects else []
                )
                if targets:
                    mlir_pass.add_pattern(RewritePattern(
                        src_dialect=sd,
                        generated_targets=targets,
                    ))
            discovered.append(mlir_pass)

    print(f"[Discover] Auto-discovered {len(discovered)} relevant passes")
    return discovered


def dump_kb_json(kb: KnowledgeBase, output_path: str):
    data = []
    for p in kb.passes:
        data.append({
            "name": p.name,
            "cost": p.cost,
            "num_patterns": len(p.patterns),
            "num_type_conversions": len(p.type_conversions),
            "num_global_transforms": len(p.global_transforms),
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[KB] Exported to {output_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-json", action="store_true")
    ap.add_argument("--llvm-path", default="/home/ubuntuaaa/projects/mlir/llvm-project")
    args = ap.parse_args()

    kb = build_comprehensive_kb(args.llvm_path)

    if args.dump_json:
        dump_kb_json(kb, "kb_dump.json")

    from solver_def import PipelineSearcher, CompilationState, CompilationTarget

    target = CompilationTarget()
    for d in ["arith", "linalg", "scf", "cf", "func", "tensor", "affine",
              "memref", "tosa", "bufferization", "math", "index", "builtin"]:
        target.mark_dialect_illegal(d)
    target.mark_type_illegal("tensor")
    target.mark_type_illegal("memref")

    test_ops = {
        Operation("linalg", "generic", operand_types={MLIRType("tensor")}),
        Operation("arith", "addf", operand_types={MLIRType("tensor")}),
        Operation("func", "func"),
    }
    test_types = {MLIRType("tensor")}

    searcher = PipelineSearcher(kb)
    pipeline = searcher.search(test_ops, test_types, target)

    print(f"\n[KB Test] Input: {[str(o) for o in test_ops]}")
    print(f"[KB Test] Pipeline: {' -> '.join(pipeline) if pipeline else 'FAILED'}")