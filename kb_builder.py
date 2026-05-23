"""
Comprehensive Knowledge Base builder for MLIR Pipeline Searcher.

Uses a curated pass list for the linalg -> scf -> cf -> arith -> llvm lowering path.
"""

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

    # Phase 6 — linalg/affine -> scf/arith
    ("convert-linalg-to-loops",  {"linalg"}, {"scf", "arith", "math", "func"}, [], 6),
    ("convert-linalg-to-affine-loops", {"linalg"}, {"affine"}, [], 6),
    ("lower-affine",             {"affine"}, {"scf", "arith", "math", "func"}, [], 6),

    # Phase 7 — scf -> cf
    ("convert-scf-to-cf", {"scf"}, {"cf"}, [], 7),

    # Phase 8 — arith/math/index -> llvm
    ("convert-arith-to-llvm", {"arith"}, {"llvm"}, [], 8),
    ("convert-math-to-llvm",  {"math"}, {"llvm"}, [], 8),
    ("convert-index-to-llvm", {"index"}, {"llvm"}, [], 8),

    # Phase 9 — cf -> llvm
    ("convert-cf-to-llvm",  {"cf"}, {"llvm"}, [], 9),

    # Phase 10 — memref finalization
    ("finalize-memref-to-llvm", {"memref"}, {"llvm"}, [("memref","llvm_ptr")], 10),
    ("expand-strided-metadata", {"memref"}, {"memref"}, [], 10),

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


SPECIAL_PASSES = {
    "canonicalize": _canonicalize_pass,
    "one-shot-bufferize": _one_shot_bufferize_pass,
    "reconcile-unrealized-casts": _reconcile_pass,
}


def build_comprehensive_kb(llvm_root: str = "") -> KnowledgeBase:
    kb = KnowledgeBase()

    for name, src_d, tgt_d, type_convs, phase in PASS_SPEC:
        cost = phase * 0.08 + 0.05

        if name in SPECIAL_PASSES:
            p = SPECIAL_PASSES[name]()
            p.phase = phase
            p.cost = cost
        else:
            p = MLIRPass(name=name, cost=cost, phase=phase)
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

    print(f"[KB] Registered {len(kb.passes)} curated passes")
    return kb


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