"""
Test harness — runs the pipeline searcher against MLIR integration test files
and validates output with mlir-opt.
"""
import subprocess
import sys
import os
from pathlib import Path

from mlir_parser import MLIRParser
from kb_builder import build_comprehensive_kb
from solver_def import PipelineSearcher, CompilationTarget

MLIR_OPT = "/home/ubuntuaaa/projects/mlir/llvm-project/build/bin/mlir-opt"
LLVM_ROOT = "/home/ubuntuaaa/projects/mlir/llvm-project"

# Dialects/types we want to lower away
ILLEGAL_DIALECTS = [
    "arith", "linalg", "scf", "affine", "cf", "func", "tosa",
    "tensor", "memref", "bufferization", "math", "index", "ub",
    "vector", "builtin",
]
ILLEGAL_TYPES = ["tensor", "memref"]


def build_target():
    target = CompilationTarget()
    for d in ILLEGAL_DIALECTS:
        target.mark_dialect_illegal(d)
    for t in ILLEGAL_TYPES:
        target.mark_type_illegal(t)
    return target


def find_pipeline(mlir_path, kb, target, parser):
    parsed = parser.parse_file(mlir_path)
    start_ops = parsed.get("ops", set())
    start_types = parsed.get("types", set())

    if not start_ops:
        return None, "no ops parsed"

    searcher = PipelineSearcher(kb)
    result = searcher.search(start_ops, start_types, target)
    if not result:
        return None, "no pipeline found"

    return list(result), None


def run_mlir_opt(mlir_path, pipeline):
    """Run mlir-opt with the generated pipeline. Returns (success, output)."""
    if not pipeline:
        return False, "empty pipeline"

    pass_pipeline = ",".join(pipeline)
    cmd = [MLIR_OPT, mlir_path, f"--pass-pipeline=builtin.module({pass_pipeline})"]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return True, proc.stdout[:500]
        return False, proc.stderr[:300] or "(no output)"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, f"mlir-opt not found at {MLIR_OPT}"


def run_tests(test_dir, exclude_patterns=None):
    """Run solver against all .mlir files in test_dir and validate with mlir-opt."""
    if exclude_patterns is None:
        exclude_patterns = []

    kb = build_comprehensive_kb(LLVM_ROOT)
    target = build_target()
    parser = MLIRParser()

    test_files = sorted(Path(test_dir).rglob("*.mlir"))

    results = {"pass": [], "fail": [], "skip": []}

    for tf in test_files:
        rel = tf.relative_to(test_dir)
        if any(pat in str(rel) for pat in exclude_patterns):
            results["skip"].append((str(rel), "excluded"))
            continue

        pipeline, err = find_pipeline(str(tf), kb, target, parser)
        if err:
            results["fail"].append((str(rel), err))
            continue

        ok, detail = run_mlir_opt(str(tf), pipeline)
        if ok:
            results["pass"].append((str(rel), " -> ".join(pipeline)))
        else:
            results["fail"].append((str(rel), f"mlir-opt rejected: {detail[:120]}"))

    return results


def print_results(results, label):
    total = len(results["pass"]) + len(results["fail"])
    if total == 0:
        print(f"\n[{label}] No tests")
        return

    print(f"\n=== {label} ===")
    print(f"PASS: {len(results['pass'])}/{total} ({100*len(results['pass'])/total:.0f}%)")
    for path, pipeline in results["pass"]:
        print(f"  [PASS] {path}")
        print(f"         {pipeline}")
    for path, reason in results["fail"]:
        print(f"  [FAIL] {path}: {reason}")
    for path, reason in results["skip"]:
        print(f"  [SKIP] {path}: {reason}")


if __name__ == "__main__":
    llvm_test = os.path.join(LLVM_ROOT, "mlir/test/Integration/Dialect")

    # Linalg CPU integration tests
    linalg_dir = os.path.join(llvm_test, "Linalg/CPU")
    linalg_results = run_tests(
        linalg_dir,
        exclude_patterns=["ArmSME", "ArmSVE", "transform", "x86vector"],
    )
    print_results(linalg_results, "Linalg CPU Integration")

    # Vector CPU integration tests
    vector_dir = os.path.join(llvm_test, "Vector/CPU")
    vector_results = run_tests(
        vector_dir,
        exclude_patterns=["ArmSME", "ArmSVE", "AMX", "x86vector", "transform", "GPU"],
    )
    print_results(vector_results, "Vector CPU Integration")

    # Math CPU integration tests
    math_dir = os.path.join(llvm_test, "Math/CPU")
    math_results = run_tests(math_dir)
    print_results(math_results, "Math CPU Integration")

    # Arith CPU integration tests
    arith_dir = os.path.join(llvm_test, "Arith/CPU")
    arith_results = run_tests(arith_dir)
    print_results(arith_results, "Arith CPU Integration")

    # Tensor integration tests
    tensor_dir = os.path.join(llvm_test, "Tensor")
    tensor_results = run_tests(tensor_dir)
    print_results(tensor_results, "Tensor Integration")

    # MemRef integration tests
    memref_dir = os.path.join(llvm_test, "MemRef")
    memref_results = run_tests(memref_dir)
    print_results(memref_results, "MemRef Integration")

    # ControlFlow integration tests
    cf_dir = os.path.join(llvm_test, "ControlFlow")
    cf_results = run_tests(cf_dir)
    print_results(cf_results, "ControlFlow Integration")

    # Standard CPU integration tests
    std_dir = os.path.join(llvm_test, "Standard/CPU")
    std_results = run_tests(std_dir)
    print_results(std_results, "Standard CPU Integration")