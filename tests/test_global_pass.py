import unittest

from definition import MLIRType, Operation
from mock_kb import build_mock_kb
from solver_def import CompilationState


class TestGlobalPass(unittest.TestCase):
    def test_canonicalize_removes_unrealized_cast(self):
        kb = build_mock_kb()
        canonicalize = next(p for p in kb.passes if p.name == "canonicalize")

        state = CompilationState(
            ops={Operation("builtin", "unrealized_conversion_cast"), Operation("arith", "addf")},
            types={MLIRType("tensor")},
        )
        self.assertTrue(canonicalize.is_applicable(set(state.ops), set(state.types)))

        next_state = kb.apply_pass(state, canonicalize)
        self.assertNotIn(Operation("builtin", "unrealized_conversion_cast"), set(next_state.ops))

    def test_wildcard_pattern_matches_any_dialect(self):
        from definition import MLIRPass, RewritePattern

        p = MLIRPass("global-rewrite")
        p.add_pattern(RewritePattern(src_dialect="*", src_name="foo", generated_targets=[("llvm", "foo")]))
        self.assertTrue(p.is_applicable({Operation("toy", "foo")}, set()))


if __name__ == "__main__":
    unittest.main()
