import unittest

from pass_importer import PassTableGenImporter


class TestPassImporter(unittest.TestCase):
    def test_regex_import(self):
        content = '''
        def ConvertArithToLLVMPass : Pass<"convert-arith-to-llvm", "ModuleOp"> {
          let summary = "Lower arith ops to LLVM dialect";
        }
        '''
        importer = PassTableGenImporter(api_key=None)
        items = importer.import_from_content(content, use_ai=False)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].name, "convert-arith-to-llvm")
        self.assertEqual(items[0].source_dialects, ["arith"])
        self.assertEqual(items[0].target_dialects, ["llvm"])


if __name__ == "__main__":
    unittest.main()
