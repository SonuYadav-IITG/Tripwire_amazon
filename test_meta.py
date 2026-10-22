"""Meta-test: Python silently lets a later `def test_x` shadow an earlier one, so a weak test can hide a strong one."""

import ast
import collections
import pathlib


def test_no_test_function_is_defined_twice_in_a_module():
    dupes = {}
    for f in pathlib.Path(__file__).parent.glob("test_*.py"):
        names = [n.name for n in ast.parse(f.read_text()).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        d = [n for n, c in collections.Counter(names).items() if c > 1]
        if d:
            dupes[f.name] = d
    assert not dupes, f"shadowed tests: {dupes}"
