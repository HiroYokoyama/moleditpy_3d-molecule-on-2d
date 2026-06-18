import os
import sys
import unittest
import importlib.util
from pathlib import Path

# Paths
_TESTS_DIR = Path(__file__).resolve().parent
_PLUGIN_ROOT = _TESTS_DIR.parent
_PLUGIN_FILE = _PLUGIN_ROOT / "3d_molecule_on_2d.py"
_WORKSPACE_ROOT = _PLUGIN_ROOT.parent

# Find the main app
_DEFAULT_APP_CANDIDATES = [
    _WORKSPACE_ROOT / "python_molecular_editor",
    _PLUGIN_ROOT / "python_molecular_editor",  # CI path
]
_APP_PATH = next((p for p in _DEFAULT_APP_CANDIDATES if p and (p / "moleditpy").exists()), None)

# Find the checker script
_CHECKER_CANDIDATES = [
    _WORKSPACE_ROOT / "moleditpy-plugins" / "api-checker" / "plugin_api_checker.py",
    _WORKSPACE_ROOT / "other" / "plugin_api_checker.py",
]
if _APP_PATH:
    _CHECKER_CANDIDATES.extend([
        _APP_PATH / "api-checker" / "plugin_api_checker.py",
        _APP_PATH / "tests" / "plugin_api_checker.py",
        _APP_PATH / "plugin_api_checker.py",
    ])
_CHECKER_PATH = next((p for p in _CHECKER_CANDIDATES if p and p.exists()), None)

# Skip conditions
HAS_APP = _APP_PATH is not None
HAS_CHECKER = _CHECKER_PATH is not None

def _load_checker():
    spec = importlib.util.spec_from_file_location("plugin_api_checker", _CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(_CHECKER_PATH)
    spec.loader.exec_module(mod)
    return mod

class TestAPIChecker(unittest.TestCase):
    @unittest.skipUnless(HAS_APP, "Main application repository (python_molecular_editor) not found")
    @unittest.skipUnless(HAS_CHECKER, "API checker script (plugin_api_checker.py) not found")
    def test_no_unknown_api_accesses(self):
        checker_mod = _load_checker()
        
        # Build the APIInfo from the main app
        extractor = checker_mod.AppAPIExtractor(_APP_PATH, verbose=False)
        api = extractor.extract()
        
        # Merge allowlists (same as running with --default-allowlist --mw-allowlist + site allowlist)
        site_allowlist = checker_mod._load_site_allowlist(_PLUGIN_FILE)
        allowlist = checker_mod._merge_allowlists(
            checker_mod._MANAGER_ALLOWLIST,
            checker_mod._MW_ALLOWLIST,
            site_allowlist
        )
        
        # Check this plugin
        checker = checker_mod.PluginFileChecker(
            _PLUGIN_FILE,
            api,
            check_context=True,  # Check context.X methods too
            allowlist=allowlist
        )
        
        issues = checker.check()
        
        # Format and fail if any issues found
        if issues:
            lines = [
                f"  [{i.code}] {Path(i.file).name} line {i.line}: {i.message}"
                for i in issues
            ]
            self.fail(
                f"{len(issues)} unknown API access(es) found in {Path(_PLUGIN_FILE).name}:\n"
                + "\n".join(lines)
            )

if __name__ == "__main__":
    unittest.main()
