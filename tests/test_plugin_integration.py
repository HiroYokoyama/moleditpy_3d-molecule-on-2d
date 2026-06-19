"""
Integration tests for 3d_molecule_on_2d.py
Verifies the plugin contract (menu action, save/load handlers) without Qt/RDKit.

Two execution modes
-------------------
1. Stub mode    Ealways runs (CI + local).
2. Real-context mode  Eruns when python_molecular_editor is present.

CI setup
--------
The ci.yml already clones the main app:

    - name: Clone main MoleditPy application
      run: git clone --depth 1 https://github.com/HiroYokoyama/python_molecular_editor.git
             python_molecular_editor
"""

import sys
import os
import types
import unittest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub Qt and heavy deps before importing the plugin
# ---------------------------------------------------------------------------


def _install_stubs():
    if "PyQt6" not in sys.modules or not hasattr(sys.modules.get("PyQt6"), "__file__"):
        pyqt6 = types.ModuleType("PyQt6")

        # sip - imported at top level by 3d_molecule_on_2d
        sip_stub = types.ModuleType("sip")
        sip_stub.isdeleted = lambda obj: False
        sys.modules["sip"] = sip_stub
        # Also handle "from PyQt6 import sip" branch
        pyqt6.sip = sip_stub
        sys.modules["PyQt6.sip"] = sip_stub

        qt_core = types.ModuleType("PyQt6.QtCore")
        qt_core.Qt = MagicMock()
        qt_core.QPointF = MagicMock()
        qt_core.QEvent = MagicMock()
        qt_core.QObject = MagicMock()

        class _QTimer:
            @staticmethod
            def singleShot(ms, fn):
                pass  # do not call fn - avoids cascading Qt dependency

        qt_core.QTimer = _QTimer
        qt_core.pyqtSignal = MagicMock()
        qt_core.QThread = MagicMock()

        qt_widgets = types.ModuleType("PyQt6.QtWidgets")
        for cls_name in [
            "QDialog",
            "QVBoxLayout",
            "QHBoxLayout",
            "QLabel",
            "QSlider",
            "QGraphicsItem",
            "QCheckBox",
            "QFrame",
            "QSpacerItem",
            "QSizePolicy",
        ]:
            setattr(qt_widgets, cls_name, MagicMock())

        qt_gui = types.ModuleType("PyQt6.QtGui")
        qt_gui.QColor = MagicMock()

        pyqt6.QtCore = qt_core
        pyqt6.QtWidgets = qt_widgets
        pyqt6.QtGui = qt_gui

        sys.modules["PyQt6"] = pyqt6
        sys.modules["PyQt6.QtCore"] = qt_core
        sys.modules["PyQt6.QtWidgets"] = qt_widgets
        sys.modules["PyQt6.QtGui"] = qt_gui

    # RDKit
    rdkit_stub = types.ModuleType("rdkit")
    rdkit_chem = types.ModuleType("rdkit.Chem")
    rdkit_chem.AllChem = MagicMock()
    rdkit_geometry = types.ModuleType("rdkit.Geometry")
    rdkit_geometry.Point3D = MagicMock()
    sys.modules.setdefault("rdkit", rdkit_stub)
    sys.modules.setdefault("rdkit.Chem", rdkit_chem)
    sys.modules.setdefault("rdkit.Chem.AllChem", MagicMock())
    sys.modules.setdefault("rdkit.Geometry", rdkit_geometry)

    # numpy is real  Eleave it
    sys.modules.setdefault("pyvista", types.ModuleType("pyvista"))


_install_stubs()

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

import importlib.util

_PLUGIN_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "3d_molecule_on_2d.py")
)
_spec = importlib.util.spec_from_file_location("_3d_molecule_on_2d", _PLUGIN_PATH)
_pkg = importlib.util.module_from_spec(_spec)
sys.modules["_3d_molecule_on_2d"] = _pkg
_spec.loader.exec_module(_pkg)

initialize = _pkg.initialize
PLUGIN_NAME = _pkg.PLUGIN_NAME
PLUGIN_VERSION = _pkg.PLUGIN_VERSION


# ---------------------------------------------------------------------------
# Stub PluginContext
# ---------------------------------------------------------------------------


class _StubContext:
    def __init__(self):
        self._menu_actions = []
        self._save_handler = None
        self._load_handler = None
        self._toolbar_actions = []

    def add_menu_action(self, path, callback, **kwargs):
        self._menu_actions.append((path, callback))

    def register_save_handler(self, fn):
        self._save_handler = fn

    def register_load_handler(self, fn):
        self._load_handler = fn

    def add_toolbar_action(self, callback, text, icon=None, tooltip=None):
        self._toolbar_actions.append((text, callback))

    # Full standard API stubs
    def get_main_window(self):
        mw = MagicMock()
        mw.menuBar.return_value.actions.return_value = []
        mw.scene = None
        return mw

    def show_status_message(self, msg, duration=0):
        pass

    def register_document_reset_handler(self, fn):
        pass

    def register_file_opener(self, ext, fn, priority=0):
        pass

    def register_drop_handler(self, fn, priority=0):
        pass

    def add_export_action(self, label, fn):
        pass

    def add_analysis_tool(self, label, fn):
        pass

    def register_window(self, key, win):
        pass

    def get_window(self, key):
        return None


# ---------------------------------------------------------------------------
# Tests: metadata
# ---------------------------------------------------------------------------


class TestMetadata(unittest.TestCase):
    def test_plugin_name(self):
        self.assertEqual(PLUGIN_NAME, "3D Molecule on 2D")

    def test_plugin_version_is_semver(self):
        parts = PLUGIN_VERSION.split(".")
        self.assertEqual(len(parts), 3)
        for p in parts:
            self.assertTrue(p.isdigit(), f"Non-numeric version part: {p!r}")


# ---------------------------------------------------------------------------
# Tests: initialize contract
# ---------------------------------------------------------------------------


class TestInitialize(unittest.TestCase):
    def setUp(self):
        # Reset module-level globals before each test
        _pkg._mw = None
        _pkg._context = None
        self.ctx = _StubContext()
        initialize(self.ctx)

    def test_registers_settings_menu_action(self):
        paths = [p for p, _ in self.ctx._menu_actions]
        self.assertTrue(
            any("3D" in p for p in paths),
            f"Expected a '3D' settings menu entry, got: {paths}",
        )

    def test_menu_action_is_callable(self):
        for _, callback in self.ctx._menu_actions:
            self.assertTrue(callable(callback))

    def test_menu_path_is_namespaced(self):
        for path, _ in self.ctx._menu_actions:
            self.assertIn("/", path)

    def test_registers_save_handler(self):
        self.assertIsNotNone(self.ctx._save_handler)

    def test_registers_load_handler(self):
        self.assertIsNotNone(self.ctx._load_handler)

    def test_save_handler_is_callable(self):
        self.assertTrue(callable(self.ctx._save_handler))

    def test_load_handler_is_callable(self):
        self.assertTrue(callable(self.ctx._load_handler))


# ---------------------------------------------------------------------------
# Tests: save_state / load_state handlers
# ---------------------------------------------------------------------------


class TestSaveLoadHandlers(unittest.TestCase):
    def setUp(self):
        _pkg._mw = None
        _pkg._context = None
        self.ctx = _StubContext()
        initialize(self.ctx)

    def test_save_returns_empty_dict_when_no_main_window(self):
        _pkg._mw = None
        result = self.ctx._save_handler()
        self.assertIsInstance(result, dict)

    def test_save_includes_depth_cue_strength_key(self):
        _pkg._mw = MagicMock()
        _pkg._mw.scene = None
        result = self.ctx._save_handler()
        self.assertIn("depth_cue_strength", result)

    def test_save_includes_3d_scale_key(self):
        _pkg._mw = MagicMock()
        _pkg._mw.scene = None
        result = self.ctx._save_handler()
        self.assertIn("3d_scale", result)

    def test_load_returns_none_safely_when_no_data(self):
        _pkg._mw = MagicMock()
        _pkg._mw.scene = None
        # Should not raise
        self.ctx._load_handler(None)

    def test_load_returns_none_safely_when_no_main_window(self):
        _pkg._mw = None
        self.ctx._load_handler({"depth_cue_strength": 0.5})

    def test_save_load_roundtrip_preserves_depth_cue_strength(self):
        _pkg._mw = MagicMock()
        _pkg._mw.scene = None
        _pkg._depth_cue_strength = 0.6
        saved = self.ctx._save_handler()
        self.assertEqual(saved["depth_cue_strength"], 0.6)


# ---------------------------------------------------------------------------
# Real PluginContext tier
# ---------------------------------------------------------------------------

# ci.yml checks out the plugin into moleditpy_3d-molecule-on-2d/
# and clones the main app next to it.
_MAIN_APP_CANDIDATES = [
    # Local dev: DEV_MAIN/python_molecular_editor/moleditpy/src
    os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "python_molecular_editor",
            "moleditpy",
            "src",
        )
    ),
    # CI: checked out as ./python_molecular_editor next to the plugin dir
    os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "python_molecular_editor",
            "moleditpy",
            "src",
        )
    ),
    os.environ.get("CI_MAIN_APP_SRC", ""),
]
_MAIN_APP_SRC = next(
    (p for p in _MAIN_APP_CANDIDATES if p and os.path.isdir(p)),
    None,
)
HAS_MAIN_APP = _MAIN_APP_SRC is not None

try:
    import pytest

    _skipif = pytest.mark.skipif(
        not HAS_MAIN_APP,
        reason="main app not found; set CI_MAIN_APP_SRC or ensure python_molecular_editor is cloned",
    )
except ImportError:

    def _skipif(cls):
        return unittest.skip("pytest not available")(cls)


def _clear_qt_stubs():
    """Remove fake PyQt6 stub modules so real PyQt6 can be imported by moleditpy."""
    to_remove = [
        k
        for k in list(sys.modules)
        if k.startswith("PyQt6") and not hasattr(sys.modules[k], "__file__")
    ]
    for k in to_remove:
        del sys.modules[k]
    # Clear any moleditpy import that may have been attempted with stubs
    for k in [k for k in list(sys.modules) if k.startswith("moleditpy")]:
        del sys.modules[k]


@_skipif
class TestWithRealPluginContext(unittest.TestCase):
    """Verify initialize() works with the actual MoleditPy PluginContext."""

    @classmethod
    def setUpClass(cls):
        if not HAS_MAIN_APP:
            return
        # Load plugin_interface.py directly to avoid triggering moleditpy/__init__.py
        # which imports PyQt6 and conflicts with PySide6 loaded by pytest-qt on Windows.
        import importlib.util as _ilu

        _pi_path = os.path.join(
            _MAIN_APP_SRC, "moleditpy", "plugins", "plugin_interface.py"
        )
        _spec = _ilu.spec_from_file_location(
            "moleditpy.plugins.plugin_interface", _pi_path
        )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        cls.PluginContext = _mod.PluginContext
        mock_manager = MagicMock()
        mw = MagicMock()
        mw.menuBar.return_value.actions.return_value = []
        mw.scene = None
        mock_manager.get_main_window.return_value = mw
        cls.real_ctx = cls.PluginContext(mock_manager, PLUGIN_NAME)

    def test_real_initialize_does_not_raise(self):
        _pkg._mw = None
        _pkg._context = None
        try:
            initialize(self.real_ctx)
        except Exception as e:
            self.fail(f"initialize(real_context) raised: {e}")

    def test_real_context_is_plugincontext_instance(self):
        self.assertIsInstance(self.real_ctx, self.PluginContext)

    def test_stub_interface_matches_real(self):
        for method in [
            "add_menu_action",
            "register_save_handler",
            "register_load_handler",
            "get_main_window",
        ]:
            self.assertTrue(
                hasattr(self.PluginContext, method),
                f"Real PluginContext missing: {method}",
            )


if __name__ == "__main__":
    unittest.main()
