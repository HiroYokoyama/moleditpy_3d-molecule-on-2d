import os
import sys
import unittest
import importlib.util
from unittest.mock import MagicMock
import numpy as np
import types

# Set up paths to import the plugin and host app
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../../python_molecular_editor/moleditpy/src"
        )
    ),
)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _restore_real_pyqt6():
    import sys

    to_remove = []
    for k, v in list(sys.modules.items()):
        if k.startswith("PyQt6") or k == "sip" or k == "_3d_molecule_on_2d":
            if (
                not hasattr(v, "__file__")
                or "mock" in str(type(v)).lower()
                or k == "_3d_molecule_on_2d"
            ):
                to_remove.append(k)
    for k in to_remove:
        del sys.modules[k]


_restore_real_pyqt6()

# Check if real PyQt6 is available
HAS_REAL_PYQT6 = False
if "PyQt6" in sys.modules:
    pyqt6_mod = sys.modules["PyQt6"]
    if hasattr(pyqt6_mod, "__file__"):
        HAS_REAL_PYQT6 = True
else:
    try:
        import PyQt6

        HAS_REAL_PYQT6 = True
    except ImportError:
        HAS_REAL_PYQT6 = False

if not HAS_REAL_PYQT6:
    # Set up stubs so that importing the plugin doesn't fail on CI
    if "sip" not in sys.modules:
        sip_stub = types.ModuleType("sip")
        sip_stub.isdeleted = lambda obj: False
        sys.modules["sip"] = sip_stub

    if "PyQt6" not in sys.modules:
        pyqt6 = types.ModuleType("PyQt6")

        qt_core = types.ModuleType("PyQt6.QtCore")
        for name in [
            "Qt",
            "QPointF",
            "QEvent",
            "QObject",
            "QTimer",
            "pyqtSignal",
            "QThread",
        ]:
            setattr(qt_core, name, MagicMock)

        qt_widgets = types.ModuleType("PyQt6.QtWidgets")
        for name in [
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
            setattr(qt_widgets, name, MagicMock)

        qt_gui = types.ModuleType("PyQt6.QtGui")
        qt_gui.QColor = MagicMock

        pyqt6.QtCore = qt_core
        pyqt6.QtWidgets = qt_widgets
        pyqt6.QtGui = qt_gui

        sys.modules["PyQt6"] = pyqt6
        sys.modules["PyQt6.QtCore"] = qt_core
        sys.modules["PyQt6.QtWidgets"] = qt_widgets
        sys.modules["PyQt6.QtGui"] = qt_gui
        sys.modules["PyQt6.sip"] = sip_stub

# Import the plugin dynamically since its filename starts with a number
_PLUGIN_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "3d_molecule_on_2d.py")
)
_spec = importlib.util.spec_from_file_location("_3d_molecule_on_2d", _PLUGIN_PATH)
_pkg = importlib.util.module_from_spec(_spec)
sys.modules["_3d_molecule_on_2d"] = _pkg
_spec.loader.exec_module(_pkg)


@unittest.skipUnless(HAS_REAL_PYQT6, "Requires real PyQt6 installed")
class TestCoreLogic(unittest.TestCase):
    def setUp(self):
        if HAS_REAL_PYQT6:
            # Restore real PyQt6 modules if they were stubbed by other tests
            _restore_real_pyqt6()
            # Re-import/reload the plugin to make sure it refers to real PyQt6 classes
            global _pkg
            _spec = importlib.util.spec_from_file_location(
                "_3d_molecule_on_2d", _PLUGIN_PATH
            )
            _pkg = importlib.util.module_from_spec(_spec)
            sys.modules["_3d_molecule_on_2d"] = _pkg
            _spec.loader.exec_module(_pkg)

    def test_blend_with_white(self):
        from PyQt6.QtGui import QColor

        # Test factor 0.0 (returns original color)
        c1 = QColor(100, 150, 200)
        res1 = _pkg.blend_with_white(c1, 0.0)
        self.assertEqual(res1.red(), 100)
        self.assertEqual(res1.green(), 150)
        self.assertEqual(res1.blue(), 200)

        # Test factor 1.0 (returns white)
        res2 = _pkg.blend_with_white(c1, 1.0)
        self.assertEqual(res2.red(), 255)
        self.assertEqual(res2.green(), 255)
        self.assertEqual(res2.blue(), 255)

        # Test intermediate factor (0.5)
        res3 = _pkg.blend_with_white(c1, 0.5)
        self.assertAlmostEqual(res3.red(), 177, delta=1)
        self.assertAlmostEqual(res3.green(), 202, delta=1)
        self.assertAlmostEqual(res3.blue(), 227, delta=1)

    def test_local_calculation_worker(self):
        from PyQt6.QtWidgets import QApplication
        from rdkit import Chem
        from rdkit.Chem import AllChem

        app = QApplication.instance() or QApplication([])

        # Generate a standard 2D MOL block for Ethane
        mol = Chem.MolFromSmiles("CC")
        AllChem.Compute2DCoords(mol)
        mol_block = Chem.MolToMolBlock(mol)
        atom_ids = [101, 102]

        # 1. Test standard embedding (embed_without_h=False)
        worker = _pkg.LocalCalculationWorker(
            mol_block, embed_without_h=False, force_direct_mode=False, atom_ids=atom_ids
        )

        status_messages = []
        finished_mols = []
        errors = []

        worker.status.connect(status_messages.append)
        worker.error.connect(errors.append)
        worker.finished.connect(finished_mols.append)

        worker.run()

        self.assertEqual(len(errors), 0, f"Worker failed with error: {errors}")
        self.assertEqual(len(finished_mols), 1)
        out_mol = finished_mols[0]
        self.assertGreater(out_mol.GetNumAtoms(), 2)  # Should contain added hydrogens

        # Verify mapping of original IDs to the embedded molecule
        self.assertEqual(out_mol.GetAtomWithIdx(0).GetIntProp("_original_atom_id"), 101)
        self.assertEqual(out_mol.GetAtomWithIdx(1).GetIntProp("_original_atom_id"), 102)

        # 2. Test embedding without hydrogens (embed_without_h=True)
        worker_no_h = _pkg.LocalCalculationWorker(
            mol_block, embed_without_h=True, force_direct_mode=False, atom_ids=atom_ids
        )
        finished_mols_no_h = []
        worker_no_h.finished.connect(finished_mols_no_h.append)
        worker_no_h.run()

        out_mol_no_h = finished_mols_no_h[0]
        self.assertEqual(out_mol_no_h.GetNumAtoms(), 2)  # Hydrogens stripped/removed
        self.assertEqual(
            out_mol_no_h.GetAtomWithIdx(0).GetIntProp("_original_atom_id"), 101
        )

        # 3. Test force_direct_mode fallback
        worker_direct = _pkg.LocalCalculationWorker(
            mol_block, embed_without_h=True, force_direct_mode=True, atom_ids=atom_ids
        )
        finished_mols_direct = []
        worker_direct.finished.connect(finished_mols_direct.append)
        worker_direct.run()

        out_mol_direct = finished_mols_direct[0]
        self.assertEqual(out_mol_direct.GetNumAtoms(), 2)

    def test_rotation_math(self):
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QPointF

        app = QApplication.instance() or QApplication([])

        # Mock Atom Item that behaves like PyQt6 graphics item
        class MockAtomItem:
            def __init__(self, x, y, z, atom_id):
                self._pos = QPointF(x, y)
                self.z_3d = z
                self.atom_id = atom_id
                self._z_value = z

            def pos(self):
                return self._pos

            def setPos(self, pos):
                self._pos = pos

            def setZValue(self, z):
                self._z_value = z

            def scene(self):
                return mock_scene

        # Mock scene containing the items
        class MockScene:
            def __init__(self):
                self.data = MagicMock()
                self._items = []

            def items(self):
                return self._items

            def update(self):
                pass

        mock_scene = MockScene()
        atom1 = MockAtomItem(0.0, 0.0, 10.0, 1)
        atom2 = MockAtomItem(10.0, 0.0, 10.0, 2)
        mock_scene._items = [atom1, atom2]

        # Mock Host Main Window
        mw = MagicMock()
        mw.scene = mock_scene
        mw.init_manager.view_2d.viewport.return_value = MagicMock()

        handler = _pkg.RotateToolHandler(mw)
        handler.target_atoms = [atom1, atom2]

        # Calculate initial distance in 3D
        p1 = np.array([atom1.pos().x(), atom1.pos().y(), atom1.z_3d])
        p2 = np.array([atom2.pos().x(), atom2.pos().y(), atom2.z_3d])
        init_dist = np.linalg.norm(p1 - p2)

        # Rotate molecule around its COG (yaw = 90 deg / pi/2 rad)
        handler.rotate_molecule(np.pi / 2, 0.0)

        # Calculate new distance
        p1_new = np.array([atom1.pos().x(), atom1.pos().y(), atom1.z_3d])
        p2_new = np.array([atom2.pos().x(), atom2.pos().y(), atom2.z_3d])
        new_dist = np.linalg.norm(p1_new - p2_new)

        # Distance must be preserved (rigid rotation)
        self.assertAlmostEqual(new_dist, init_dist, places=5)

        # Center of gravity must be preserved
        cog_init = (p1 + p2) / 2.0
        cog_new = (p1_new + p2_new) / 2.0
        np.testing.assert_allclose(cog_init, cog_new, atol=1e-5)

        # Verify that Z coordinates rotated (are no longer both exactly 10.0)
        self.assertNotAlmostEqual(atom1.z_3d, 10.0, places=5)


class _FakeAtom:
    """Atom graphics-item stand-in. `has_z` False mimics a freshly recreated
    item that does not yet carry the plugin's dynamic z_3d attribute."""

    def __init__(self, z=None):
        self._zval = 0.0
        if z is not None:
            self.z_3d = z

    def setZValue(self, z):
        self._zval = z


class _FakeScene:
    def __init__(self, atom_items):
        self.atom_items = atom_items

    def items(self):
        return []

    def views(self):
        return []


class _FakeHost:
    def __init__(self, scene):
        self.scene = scene


class _FakeStateManager:
    """Chained-original stand-in: records what the wrapped set/get did."""

    def __init__(self, host, base_state=None):
        self.host = host
        self._base = base_state or {"atoms": {}, "bonds": {}}
        self.restored_with = None

    def _original_get_current_state(self):
        return dict(self._base)

    def _original_set_state_from_data(self, data):
        self.restored_with = data


class TestUndoZRestore(unittest.TestCase):
    """Regression tests for 3D depth (z_3d) surviving undo/redo, including the
    two-plugin case where set_state_from_data recreates atoms fresh."""

    def test_get_current_state_captures_z(self):
        atoms = {1: _FakeAtom(3.5), 2: _FakeAtom(-1.25)}
        sm = _FakeStateManager(_FakeHost(_FakeScene(atoms)))
        state = _pkg.patched_get_current_state(sm)
        self.assertIn("mol3d_on_2d_z", state)
        self.assertEqual(state["mol3d_on_2d_z"], {"1": 3.5, "2": -1.25})

    def test_get_current_state_omits_key_when_no_depth(self):
        atoms = {1: _FakeAtom()}  # no z_3d attribute
        sm = _FakeStateManager(_FakeHost(_FakeScene(atoms)))
        state = _pkg.patched_get_current_state(sm)
        self.assertNotIn("mol3d_on_2d_z", state)

    def test_set_state_restores_z_onto_fresh_atoms(self):
        # Fresh atoms have NO z_3d attribute (as after restore_atoms_and_bonds).
        # The restore must add it, not skip them (the "undo made it flat" bug).
        fresh = {1: _FakeAtom(), 2: _FakeAtom()}
        sm = _FakeStateManager(_FakeHost(_FakeScene(fresh)))
        _pkg.patched_set_state_from_data(sm, {"mol3d_on_2d_z": {"1": 4.2, "2": -3.1}})
        self.assertEqual(getattr(fresh[1], "z_3d", None), 4.2)
        self.assertEqual(getattr(fresh[2], "z_3d", None), -3.1)
        self.assertEqual(fresh[1]._zval, 4.2)

    def test_round_trip_capture_then_restore(self):
        # Capture depth, flatten (fresh atoms), restore from snapshot.
        src = {1: _FakeAtom(2.0), 2: _FakeAtom(-2.0)}
        snap = _pkg.patched_get_current_state(
            _FakeStateManager(_FakeHost(_FakeScene(src)))
        )
        fresh = {1: _FakeAtom(), 2: _FakeAtom()}
        _pkg.patched_set_state_from_data(
            _FakeStateManager(_FakeHost(_FakeScene(fresh))), snap
        )
        self.assertEqual(getattr(fresh[1], "z_3d", None), 2.0)
        self.assertEqual(getattr(fresh[2], "z_3d", None), -2.0)

    def test_two_plugin_chain_preserves_z(self):
        # Simulate the reaction-sketcher wrapping our patched functions: it adds
        # its own keys on capture and reloads its own items on restore, but must
        # not drop our z data.
        def rs_get(inner):
            state = inner()
            state["rs_items"] = ["arrow"]
            return state

        def rs_set(inner, data):
            inner(data)  # chained original (ours) runs first

        src = {1: _FakeAtom(5.0)}
        our_state = _pkg.patched_get_current_state(
            _FakeStateManager(_FakeHost(_FakeScene(src)))
        )
        wrapped = rs_get(lambda: our_state)
        self.assertIn("mol3d_on_2d_z", wrapped)
        self.assertIn("rs_items", wrapped)

        fresh = {1: _FakeAtom()}
        sm = _FakeStateManager(_FakeHost(_FakeScene(fresh)))
        rs_set(lambda d: _pkg.patched_set_state_from_data(sm, d), wrapped)
        self.assertEqual(getattr(fresh[1], "z_3d", None), 5.0)


if __name__ == "__main__":
    unittest.main()
