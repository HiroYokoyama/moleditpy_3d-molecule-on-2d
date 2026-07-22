"""
Coverage tests for the paint patching, z-range helpers, molecule graph
discovery, and monkeypatch install/restore logic in 3d_molecule_on_2d.py.
"""

import os
import sys
import importlib.util
import unittest
from unittest.mock import MagicMock

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../../python_molecular_editor/moleditpy/src"
        )
    ),
)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _restore_real_pyqt6():
    """Undo any PyQt6/sip stubbing left behind by other test modules (e.g.
    test_export.py replaces PyQt6.QtCore with a bare MagicMock module that
    lacks real symbols like QRectF); tests here need the genuine bindings."""
    to_remove = []
    for k, v in list(sys.modules.items()):
        if k.startswith("PyQt6") or k in ("sip", "_3d_molecule_on_2d_render"):
            if not hasattr(v, "__file__") or "mock" in str(type(v)).lower():
                to_remove.append(k)
    for k in to_remove:
        del sys.modules[k]


_restore_real_pyqt6()

from PyQt6.QtWidgets import QApplication

_PLUGIN_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "3d_molecule_on_2d.py")
)


def _load_plugin():
    _restore_real_pyqt6()
    spec = importlib.util.spec_from_file_location(
        "_3d_molecule_on_2d_render", _PLUGIN_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_3d_molecule_on_2d_render"] = mod
    spec.loader.exec_module(mod)
    return mod


class AtomItem:
    def __init__(self, atom_id, z=0.0, symbol="C", mol_z_min=None, mol_z_max=None):
        self.atom_id = atom_id
        self.symbol = symbol
        self.z_3d = z
        if mol_z_min is not None:
            self.mol_z_min = mol_z_min
        if mol_z_max is not None:
            self.mol_z_max = mol_z_max
        self._pos = MagicMock()
        self._pos.x.return_value = 0.0
        self._pos.y.return_value = 0.0
        self._scene = None

    def pos(self):
        return self._pos

    def scene(self):
        return self._scene


class BondItem:
    def __init__(self, atom1, atom2):
        self.atom1 = atom1
        self.atom2 = atom2
        self._scene = None

    def scene(self):
        return self._scene


class FakeScene:
    def __init__(self, items=None, data=None):
        self._items = items or []
        self.data = data
        self.atom_items = {}

    def items(self):
        return self._items

    def views(self):
        return []


class TestGetSceneZRange(unittest.TestCase):
    def setUp(self):
        self.pkg = _load_plugin()

    def test_no_data_attr_returns_default_range(self):
        scene = MagicMock(spec=[])
        z_min, z_max = self.pkg.get_scene_z_range(scene)
        self.assertEqual((z_min, z_max), (-5.0, 5.0))

    def test_empty_atoms_returns_default(self):
        scene = FakeScene(data=MagicMock(atoms={}))
        z_min, z_max = self.pkg.get_scene_z_range(scene)
        self.assertEqual((z_min, z_max), (-5.0, 5.0))

    def test_atom_items_path_computes_min_max(self):
        scene = FakeScene(data=MagicMock(atoms={1: {}, 2: {}}))
        scene.atom_items = {1: AtomItem(1, z=2.0), 2: AtomItem(2, z=8.0)}
        z_min, z_max = self.pkg.get_scene_z_range(scene)
        self.assertEqual((z_min, z_max), (2.0, 8.0))

    def test_falls_back_to_atom_data_item_when_no_atom_items(self):
        scene = FakeScene(
            data=MagicMock(
                atoms={1: {"item": AtomItem(1, z=3.0)}, 2: {"item": AtomItem(2, z=1.0)}}
            )
        )
        scene.atom_items = None
        z_min, z_max = self.pkg.get_scene_z_range(scene)
        self.assertEqual((z_min, z_max), (1.0, 3.0))


class TestFindMolecules(unittest.TestCase):
    def setUp(self):
        self.pkg = _load_plugin()

    def test_no_scene_returns_empty(self):
        result = self.pkg.find_molecules(None)
        self.assertEqual(result, ([], [], []))

    def test_no_atoms_returns_empty(self):
        scene = FakeScene(items=[])
        result = self.pkg.find_molecules(scene)
        self.assertEqual(result, ([], [], []))

    def test_connected_components_grouped(self):
        a1, a2, a3 = AtomItem(1), AtomItem(2), AtomItem(3)
        b1 = BondItem(a1, a2)
        scene = FakeScene(items=[a1, a2, a3, b1])
        molecules, all_atoms, all_bonds = self.pkg.find_molecules(scene)
        self.assertEqual(len(molecules), 2)  # {a1,a2} and {a3}
        self.assertEqual(len(all_atoms), 3)
        self.assertEqual(len(all_bonds), 1)

    def test_deleted_items_skipped(self):
        a1 = AtomItem(1)
        self.pkg.sip_isdeleted_safe = lambda item: item is a1
        scene = FakeScene(items=[a1])
        molecules, all_atoms, all_bonds = self.pkg.find_molecules(scene)
        self.assertEqual(all_atoms, [])


class TestUpdateMoleculeZRanges(unittest.TestCase):
    def setUp(self):
        self.pkg = _load_plugin()

    def test_updates_z_min_max_on_atoms(self):
        a1, a2 = AtomItem(1, z=0.0), AtomItem(2, z=30.0)
        bond = BondItem(a1, a2)
        scene = FakeScene(items=[a1, a2, bond])
        self.pkg.update_molecule_z_ranges(scene)
        self.assertEqual(a1.mol_z_max, 30.0)
        self.assertEqual(a2.mol_z_max, 30.0)
        self.assertEqual(a1.mol_z_min, 30.0 - 30.0)

    def test_find_molecules_exception_returns_silently(self):
        scene = object()  # find_molecules will hit AttributeError
        self.pkg.update_molecule_z_ranges(scene)  # should not raise

    def test_empty_molecule_atoms_list_skipped(self):
        scene = FakeScene(items=[])
        self.pkg.update_molecule_z_ranges(scene)  # no molecules, no-op


class TestPaintPatches(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()

    def test_patched_atom_paint_falls_back_without_z(self):
        original = MagicMock()
        self.pkg._original_atom_paint = original
        item = AtomItem(1)
        item.z_3d = None
        self.pkg.patched_atom_paint(item, MagicMock(), MagicMock(), MagicMock())
        original.assert_called_once()

    def test_patched_atom_paint_depth_cue_applied(self):
        _restore_real_pyqt6()
        from moleditpy.ui import atom_item
        from PyQt6.QtGui import QColor

        # Other test modules may have left CPK_COLORS["C"] pointing at a mock
        # QColor from a stubbed PyQt6; force a known-good real QColor so the
        # blend/restore path under test behaves deterministically.
        known_color = QColor(200, 200, 200)
        atom_item.CPK_COLORS["C"] = known_color

        original = MagicMock()
        self.pkg._original_atom_paint = original
        self.pkg._show_depth_cues = True
        self.pkg._depth_cue_strength = 1.0

        item = AtomItem(1, z=0.0, mol_z_min=0.0, mol_z_max=10.0)
        item._scene = MagicMock()
        item._scene.views.return_value = []

        self.pkg.patched_atom_paint(item, MagicMock(), MagicMock(), MagicMock())
        original.assert_called_once()
        # Color should have been restored to the original afterward
        self.assertIs(atom_item.CPK_COLORS["C"], known_color)

    def test_patched_atom_paint_no_scene_falls_back(self):
        original = MagicMock()
        self.pkg._original_atom_paint = original
        item = AtomItem(1, z=1.0)
        item._scene = None
        self.pkg.patched_atom_paint(item, MagicMock(), MagicMock(), MagicMock())
        original.assert_called_once()

    def test_patched_bond_paint_no_original_returns(self):
        self.pkg._original_bond_paint = None
        a1, a2 = AtomItem(1), AtomItem(2)
        bond = BondItem(a1, a2)
        self.pkg.patched_bond_paint(bond, MagicMock(), MagicMock(), MagicMock())  # no crash

    def test_patched_bond_paint_falls_back_without_scene(self):
        original = MagicMock()
        self.pkg._original_bond_paint = original
        a1, a2 = AtomItem(1, z=1.0), AtomItem(2, z=2.0)
        bond = BondItem(a1, a2)
        bond._scene = None
        self.pkg.patched_bond_paint(bond, MagicMock(), MagicMock(), MagicMock())
        original.assert_called_once()

    def test_patched_bond_paint_depth_cue_no_window(self):
        original = MagicMock()
        self.pkg._original_bond_paint = original
        self.pkg._show_depth_cues = True
        self.pkg._depth_cue_strength = 1.0
        a1 = AtomItem(1, z=0.0, mol_z_min=0.0, mol_z_max=10.0)
        a2 = AtomItem(2, z=0.0, mol_z_min=0.0, mol_z_max=10.0)
        bond = BondItem(a1, a2)
        scene = MagicMock()
        scene.views.return_value = []  # no window -> falls back
        bond._scene = scene
        self.pkg.patched_bond_paint(bond, MagicMock(), MagicMock(), MagicMock())
        original.assert_called_once()

    def test_patched_bond_paint_depth_cue_with_window(self):
        original = MagicMock()
        self.pkg._original_bond_paint = original
        self.pkg._show_depth_cues = True
        self.pkg._depth_cue_strength = 1.0
        a1 = AtomItem(1, z=0.0, mol_z_min=0.0, mol_z_max=10.0)
        a2 = AtomItem(2, z=0.0, mol_z_min=0.0, mol_z_max=10.0)
        bond = BondItem(a1, a2)

        win = MagicMock()
        win.init_manager.settings = {"bond_color_2d": "#222222"}
        view = MagicMock()
        view.window.return_value = win
        scene = MagicMock()
        scene.views.return_value = [view]
        bond._scene = scene

        self.pkg.patched_bond_paint(bond, MagicMock(), MagicMock(), MagicMock())
        original.assert_called_once()
        # setting restored to original value afterward
        self.assertEqual(win.init_manager.settings["bond_color_2d"], "#222222")


class TestGetOriginalId(unittest.TestCase):
    def setUp(self):
        self.pkg = _load_plugin()

    def test_none_atom_returns_none(self):
        self.assertIsNone(self.pkg.get_original_id(None))

    def test_no_props_returns_none(self):
        atom = MagicMock()
        atom.HasProp.return_value = False
        self.assertIsNone(self.pkg.get_original_id(atom))

    def test_bad_value_falls_through_to_none(self):
        atom = MagicMock()
        atom.HasProp.side_effect = lambda name: name == "_original_atom_id"
        atom.GetProp.return_value = "not-an-int"
        self.assertIsNone(self.pkg.get_original_id(atom))

    def test_legacy_property_used_when_official_missing(self):
        atom = MagicMock()

        def has_prop(name):
            return name == "original_id"

        atom.HasProp.side_effect = has_prop
        atom.GetProp.return_value = "42"
        self.assertEqual(self.pkg.get_original_id(atom), 42)


class TestToggleMonkeyPatchesRealClasses(unittest.TestCase):
    """Exercises the real monkeypatch install/restore against the actual
    moleditpy classes (available since the main app repo is a sibling)."""

    def setUp(self):
        _restore_real_pyqt6()
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()
        # Some other test module may have left a patch installed (and never
        # restored it) from a *different* loaded copy of this plugin; clear
        # that out first so identity checks below are meaningful.
        self._unpatch_all()

    def _unpatch_all(self):
        try:
            self.pkg.toggle_monkey_patches(False)
        except Exception:
            pass
        try:
            self.pkg.patch_export_logic(False)
        except Exception:
            pass
        try:
            self.pkg.patch_state_logic(False)
        except Exception:
            pass

    def tearDown(self):
        # Always restore, regardless of test outcome
        self._unpatch_all()

    def test_toggle_monkey_patches_install_and_restore(self):
        from moleditpy.ui.atom_item import AtomItem as RealAtomItem
        from moleditpy.ui.bond_item import BondItem as RealBondItem

        original_atom_paint = RealAtomItem.paint
        original_bond_paint = RealBondItem.paint

        self.pkg.toggle_monkey_patches(True)
        self.assertIs(RealAtomItem.paint, self.pkg.patched_atom_paint)
        self.assertIs(RealBondItem.paint, self.pkg.patched_bond_paint)

        self.pkg.toggle_monkey_patches(False)
        self.assertIs(RealAtomItem.paint, original_atom_paint)
        self.assertIs(RealBondItem.paint, original_bond_paint)

    def test_patch_export_logic_install_and_restore(self):
        from moleditpy.core.molecular_data import MolecularData

        orig_to_rdkit = MolecularData.to_rdkit_mol
        orig_to_mol_block = MolecularData.to_mol_block

        self.pkg.patch_export_logic(True)
        self.assertIs(MolecularData.to_rdkit_mol, self.pkg.patched_to_rdkit_mol)
        self.assertIs(MolecularData.to_mol_block, self.pkg.patched_to_mol_block)

        self.pkg.patch_export_logic(False)
        self.assertIs(MolecularData.to_rdkit_mol, orig_to_rdkit)
        self.assertIs(MolecularData.to_mol_block, orig_to_mol_block)

    def test_patch_state_logic_install_and_restore(self):
        from moleditpy.ui.app_state import StateManager

        orig_get = StateManager.get_current_state
        orig_set = StateManager.set_state_from_data

        self.pkg.patch_state_logic(True)
        self.assertIs(StateManager.get_current_state, self.pkg.patched_get_current_state)
        self.assertIs(
            StateManager.set_state_from_data, self.pkg.patched_set_state_from_data
        )

        self.pkg.patch_state_logic(False)
        self.assertIs(StateManager.get_current_state, orig_get)
        self.assertIs(StateManager.set_state_from_data, orig_set)


class TestSipIsdeletedSafe(unittest.TestCase):
    def setUp(self):
        self.pkg = _load_plugin()

    def test_returns_false_on_exception(self):
        try:
            from PyQt6 import sip
        except ImportError:
            import sip

        old = sip.isdeleted
        try:
            sip.isdeleted = lambda obj: (_ for _ in ()).throw(RuntimeError("boom"))
            self.assertFalse(self.pkg.sip_isdeleted_safe(object()))
        finally:
            sip.isdeleted = old

    def test_returns_actual_value(self):
        self.assertFalse(self.pkg.sip_isdeleted_safe(object()))


if __name__ == "__main__":
    unittest.main()
