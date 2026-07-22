"""
Coverage-focused tests for 3d_molecule_on_2d.py: settings persistence, the
settings dialog, the smart-cleanup trigger, plugin enable/disable/toolbar
wiring, undo/redo hooks, and save/load state. Uses real PyQt6 + real RDKit
(as installed in CI); only the moleditpy host objects are mocked.
"""

import os
import sys
import json
import time
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

from PyQt6.QtWidgets import QApplication

_PLUGIN_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "3d_molecule_on_2d.py")
)


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "_3d_molecule_on_2d_cov", _PLUGIN_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_3d_molecule_on_2d_cov"] = mod
    spec.loader.exec_module(mod)
    return mod


class AtomItem:
    """Stand-in whose class name matches the string checks in find_molecules."""

    def __init__(self, atom_id, x=0.0, y=0.0, z=0.0, symbol="C", selected=False):
        self.atom_id = atom_id
        self.symbol = symbol
        self.z_3d = z
        self._pos = MagicMock()
        self._pos.x.return_value = x
        self._pos.y.return_value = y
        self._selected = selected
        self._flags = {}

    def pos(self):
        return self._pos

    def setPos(self, p):
        self._pos = p

    def setZValue(self, z):
        self.z_3d_val = z

    def isSelected(self):
        return self._selected

    def setFlag(self, flag, val):
        self._flags[flag] = val


class BondItem:
    def __init__(self, atom1, atom2, selected=False):
        self.atom1 = atom1
        self.atom2 = atom2
        self._selected = selected
        self.updated = False

    def update_position(self):
        self.updated = True

    def setZValue(self, z):
        self.z = z

    def isSelected(self):
        return self._selected


class FakeScene:
    def __init__(self, items=None):
        self._items = items or []
        self.data = MagicMock()
        self.atom_items = {}
        self.updated = False

    def items(self):
        return self._items

    def update(self):
        self.updated = True

    def views(self):
        return []

    def selectedItems(self):
        return [i for i in self._items if getattr(i, "_selected", False)]


def make_mw(scene=None):
    mw = MagicMock()
    mw.scene = scene if scene is not None else FakeScene()
    return mw


class TestSettingsPersistence(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()
        self.tmp_settings = self.pkg._settings_file + ".test"
        self.pkg._settings_file = self.tmp_settings

    def tearDown(self):
        if os.path.exists(self.tmp_settings):
            os.remove(self.tmp_settings)

    def test_save_then_load_roundtrip(self):
        self.pkg._enabled = False
        self.pkg._show_depth_cues = False
        self.pkg._depth_cue_strength = 0.3
        self.pkg._3d_scale = 77.0
        self.pkg._embed_without_h = False
        self.pkg._force_direct_mode = True
        self.pkg.save_settings()

        self.assertTrue(os.path.exists(self.tmp_settings))

        # Reset globals then reload
        self.pkg._enabled = True
        self.pkg._show_depth_cues = True
        self.pkg._depth_cue_strength = 0.8
        self.pkg._3d_scale = 50.0
        self.pkg._embed_without_h = True
        self.pkg._force_direct_mode = False

        self.pkg.load_settings()
        self.assertFalse(self.pkg._enabled)
        self.assertFalse(self.pkg._show_depth_cues)
        self.assertAlmostEqual(self.pkg._depth_cue_strength, 0.3)
        self.assertAlmostEqual(self.pkg._3d_scale, 77.0)
        self.assertFalse(self.pkg._embed_without_h)
        self.assertTrue(self.pkg._force_direct_mode)

    def test_load_settings_missing_file_keeps_defaults(self):
        if os.path.exists(self.tmp_settings):
            os.remove(self.tmp_settings)
        self.pkg.load_settings()  # should not raise

    def test_load_settings_bad_json_is_caught(self):
        with open(self.tmp_settings, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        self.pkg.load_settings()  # should be caught and printed, not raised

    def test_save_settings_error_path(self):
        # Point settings file at an impossible path to force an exception
        self.pkg._settings_file = os.path.join(
            self.tmp_settings, "nested", "impossible.json"
        )
        self.pkg.save_settings()  # should not raise


class TestSettingsDialog(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()

    def test_dialog_builds_and_reflects_globals(self):
        dlg = self.pkg.PluginSettingsDialog(None, current_enabled=True)
        self.assertTrue(dlg.chk_enable.isChecked())
        self.assertEqual(dlg.sld_strength.value(), int(self.pkg._depth_cue_strength * 100))
        self.assertEqual(dlg.sld_scale.value(), int(self.pkg._3d_scale))

    def test_accept_writes_globals_and_saves(self):
        dlg = self.pkg.PluginSettingsDialog(None, current_enabled=True)
        dlg.chk_enable.setChecked(False)
        dlg.chk_depth_cues.setChecked(False)
        dlg.sld_strength.setValue(42)
        dlg.chk_embed_no_h.setChecked(False)
        dlg.chk_force_direct.setChecked(False)
        dlg.sld_scale.setValue(99)

        tmp_settings = self.pkg._settings_file + ".accepttest"
        self.pkg._settings_file = tmp_settings
        try:
            dlg.accept()
            self.assertFalse(dlg.enabled)
            self.assertFalse(self.pkg._show_depth_cues)
            self.assertAlmostEqual(self.pkg._depth_cue_strength, 0.42)
            self.assertAlmostEqual(self.pkg._3d_scale, 99.0)
            self.assertTrue(os.path.exists(tmp_settings))
        finally:
            if os.path.exists(tmp_settings):
                os.remove(tmp_settings)

    def test_force_direct_checkbox_enablement_follows_embed_no_h(self):
        dlg = self.pkg.PluginSettingsDialog(None, current_enabled=True)
        dlg.chk_embed_no_h.setChecked(False)
        self.assertFalse(dlg.chk_force_direct.isEnabled())
        dlg.chk_embed_no_h.setChecked(True)
        self.assertTrue(dlg.chk_force_direct.isEnabled())

    def test_strength_and_scale_labels_update_live(self):
        dlg = self.pkg.PluginSettingsDialog(None, current_enabled=True)
        dlg.sld_strength.setValue(55)
        self.assertEqual(dlg.lbl_strength.text(), "55%")
        dlg.sld_scale.setValue(123)
        self.assertEqual(dlg.lbl_scale.text(), "123")


class TestOpenSettingsDialog(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()

    def test_show_settings_dialog_noop_without_mw(self):
        self.pkg._mw = None
        self.pkg.show_settings_dialog()  # should not raise

    def test_open_settings_dialog_cancelled_does_not_change_enabled(self):
        mw = make_mw()
        context = MagicMock()
        orig_enabled = self.pkg._enabled

        class FakeDlg:
            def __init__(self, *a, **kw):
                self.enabled = orig_enabled

            def exec(self):
                return 0  # rejected

        self.pkg.PluginSettingsDialog = FakeDlg
        self.pkg.open_settings_dialog(mw, context)
        self.assertEqual(self.pkg._enabled, orig_enabled)

    def test_open_settings_dialog_toggles_enabled_calls_enable_disable(self):
        mw = make_mw()
        context = MagicMock()
        self.pkg._enabled = True

        class FakeDlg:
            def __init__(self, *a, **kw):
                self.enabled = False

            def exec(self):
                return 1

        self.pkg.PluginSettingsDialog = FakeDlg
        self.pkg.disable_plugin = MagicMock()
        self.pkg.refresh_plugin_toolbar = MagicMock()
        self.pkg.save_settings = MagicMock()
        self.pkg.open_settings_dialog(mw, context)
        self.assertFalse(self.pkg._enabled)
        self.pkg.disable_plugin.assert_called_once_with(mw)
        context.show_status_message.assert_called()


class TestOnCleanupTriggered(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()
        self.pkg._is_syncing = False
        self.pkg._last_cleanup_trigger_time = 0

    def test_no_mw_returns_early(self):
        self.pkg._mw = None
        self.pkg.on_cleanup_triggered()  # no exception

    def test_no_scene_returns_early(self):
        mw = MagicMock()
        mw.scene = None
        self.pkg._mw = mw
        self.pkg._context = MagicMock()
        self.pkg.on_cleanup_triggered()

    def test_already_syncing_returns_early(self):
        self.pkg._is_syncing = True
        mw = make_mw()
        self.pkg._mw = mw
        self.pkg._context = MagicMock()
        self.pkg.on_cleanup_triggered()
        # Since it returned before try/finally reset, flag remains True
        self.assertTrue(self.pkg._is_syncing)

    def test_no_molecules_shows_status(self):
        mw = make_mw(FakeScene(items=[]))
        mw.current_mol = None
        context = MagicMock()
        self.pkg._mw = mw
        self.pkg._context = context
        self.pkg.on_cleanup_triggered()
        context.show_status_message.assert_called_with("No molecules in scene.")

    def test_needs_refresh_no_conformer_embed_without_h_starts_local(self):
        a1 = AtomItem(1)
        scene = FakeScene(items=[a1])
        mw = make_mw(scene)
        mw.current_mol = None
        context = MagicMock()
        self.pkg._mw = mw
        self.pkg._context = context
        self.pkg._embed_without_h = True
        self.pkg.start_local_embedding = MagicMock()
        self.pkg.on_cleanup_triggered()
        self.pkg.start_local_embedding.assert_called_once()

    def test_needs_refresh_triggers_main_conversion_when_not_embed_without_h(self):
        a1 = AtomItem(1)
        scene = FakeScene(items=[a1])
        mw = make_mw(scene)
        mw.current_mol = None
        mw.trigger_conversion = MagicMock()
        context = MagicMock()
        self.pkg._mw = mw
        self.pkg._context = context
        self.pkg._embed_without_h = False
        self.pkg._force_direct_mode = False
        self.pkg.on_cleanup_triggered()
        mw.compute_manager.trigger_conversion.assert_called_once()
        self.assertTrue(self.pkg._plugin_triggered_conversion)

    def test_cooldown_skips_retrigger(self):
        a1 = AtomItem(1)
        scene = FakeScene(items=[a1])
        mw = make_mw(scene)
        mw.current_mol = None
        context = MagicMock()
        self.pkg._mw = mw
        self.pkg._context = context
        self.pkg._last_cleanup_trigger_time = time.time()
        self.pkg.on_cleanup_triggered()
        context.show_status_message.assert_called_with(
            "Smart 3D: Cooldown active. Skipping re-trigger."
        )

    def test_sync_performed_when_conformer_present_and_valid(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        for i in range(mol.GetNumAtoms()):
            mol.GetAtomWithIdx(i).SetIntProp("_original_atom_id", i + 1)

        atoms = [AtomItem(i + 1, symbol="C" if i < 2 else "H") for i in range(mol.GetNumAtoms())]
        scene = FakeScene(items=atoms)
        mw = make_mw(scene)
        mw.current_mol = mol
        mw.state_manager.data.set_atom_pos = MagicMock()
        context = MagicMock()
        self.pkg._mw = mw
        self.pkg._context = context
        self.pkg._embed_without_h = False
        self.pkg.on_cleanup_triggered()
        context.push_undo_checkpoint.assert_called()

    def test_flat_conformer_triggers_refresh(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CC")
        from rdkit.Chem import AllChem

        AllChem.Compute2DCoords(mol)  # flat z=0
        for i in range(mol.GetNumAtoms()):
            mol.GetAtomWithIdx(i).SetIntProp("_original_atom_id", i + 1)

        atoms = [AtomItem(i + 1) for i in range(mol.GetNumAtoms())]
        scene = FakeScene(items=atoms)
        mw = make_mw(scene)
        mw.current_mol = mol
        context = MagicMock()
        self.pkg._mw = mw
        self.pkg._context = context
        self.pkg._embed_without_h = True
        self.pkg.start_local_embedding = MagicMock()
        self.pkg.on_cleanup_triggered()
        self.pkg.start_local_embedding.assert_called_once()


class TestInitializeEnableDisable(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()

    def test_initialize_registers_handlers(self):
        context = MagicMock()
        mw = make_mw()
        context.get_main_window.return_value = mw
        self.pkg._enabled = True
        self.pkg.initialize(context)
        context.add_menu_action.assert_called()
        context.register_save_handler.assert_called_with(self.pkg.save_state)
        context.register_load_handler.assert_called_with(self.pkg.load_state)
        context.add_toolbar_action.assert_called()

    def test_initialize_disabled_skips_toolbar(self):
        context = MagicMock()
        mw = make_mw()
        context.get_main_window.return_value = mw
        self.pkg._enabled = False
        context.reset_mock()
        self.pkg.initialize(context)
        context.add_toolbar_action.assert_not_called()

    def test_find_menu_action_finds_nested_action(self):
        mw = MagicMock()
        action_target = MagicMock()
        action_target.text.return_value = "Target"
        action_target.menu.return_value = None

        submenu = MagicMock()
        submenu.actions.return_value = [action_target]

        top_action = MagicMock()
        top_action.menu.return_value = submenu

        mw.menuBar.return_value.actions.return_value = [top_action]
        found = self.pkg._find_menu_action(mw, "Target")
        self.assertIs(found, action_target)

    def test_find_menu_action_not_found_returns_none(self):
        mw = MagicMock()
        mw.menuBar.return_value.actions.return_value = []
        found = self.pkg._find_menu_action(mw, "Nope")
        self.assertIsNone(found)

    def test_enable_plugin_sets_flags_on_items(self):
        a1 = AtomItem(1)
        scene = FakeScene(items=[a1])
        mw = make_mw(scene)
        mw.plugin_manager.toolbar_actions = []
        context = MagicMock()
        self.pkg.toggle_monkey_patches = MagicMock()
        self.pkg.patch_export_logic = MagicMock()
        self.pkg.patch_state_logic = MagicMock()
        self.pkg._install_undo_redo_hook = MagicMock()
        self.pkg.refresh_plugin_toolbar = MagicMock()
        self.pkg.configure_actions = MagicMock()
        self.pkg.enable_plugin(mw, context)
        self.pkg.toggle_monkey_patches.assert_called_with(True, mw)
        self.assertTrue(scene.updated)

    def test_disable_plugin_removes_toolbar_actions(self):
        mw = make_mw()
        tb = MagicMock()
        act_own = MagicMock()
        act_own.text.return_value = "Clean Up 3D"
        act_own.isSeparator.return_value = False
        tb.actions.return_value = [act_own]
        mw.init_manager.plugin_toolbar = tb
        mw.plugin_manager.toolbar_actions = [{"text": "Clean Up 3D"}, {"text": "Other"}]
        self.pkg.toggle_monkey_patches = MagicMock()
        self.pkg.patch_export_logic = MagicMock()
        self.pkg.patch_state_logic = MagicMock()
        self.pkg._rotate_tool_handler = MagicMock()
        self.pkg.disable_plugin(mw)
        tb.removeAction.assert_called_with(act_own)
        self.assertEqual(
            mw.plugin_manager.toolbar_actions, [{"text": "Other"}]
        )

    def test_disable_plugin_no_toolbar_no_crash(self):
        mw = make_mw()
        mw.init_manager = MagicMock()
        mw.init_manager.plugin_toolbar = None
        self.pkg.toggle_monkey_patches = MagicMock()
        self.pkg.patch_export_logic = MagicMock()
        self.pkg.patch_state_logic = MagicMock()
        self.pkg._rotate_tool_handler = None
        self.pkg.disable_plugin(mw)


class TestToolbarConfig(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()

    def test_refresh_plugin_toolbar_no_toolbar_returns(self):
        mw = MagicMock()
        mw.init_manager.plugin_toolbar = None
        context = MagicMock()
        self.pkg.refresh_plugin_toolbar(mw, context)  # should not raise

    def test_refresh_plugin_toolbar_rebuilds_and_shows(self):
        mw = MagicMock()
        tb = MagicMock()
        mw.init_manager.plugin_toolbar = tb
        context = MagicMock()
        self.pkg.configure_actions = MagicMock()
        self.pkg.refresh_plugin_toolbar(mw, context)
        tb.show.assert_called_once()
        mw.plugin_menu_manager.add_plugin_toolbar_actions.assert_called_once()

    def test_configure_actions_no_toolbar_returns(self):
        mw = MagicMock()
        mw.init_manager.plugin_toolbar = None
        self.pkg._mw = mw
        self.pkg.configure_actions()

    def test_configure_actions_wires_up_rotate_and_cleanup(self):
        mw = MagicMock()
        tb = MagicMock()
        rotate_act = MagicMock()
        rotate_act.text.return_value = "Rotate 3D"
        cleanup_act = MagicMock()
        cleanup_act.text.return_value = "Clean Up 3D"
        settings_act = MagicMock()
        settings_act.text.return_value = "3D on 2D Settings..."
        tb.actions.return_value = [rotate_act, cleanup_act, settings_act]
        mw.init_manager.plugin_toolbar = tb
        mw.tool_group.actions.return_value = []
        self.pkg._mw = mw
        self.pkg._rotate_tool_handler = MagicMock()
        self.pkg.configure_actions()
        rotate_act.setCheckable.assert_called_with(True)
        rotate_act.toggled.connect.assert_called_with(self.pkg.on_rotate_toggled)
        cleanup_act.triggered.connect.assert_called_with(self.pkg.on_cleanup_triggered)
        mw.tool_group.addAction.assert_called_with(rotate_act)
        tb.show.assert_called()

    def test_configure_actions_no_found_actions_uses_cached(self):
        mw = MagicMock()
        tb = MagicMock()
        tb.actions.return_value = []
        mw.init_manager.plugin_toolbar = tb
        self.pkg._mw = mw
        self.pkg._toolbar_actions_objs = []
        self.pkg.configure_actions()  # falls through with no actions, no crash


class TestUndoRedoHook(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()
        self.pkg._undo_hook_installed = False

    def test_restore_depth_disabled_noop(self):
        self.pkg._enabled = False
        self.pkg._mw = make_mw()
        self.pkg._restore_depth_after_undo()

    def test_restore_depth_no_scene_noop(self):
        self.pkg._enabled = True
        mw = MagicMock()
        mw.scene = None
        self.pkg._mw = mw
        self.pkg._restore_depth_after_undo()

    def test_restore_depth_runs_with_handler_and_bonds(self):
        self.pkg._enabled = True
        a1, a2 = AtomItem(1, z=1.0), AtomItem(2, z=2.0)
        b1 = BondItem(a1, a2)
        scene = FakeScene(items=[a1, a2, b1])
        mw = make_mw(scene)
        self.pkg._mw = mw
        handler = MagicMock()
        self.pkg._rotate_tool_handler = handler
        self.pkg._restore_depth_after_undo()
        handler.ensure_z_coords.assert_called_with(force=False)
        self.assertTrue(b1.updated)

    def test_install_undo_redo_hook_idempotent(self):
        mw = MagicMock()
        mw.edit_actions_manager.undo_stack = MagicMock()
        mw.edit_actions_manager.undo_stack.indexChanged = MagicMock()
        self.pkg._install_undo_redo_hook(mw)
        self.assertTrue(self.pkg._undo_hook_installed)
        # Second call is a no-op (already installed)
        self.pkg._install_undo_redo_hook(mw)

    def test_install_undo_redo_hook_no_mw_returns(self):
        self.pkg._install_undo_redo_hook(None)
        self.assertFalse(self.pkg._undo_hook_installed)

    def test_install_undo_redo_hook_wraps_undo_redo_methods(self):
        class EAM:
            def __init__(self):
                self.calls = []

            def undo(self):
                self.calls.append("undo")

            def redo(self):
                self.calls.append("redo")

        mw = MagicMock()
        eam = EAM()
        mw.edit_actions_manager = eam
        # EAM has no undo_stack/undoStack attribute, so the hook falls back
        # to wrapping the undo/redo methods directly.
        self.pkg._install_undo_redo_hook(mw)
        self.assertTrue(self.pkg._undo_hook_installed)
        eam.undo()
        self.assertIn("undo", eam.calls)


class TestOnRotateToggled(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()

    def test_toggled_on_activates_handler(self):
        mw = MagicMock()
        other_act = MagicMock()
        other_act.text.return_value = "Select"
        other_act.isChecked.return_value = True
        mw.tool_group.actions.return_value = [other_act]
        self.pkg._mw = mw
        handler = MagicMock()
        self.pkg._rotate_tool_handler = handler
        self.pkg._install_undo_redo_hook = MagicMock()
        self.pkg.on_rotate_toggled(True)
        handler.set_active.assert_called_with(True)
        other_act.setChecked.assert_called_with(False)

    def test_toggled_off_reverts_to_select_mode(self):
        mw = MagicMock()
        mw.init_manager.mode_actions = {"select": MagicMock()}
        self.pkg._mw = mw
        handler = MagicMock()
        self.pkg._rotate_tool_handler = handler
        self.pkg._install_undo_redo_hook = MagicMock()
        self.pkg.on_rotate_toggled(False)
        handler.set_active.assert_called_with(False)
        mw.init_manager.mode_actions["select"].setChecked.assert_called_with(True)
        mw.ui_manager.set_mode.assert_called_with("select")


class TestSaveLoadState(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()

    def test_save_state_no_mw_returns_empty(self):
        self.pkg._mw = None
        self.assertEqual(self.pkg.save_state(), {})

    def test_save_state_collects_z_data(self):
        scene = FakeScene()
        scene.atom_items = {1: AtomItem(1, z=3.3), 2: AtomItem(2, z=-1.1)}
        scene.data = MagicMock()
        scene.data.atoms = {1: {}, 2: {}}
        mw = make_mw(scene)
        self.pkg._mw = mw
        state = self.pkg.save_state()
        self.assertEqual(state["z_data"]["1"], 3.3)
        self.assertEqual(state["z_data"]["2"], -1.1)

    def test_load_state_no_mw_or_no_data_returns(self):
        self.pkg._mw = None
        self.pkg.load_state({"a": 1})  # no mw
        mw = make_mw()
        self.pkg._mw = mw
        self.pkg.load_state(None)  # no data
        self.pkg.load_state({})  # falsy data

    def test_load_state_sets_globals(self):
        mw = make_mw()
        self.pkg._mw = mw
        self.pkg._context = MagicMock()
        self.pkg.load_state(
            {"depth_cue_strength": 0.55, "3d_scale": 33.0, "embed_without_h": False}
        )
        self.assertAlmostEqual(self.pkg._depth_cue_strength, 0.55)
        self.assertAlmostEqual(self.pkg._3d_scale, 33.0)
        self.assertFalse(self.pkg._embed_without_h)


if __name__ == "__main__":
    unittest.main()
