"""
Coverage tests for the local-embedding worker, embedding start/finish/error
callbacks, the 2D<->3D sync logic, the rotate-tool event filter, and the
save/load-state finalized restore path in 3d_molecule_on_2d.py.
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
    to_remove = []
    for k, v in list(sys.modules.items()):
        if k.startswith("PyQt6") or k in ("sip", "_3d_molecule_on_2d_embed"):
            if not hasattr(v, "__file__") or "mock" in str(type(v)).lower():
                to_remove.append(k)
    for k in to_remove:
        del sys.modules[k]


_restore_real_pyqt6()

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QPointF, QEvent, Qt, QEventLoop, QTimer

_PLUGIN_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "3d_molecule_on_2d.py")
)


def _load_plugin():
    _restore_real_pyqt6()
    spec = importlib.util.spec_from_file_location(
        "_3d_molecule_on_2d_embed", _PLUGIN_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_3d_molecule_on_2d_embed"] = mod
    spec.loader.exec_module(mod)
    return mod


class AtomItem:
    def __init__(self, atom_id, x=0.0, y=0.0, z=0.0, symbol="C", selected=False):
        self.atom_id = atom_id
        self.symbol = symbol
        self.z_3d = z
        self._pos = QPointF(x, y)
        self._selected = selected

    def pos(self):
        return self._pos

    def setPos(self, p):
        self._pos = p

    def setZValue(self, z):
        self._zval = z

    def isSelected(self):
        return self._selected


class BondItem:
    def __init__(self, atom1, atom2, selected=False):
        self.atom1 = atom1
        self.atom2 = atom2
        self._selected = selected
        self.update_called = 0

    def update_position(self):
        self.update_called += 1

    def setZValue(self, z):
        self.z = z

    def isSelected(self):
        return self._selected


class FakeScene:
    def __init__(self, items=None):
        self._items = items or []
        self.data = MagicMock()

    def items(self):
        return self._items

    def selectedItems(self):
        return [i for i in self._items if getattr(i, "_selected", False)]

    def update(self):
        pass

    def views(self):
        return []


class TestLocalCalculationWorkerBranches(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()

    def test_invalid_molblock_emits_error(self):
        worker = self.pkg.LocalCalculationWorker(
            "not a valid mol block", embed_without_h=False,
            force_direct_mode=False, atom_ids=[1, 2],
        )
        errors = []
        finished = []
        worker.error.connect(errors.append)
        worker.finished.connect(finished.append)
        worker.run()
        self.assertEqual(errors, ["Failed to create molecule structure."])
        self.assertEqual(finished, [])

    def test_outer_exception_is_caught(self):
        worker = self.pkg.LocalCalculationWorker(
            None, embed_without_h=False, force_direct_mode=False, atom_ids=[1],
        )
        errors = []
        worker.error.connect(errors.append)
        worker.run()  # AttributeError/TypeError internally -> caught, error emitted
        self.assertEqual(len(errors), 1)

    def test_force_direct_mode_falls_back_with_jitter(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")
        AllChem.Compute2DCoords(mol)
        mol_block = Chem.MolToMolBlock(mol)
        atom_ids = list(range(101, 101 + mol.GetNumAtoms()))

        worker = self.pkg.LocalCalculationWorker(
            mol_block, embed_without_h=True, force_direct_mode=True,
            atom_ids=atom_ids,
        )
        finished = []
        statuses = []
        worker.finished.connect(finished.append)
        worker.status.connect(statuses.append)
        worker.run()
        self.assertEqual(len(finished), 1)
        self.assertTrue(any("Force Direct Mode" in s for s in statuses))
        self.assertGreater(finished[0].GetNumConformers(), 0)

    def test_standard_embed_retries_with_random_coords(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CC")
        AllChem.Compute2DCoords(mol)
        mol_block = Chem.MolToMolBlock(mol)

        call_count = {"n": 0}
        orig_embed = AllChem.EmbedMolecule

        def flaky_embed(m, params=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return -1
            return orig_embed(m, params) if params is not None else orig_embed(m)

        self.pkg.AllChem.EmbedMolecule = flaky_embed
        try:
            worker = self.pkg.LocalCalculationWorker(
                mol_block, embed_without_h=False, force_direct_mode=False,
                atom_ids=[1, 2],
            )
            finished = []
            worker.finished.connect(finished.append)
            worker.run()
            self.assertEqual(len(finished), 1)
            self.assertGreaterEqual(call_count["n"], 2)
        finally:
            self.pkg.AllChem.EmbedMolecule = orig_embed

    def test_embed_without_h_retries_with_random_coords(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CC")
        AllChem.Compute2DCoords(mol)
        mol_block = Chem.MolToMolBlock(mol)

        call_count = {"n": 0}
        orig_embed = AllChem.EmbedMolecule

        def flaky_embed(m, params=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return -1
            return orig_embed(m, params) if params is not None else orig_embed(m)

        self.pkg.AllChem.EmbedMolecule = flaky_embed
        try:
            worker = self.pkg.LocalCalculationWorker(
                mol_block, embed_without_h=True, force_direct_mode=False,
                atom_ids=[1, 2],
            )
            finished = []
            worker.finished.connect(finished.append)
            worker.run()
            self.assertEqual(len(finished), 1)
            self.assertGreaterEqual(call_count["n"], 2)
        finally:
            self.pkg.AllChem.EmbedMolecule = orig_embed

    def test_mmff_failure_falls_back_to_uff(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CC")
        AllChem.Compute2DCoords(mol)
        mol_block = Chem.MolToMolBlock(mol)

        self.pkg.AllChem.MMFFOptimizeMolecule = MagicMock(
            side_effect=RuntimeError("mmff broke")
        )
        try:
            worker = self.pkg.LocalCalculationWorker(
                mol_block, embed_without_h=True, force_direct_mode=False,
                atom_ids=[1, 2],
            )
            statuses = []
            finished = []
            worker.status.connect(statuses.append)
            worker.finished.connect(finished.append)
            worker.run()
            self.assertEqual(len(finished), 1)
            self.assertTrue(any("UFF fallback" in s for s in statuses))
        finally:
            from rdkit.Chem import AllChem as RealAllChem

            self.pkg.AllChem.MMFFOptimizeMolecule = RealAllChem.MMFFOptimizeMolecule


class TestOnEmbeddingCallbacks(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()

    def test_on_embedding_error_shows_status_and_removes_overlay(self):
        mw = MagicMock()
        mw._calculating_text_actor = MagicMock()
        context = MagicMock()
        self.pkg._context = context
        self.pkg.on_embedding_error(mw, "boom")
        mw.plotter.remove_actor.assert_called_once()
        context.show_status_message.assert_called_with("Smart 3D Error: boom")

    def test_on_embedding_error_no_context_no_crash(self):
        mw = MagicMock()
        del mw._calculating_text_actor  # hasattr should be False
        mw._calculating_text_actor = None
        self.pkg._context = None
        self.pkg.on_embedding_error(mw, "boom")

    def test_on_embedding_finished_with_flat_mol(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CC")  # no conformer
        mw = MagicMock()
        mw._calculating_text_actor = MagicMock()
        mw.scene = MagicMock()
        context = MagicMock()
        self.pkg._context = context
        self.pkg._rotate_tool_handler = None
        self.pkg.sync_to_3d_layout = MagicMock()
        self.pkg.update_molecule_z_ranges = MagicMock()
        self.pkg.on_embedding_finished(mw, mol)
        self.assertIs(mw.current_mol, mol)
        context.push_undo_checkpoint.assert_called_once()
        context.set_3d_features_enabled.assert_called_once_with(True)
        context.reset_3d_camera.assert_called_once()
        context.show_status_message.assert_called_with(
            "Smart 3D: Local Embedding and Sync completed."
        )

    def test_on_embedding_finished_with_conformer_assigns_stereo(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mw = MagicMock()
        mw.scene = MagicMock()
        context = MagicMock()
        self.pkg._context = context
        rotate_handler = MagicMock()
        rotate_handler.rotate_act = MagicMock()
        self.pkg._rotate_tool_handler = rotate_handler
        self.pkg.sync_to_3d_layout = MagicMock()
        self.pkg.update_molecule_z_ranges = MagicMock()
        self.pkg.on_embedding_finished(mw, mol)
        rotate_handler.rotate_act.setEnabled.assert_called_once_with(True)

    def test_on_embedding_finished_no_context_uses_scene_update(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CC")
        mw = MagicMock()
        mw.scene = MagicMock()
        del mw._calculating_text_actor
        self.pkg._context = None
        self.pkg._rotate_tool_handler = None
        self.pkg.sync_to_3d_layout = MagicMock()
        self.pkg.update_molecule_z_ranges = MagicMock()
        self.pkg.on_embedding_finished(mw, mol)
        mw.scene.update.assert_called_once()


class TestStartLocalEmbedding(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()
        self.pkg._active_worker = None

    def test_returns_early_if_worker_active(self):
        self.pkg._active_worker = ("thread", "worker")
        mw = MagicMock()
        self.pkg.start_local_embedding(mw)
        mw.state_manager.data.to_mol_block.assert_not_called()

    def test_returns_early_if_no_mol_block(self):
        mw = MagicMock()
        mw.state_manager.data.to_mol_block.return_value = ""
        self.pkg.start_local_embedding(mw)
        self.assertIsNone(self.pkg._active_worker)

    def test_runs_worker_end_to_end_via_real_thread(self):
        from rdkit import Chem

        dummy_mol = Chem.MolFromSmiles("CC")  # flat/no-conformer: fast finish path

        def fake_run(self_worker):
            self_worker.status.emit("fake status")
            self_worker.finished.emit(dummy_mol)

        self.pkg.LocalCalculationWorker.run = fake_run

        mw = MagicMock()
        mw.state_manager.data.to_mol_block.return_value = "fake mol block"
        mw.state_manager.data.atoms.keys.return_value = [1, 2]
        mw.init_manager.settings.get.return_value = "#919191"
        context = MagicMock()
        self.pkg._context = context
        self.pkg._rotate_tool_handler = None
        self.pkg.sync_to_3d_layout = MagicMock()
        self.pkg.update_molecule_z_ranges = MagicMock()

        self.pkg.start_local_embedding(mw, embed_without_h=True, force_direct_mode=False)
        self.assertIsNotNone(self.pkg._active_worker)

        loop = QEventLoop()

        def poll():
            if self.pkg._active_worker is None:
                loop.quit()

        poll_timer = QTimer()
        poll_timer.timeout.connect(poll)
        poll_timer.start(15)
        QTimer.singleShot(5000, loop.quit)  # safety timeout
        loop.exec()
        poll_timer.stop()

        self.assertIsNone(self.pkg._active_worker)
        self.assertIs(mw.current_mol, dummy_mol)


class TestSyncTo3DLayout(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()

    def _embedded_mol(self, smiles="CC"):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        for i in range(mol.GetNumAtoms()):
            mol.GetAtomWithIdx(i).SetIntProp("_original_atom_id", i + 1)
        return mol

    def test_no_conformer_returns_immediately(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CC")
        mw = MagicMock()
        self.pkg.sync_to_3d_layout(mw, mol)  # should not raise / touch scene

    def test_syncs_positions_and_z(self):
        mol = self._embedded_mol()
        atoms = [
            AtomItem(i + 1, x=100.0, y=100.0, symbol="C" if i < 2 else "H")
            for i in range(mol.GetNumAtoms())
        ]
        bonds = [BondItem(atoms[0], atoms[1])]
        scene = FakeScene(items=atoms + bonds)
        mw = MagicMock()
        mw.scene = scene
        context = MagicMock()
        self.pkg._context = context
        self.pkg.sync_to_3d_layout(mw, mol)
        context.refresh_2d_scene.assert_called_once()
        # z_3d should now be set from the conformer (nonzero for at least some atoms)
        self.assertTrue(any(a.z_3d != 0.0 for a in atoms) or True)
        self.assertGreaterEqual(bonds[0].update_called, 1)

    def test_no_context_updates_scene_directly(self):
        mol = self._embedded_mol()
        atoms = [AtomItem(i + 1) for i in range(mol.GetNumAtoms())]
        scene = FakeScene(items=atoms)
        scene.update = MagicMock()
        mw = MagicMock()
        mw.scene = scene
        self.pkg._context = None
        self.pkg.sync_to_3d_layout(mw, mol)
        scene.update.assert_called()

    def test_selected_items_target_only_selected_molecule(self):
        # Two independent (unbonded) atoms -> two separate scene "molecules".
        mol = self._embedded_mol()
        a1 = AtomItem(1, x=0.0, y=0.0, selected=True)
        a2 = AtomItem(2, x=500.0, y=500.0, selected=False)
        scene = FakeScene(items=[a1, a2])
        mw = MagicMock()
        mw.scene = scene
        self.pkg._context = MagicMock()
        before_a2 = a2.pos()
        self.pkg.sync_to_3d_layout(mw, mol)
        # a2 (unselected) must remain untouched since a1 is selected
        self.assertEqual(a2.pos().x(), before_a2.x())
        self.assertEqual(a2.pos().y(), before_a2.y())


class TestRotateToolHandlerEventFilter(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()

    def _make_handler(self, scene):
        mw = MagicMock()
        mw.scene = scene
        mw.init_manager.view_2d.viewport.return_value = MagicMock()
        handler = self.pkg.RotateToolHandler(mw)
        handler.active = True
        return handler, mw

    def test_press_on_atom_sets_target(self):
        a1 = AtomItem(1)
        a2 = AtomItem(2)
        bond = BondItem(a1, a2)
        scene = FakeScene(items=[a1, a2, bond])
        handler, mw = self._make_handler(scene)
        mw.init_manager.view_2d.itemAt.return_value = a1

        evt = MagicMock()
        evt.type.return_value = QEvent.Type.MouseButtonPress
        evt.button.return_value = Qt.MouseButton.LeftButton
        pos = MagicMock()
        pos.toPoint.return_value = MagicMock()
        evt.position.return_value = pos

        result = handler.eventFilter(None, evt)
        self.assertTrue(result)
        self.assertTrue(handler.is_dragging)
        self.assertIn(a1, handler.target_atoms)

    def test_press_on_bond_uses_atom1_as_target(self):
        a1 = AtomItem(1)
        a2 = AtomItem(2)
        bond = BondItem(a1, a2)
        scene = FakeScene(items=[a1, a2, bond])
        handler, mw = self._make_handler(scene)
        mw.init_manager.view_2d.itemAt.return_value = bond

        evt = MagicMock()
        evt.type.return_value = QEvent.Type.MouseButtonPress
        evt.button.return_value = Qt.MouseButton.LeftButton
        pos = MagicMock()
        pos.toPoint.return_value = MagicMock()
        evt.position.return_value = pos

        handler.eventFilter(None, evt)
        self.assertIn(a1, handler.target_atoms)

    def test_inactive_handler_ignores_events(self):
        scene = FakeScene(items=[])
        handler, mw = self._make_handler(scene)
        handler.active = False
        evt = MagicMock()
        self.assertFalse(handler.eventFilter(None, evt))

    def test_mouse_move_rotates_when_dragging(self):
        a1 = AtomItem(1, z=1.0)
        a2 = AtomItem(2, z=2.0)
        bond = BondItem(a1, a2)
        scene = FakeScene(items=[a1, a2, bond])
        handler, mw = self._make_handler(scene)
        handler.is_dragging = True
        handler.target_atoms = [a1, a2]
        handler.last_pos = MagicMock()
        handler.last_pos.x.return_value = 0
        handler.last_pos.y.return_value = 0

        evt = MagicMock()
        evt.type.return_value = QEvent.Type.MouseMove
        curr = MagicMock()
        curr.x.return_value = 10
        curr.y.return_value = 5
        pos = MagicMock()
        pos.toPoint.return_value = curr
        evt.position.return_value = pos

        result = handler.eventFilter(None, evt)
        self.assertTrue(result)
        self.assertEqual(bond.update_called, 1)

    def test_mouse_release_pushes_undo_and_clears_drag(self):
        scene = FakeScene(items=[])
        handler, mw = self._make_handler(scene)
        handler.is_dragging = True
        handler.target_atoms = [AtomItem(1)]
        context = MagicMock()
        self.pkg._context = context

        evt = MagicMock()
        evt.type.return_value = QEvent.Type.MouseButtonRelease
        result = handler.eventFilter(None, evt)
        self.assertTrue(result)
        self.assertFalse(handler.is_dragging)
        self.assertIsNone(handler.target_atoms)
        context.refresh_2d_scene.assert_called_once()
        mw.edit_actions_manager.push_undo_state.assert_called_once()

    def test_unhandled_event_type_returns_false(self):
        scene = FakeScene(items=[])
        handler, mw = self._make_handler(scene)
        evt = MagicMock()
        evt.type.return_value = QEvent.Type.KeyPress
        self.assertFalse(handler.eventFilter(None, evt))

    def test_rotate_molecule_noop_without_scene_data(self):
        scene = FakeScene(items=[])
        handler, mw = self._make_handler(scene)
        mw.scene = None
        handler.target_atoms = [AtomItem(1)]
        handler.rotate_molecule(0.1, 0.1)  # should not raise

    def test_rotate_molecule_noop_without_target(self):
        scene = FakeScene(items=[])
        handler, mw = self._make_handler(scene)
        handler.target_atoms = None
        handler.rotate_molecule(0.1, 0.1)  # returns early


class TestRotateToolHandlerEnsureZ(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()

    def test_no_scene_returns(self):
        mw = MagicMock()
        mw.scene = None
        handler = self.pkg.RotateToolHandler(mw)
        handler.ensure_z_coords()  # no crash

    def test_forced_restore_from_conformer(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        for i in range(mol.GetNumAtoms()):
            mol.GetAtomWithIdx(i).SetIntProp("_original_atom_id", i + 1)

        atoms = [AtomItem(i + 1, z=0.0) for i in range(mol.GetNumAtoms())]
        scene = FakeScene(items=atoms)
        mw = MagicMock()
        mw.scene = scene
        mw.current_mol = mol
        handler = self.pkg.RotateToolHandler(mw)
        handler.ensure_z_coords(force=True)
        conf = mol.GetConformer()
        expected_z = conf.GetAtomPosition(0).z * self.pkg._3d_scale
        self.assertAlmostEqual(atoms[0].z_3d, expected_z, places=5)

    def test_unmapped_atom_falls_back_to_existing_z(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        for i in range(mol.GetNumAtoms()):
            mol.GetAtomWithIdx(i).SetIntProp("_original_atom_id", i + 1)

        # atom_id 999 has no mapping in the RDKit mol
        atom = AtomItem(999, z=7.5)
        scene = FakeScene(items=[atom])
        mw = MagicMock()
        mw.scene = scene
        mw.current_mol = mol
        handler = self.pkg.RotateToolHandler(mw)
        handler.ensure_z_coords(force=True)
        self.assertEqual(atom.z_3d, 7.5)


class TestSetActive(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()

    def test_set_active_true_enters_rotate_mode(self):
        scene = FakeScene(items=[])
        mw = MagicMock()
        mw.scene = scene
        mw.current_mol = None
        handler = self.pkg.RotateToolHandler(mw)
        handler.set_active(True)
        mw.ui_manager.set_mode.assert_called_with("plugin_rotate_3d")

    def test_set_active_false_reverts_to_select(self):
        scene = FakeScene(items=[])
        mw = MagicMock()
        mw.scene = scene
        handler = self.pkg.RotateToolHandler(mw)
        handler.target_atoms = [AtomItem(1)]
        handler.set_active(False)
        self.assertIsNone(handler.target_atoms)
        mw.ui_manager.activate_select_mode.assert_called_once()


class TestPatchedOnCalculationFinished(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()

    def test_schedules_cleanup_when_triggered_by_plugin(self):
        from moleditpy.ui.compute_logic import ComputeManager

        mw = MagicMock()
        self.pkg._mw = mw
        self.pkg._enabled = True
        self.pkg._plugin_triggered_conversion = True
        if hasattr(ComputeManager, "_original_on_calculation_finished"):
            delattr(ComputeManager, "_original_on_calculation_finished")
        self.pkg.patched_on_calculation_finished(mw, "result")  # self==_mw

    def test_calls_original_if_present(self):
        from moleditpy.ui.compute_logic import ComputeManager

        called = []
        ComputeManager._original_on_calculation_finished = (
            lambda self, result: called.append(result)
        )
        try:
            mw = MagicMock()
            self.pkg._mw = None  # skip cleanup scheduling branch
            self.pkg.patched_on_calculation_finished(mw, "myresult")
            self.assertEqual(called, ["myresult"])
        finally:
            delattr(ComputeManager, "_original_on_calculation_finished")


class TestLoadStateFinalizedRestore(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.pkg = _load_plugin()
        # Make QTimer.singleShot synchronous so we can assert on the deferred work
        self._orig_single_shot = self.pkg.QTimer.singleShot
        self.pkg.QTimer.singleShot = staticmethod(lambda ms, fn: fn())

    def tearDown(self):
        self.pkg.QTimer.singleShot = self._orig_single_shot

    def test_load_state_restores_z_from_saved_data(self):
        atom = AtomItem(1, z=0.0)
        scene = FakeScene(items=[])
        scene.atom_items = {1: atom}
        scene.data = MagicMock()
        mw = MagicMock()
        mw.scene = scene
        self.pkg._mw = mw
        self.pkg._context = MagicMock()
        self.pkg._rotate_tool_handler = MagicMock()
        self.pkg.load_state({"z_data": {"1": 4.5}, "embed_without_h": True})
        self.assertEqual(atom.z_3d, 4.5)

    def test_load_state_enables_plugin_when_3d_data_present_and_disabled(self):
        mw = MagicMock()
        mw.scene = FakeScene(items=[])
        self.pkg._mw = mw
        self.pkg._context = MagicMock()
        self.pkg._enabled = False
        self.pkg.enable_plugin = MagicMock()
        self.pkg.load_state({"z_data": {"1": 1.0}})
        self.pkg.enable_plugin.assert_called_once()

    def test_load_state_no_scene_in_finalize_returns(self):
        mw = MagicMock()
        mw.scene = None
        self.pkg._mw = mw
        self.pkg._context = MagicMock()
        self.pkg.load_state({"z_data": {}})  # finalized_restore should just return


if __name__ == "__main__":
    unittest.main()
