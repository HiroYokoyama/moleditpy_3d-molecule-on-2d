"""Real host scene/view smoke scenario, run in an isolated interpreter."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root.parent / "python_molecular_editor/moleditpy/src"))
from moleditpy.core.molecular_data import MolecularData
from moleditpy.ui.atom_item import AtomItem
from moleditpy.ui.bond_item import BondItem
from moleditpy.ui.molecule_scene import MoleculeScene
from moleditpy.ui.zoomable_view import ZoomableView

app = QApplication([])
window = QMainWindow()
window.init_manager = SimpleNamespace(settings={})
window.ui_manager = SimpleNamespace(is_2d_editable=True)
window.edit_actions_manager = SimpleNamespace(push_undo_state=Mock(), undo=Mock(), redo=Mock())
window.edit_3d_manager = SimpleNamespace(update_2d_measurement_labels=Mock())
window.state_manager = SimpleNamespace(data=MolecularData())
window.current_mol = None
window.scene = scene = MoleculeScene(window.state_manager.data, window)
window.init_manager.scene = scene
view = ZoomableView(scene, window)
window.init_manager.view_2d = view
window.setCentralWidget(view)
window.resize(600, 400)
scene.setSceneRect(-250, -150, 500, 300)
window.show()
app.processEvents()

spec = importlib.util.spec_from_file_location("plugin", root / "3d_molecule_on_2d.py")
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)
plugin._mw = window
context = SimpleNamespace(refresh_2d_scene=scene.update_all_items)
plugin._context = context
original_item_change = BondItem.itemChange

try:
    # Exercise enabling on an empty editor and on an existing molecule.
    for cycle in range(3):
        plugin.enable_plugin(window, context)
        if "--check-native" in sys.argv:
            assert BondItem.itemChange is original_item_change
        scene.mode = "atom_C"
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=view.mapFromScene(QPointF(-70, 0)))
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=view.mapFromScene(QPointF(70, 0)))
        assert len(scene.atom_items) >= 2
        atoms = list(scene.atom_items.values())
        scene.create_bond(atoms[0], atoms[1])
        bond = next(iter(scene.bond_items.values()))
        if "--check-native" in sys.argv:
            assert not bond.flags() & bond.GraphicsItemFlag.ItemIsMovable
        scene.mode = "select"
        for _ in range(15):
            for point in (QPointF(-70, 0), QPointF(0, 0), QPointF(160, 80)):
                QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=view.mapFromScene(point))
            app.processEvents()
        plugin.disable_plugin(window)
        if "--check-native" in sys.argv:
            assert BondItem.itemChange is original_item_change
    assert AtomItem.paint is not plugin.patched_atom_paint
finally:
    plugin.disable_plugin(window)
    window.close()
    app.processEvents()
print("native click scenario passed", flush=True)
