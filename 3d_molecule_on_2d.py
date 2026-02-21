import os
import time
import json
import numpy as np
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D
except ImportError:
    pass
try:
    from PyQt6 import sip
except ImportError:
    import sip
from PyQt6.QtWidgets import QMenu, QToolBar, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGraphicsScene, QGraphicsItem, QCheckBox, QApplication
from PyQt6.QtCore import Qt, QPointF, QEvent, QObject, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QColor, QPen, QIcon, QAction, QActionGroup, QPainter, QBrush

# Metadata
PLUGIN_NAME = "3D Molecule on 2D"
PLUGIN_VERSION = "1.3.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = "Integrated 3D depth cues, rotation, and 3D-aware Mol export."

# Global state
_enabled = True
_show_depth_cues = True
_rotate_tool_handler = None
_depth_cue_strength = 0.8  # 0.0 to 1.0
_3d_scale = 50.0           # Strictly matches 1.0 / ANGSTROM_PER_PIXEL (1.0 / 0.02 = 50)
_embed_without_h = False   # New option
_active_worker = None
_mw = None
_context = None
_settings_file = os.path.splitext(os.path.abspath(__file__))[0] + ".json"

# Store original paint methods
_original_atom_paint = None
_original_bond_paint = None
_original_save_as_mol = None
_toolbar_actions_objs = []
_export_in_progress = False
_plugin_triggered_conversion = False
_last_cleanup_trigger_time = 0
_is_syncing = False


def blend_with_white(color, factor):
    # Linearly interpolate color towards white.
    r = int(color.red() + (255 - color.red()) * factor)
    g = int(color.green() + (255 - color.green()) * factor)
    b = int(color.blue() + (255 - color.blue()) * factor)
    return QColor(r, g, b)

def get_original_id(atom):
    """Chain of responsibility style property lookup for atom IDs."""
    if not atom: return None
    # Prioritize official property, fallback to legacy
    for prop_name in ["_original_atom_id", "original_id"]:
        if atom.HasProp(prop_name):
            try:
                # Handle both int and string properties safely
                val = atom.GetProp(prop_name)
                return int(val)
            except (ValueError, TypeError):
                continue
    return None

class PluginSettingsDialog(QDialog):
    def __init__(self, parent=None, current_enabled=True):
        super().__init__(parent)
        self.setWindowTitle(f"{PLUGIN_NAME} Settings")
        self.enabled = current_enabled
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.chk_enable = QCheckBox("Enable Plugin")
        self.chk_enable.setChecked(self.enabled)
        layout.addWidget(self.chk_enable)

        global _embed_without_h
        self.chk_embed_no_h = QCheckBox("Embed without Hydrogens (Cleaner 3D)")
        self.chk_embed_no_h.setChecked(_embed_without_h)
        self.chk_embed_no_h.setToolTip("Embed heavy atoms first, then add hydrogens. Often results in cleaner/more symmetric structures.")
        layout.addWidget(self.chk_embed_no_h)

        from PyQt6.QtWidgets import QDialogButtonBox
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        self.setLayout(layout)

    def accept(self):
        global _embed_without_h
        self.enabled = self.chk_enable.isChecked()
        _embed_without_h = self.chk_embed_no_h.isChecked()
        save_settings()
        super().accept()

def load_settings():
    global _enabled, _depth_cue_strength, _3d_scale, _embed_without_h
    try:
        if os.path.exists(_settings_file):
            import json
            with open(_settings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                _enabled = data.get('enabled', True)
                _depth_cue_strength = data.get('depth_cue_strength', 0.8)
                _3d_scale = data.get('3d_scale', 50.0)
                _embed_without_h = data.get('embed_without_h', False)
    except Exception as e:
        print(f"[{PLUGIN_NAME}] Error loading settings: {e}")

def save_settings():
    try:
        import json
        data = {
            'enabled': _enabled,
            'depth_cue_strength': _depth_cue_strength,
            '3d_scale': _3d_scale,
            'embed_without_h': _embed_without_h
        }
        with open(_settings_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[{PLUGIN_NAME}] Error saving settings: {e}")

def on_cleanup_triggered(*args, allow_trigger=True, **kwargs):
    """
    Smart 3D Trigger with Recursion Guard.
    """
    global _mw, _last_cleanup_trigger_time, _is_syncing
    mw = _mw
    if not mw or not mw.scene: return
    if _is_syncing: return
    
    _is_syncing = True
    try:
        # 0. Detect molecules in the scene
        molecules, all_atoms, all_bonds = find_molecules(mw.scene)
        if not molecules:
            mw.statusBar().showMessage("No molecules in scene.")
            return

        # 1. Determine if we NEED a 3D refresh
        mol = mw.current_mol
        needs_3d_refresh = False
        status_msg = ""
        
        if not mol or mol.GetNumConformers() == 0:
            needs_3d_refresh = True
            status_msg = "No 3D conformer found."
        else:
            # Check 1: Is the conformer actually 3D (non-flat)?
            conf = mol.GetConformer()
            zs = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
            if not zs or (max(zs) - min(zs)) < 1e-3:
                needs_3d_refresh = True
                status_msg = "Scene is 2D (flat)."
            else:
                # Check 2: Are all heavy atoms mapped?
                id_to_idx = {get_original_id(mol.GetAtomWithIdx(i)): i 
                             for i in range(mol.GetNumAtoms()) 
                             if get_original_id(mol.GetAtomWithIdx(i)) is not None}
                
                for atom_item in all_atoms:
                    if getattr(atom_item, "symbol", "") == "H": continue
                    aid = getattr(atom_item, "atom_id", None)
                    if aid is None or aid not in id_to_idx:
                        needs_3d_refresh = True
                        status_msg = f"Heavy atom mapping missing."
                        break
        
        # 2. Action Logic
        if needs_3d_refresh and allow_trigger:
            # Loop protection: No more than one trigger per 5 seconds
            now = time.time()
            if now - _last_cleanup_trigger_time < 5.0:
                mw.statusBar().showMessage("Smart 3D: Cooldown active. Skipping re-trigger.")
                return
            
            _last_cleanup_trigger_time = now
            
            # Branch logic: use local threaded embedding only if "without hydrogen" is specified.
            # Otherwise, use the main application's standard conversion logic "as it did".
            global _embed_without_h
            if _embed_without_h:
                mw.statusBar().showMessage(f"Smart 3D: {status_msg} Local Embedding starting (threaded)...")
                start_local_embedding(mw, _embed_without_h)
            else:
                mw.statusBar().showMessage(f"Smart 3D: {status_msg} Triggering Main Conversion...")
                if hasattr(mw, "trigger_conversion"):
                    global _plugin_triggered_conversion
                    _plugin_triggered_conversion = True
                    mw.trigger_conversion()
            return # Wait for conversion to finish
        
        # 3. Perform Sync
        if mol and mol.GetNumConformers() > 0:
            if needs_3d_refresh:
                mw.statusBar().showMessage("Smart 3D: Syncing whatever available...")
            else:
                mw.statusBar().showMessage("Smart 3D: Syncing layout to 3D...")
            sync_to_3d_layout(mw, mol)
        else:
            mw.statusBar().showMessage("Smart 3D: Conversion needed for synchronization.")
    finally:
        _is_syncing = False
    
    mw.scene.update()

    global _rotate_tool_handler
    if _rotate_tool_handler and hasattr(_rotate_tool_handler, "rotate_act") and _rotate_tool_handler.rotate_act:
        _rotate_tool_handler.rotate_act.setChecked(True)

def show_settings_dialog(*args, **kwargs):
    global _depth_cue_strength, _3d_scale, _mw
    mw = _mw
    if not mw: return
    dlg = QDialog(mw)
    dlg.setWindowTitle("Depth Cue Settings")
    layout = QVBoxLayout(dlg)
    depth_layout = QHBoxLayout()
    depth_label = QLabel("Cue Strength:")
    depth_value_label = QLabel(f"{int(_depth_cue_strength * 100)}%")
    depth_slider = QSlider(Qt.Orientation.Horizontal)
    depth_slider.setRange(0, 100)
    depth_slider.setValue(int(_depth_cue_strength * 100))
    def on_val_changed(v):
        global _depth_cue_strength
        _depth_cue_strength = v / 100.0
        depth_value_label.setText(f"{v}%")
        if mw.scene: mw.scene.update()
    depth_slider.valueChanged.connect(on_val_changed)
    depth_layout.addWidget(depth_label); depth_layout.addWidget(depth_slider); depth_layout.addWidget(depth_value_label)
    layout.addLayout(depth_layout)
    
    global _embed_without_h
    h_layout = QHBoxLayout()
    h_checkbox = QCheckBox("Embed without Hydrogens")
    h_checkbox.setChecked(_embed_without_h)
    def on_h_toggled(checked):
        global _embed_without_h
        _embed_without_h = checked
        save_settings()
    h_checkbox.toggled.connect(on_h_toggled)
    h_layout.addWidget(h_checkbox)
    layout.addLayout(h_layout)

    # 3D Scale UI
    scale_layout = QHBoxLayout()
    scale_label = QLabel("3D Scale:")
    scale_value_label = QLabel(f"{_3d_scale:.1f}")
    scale_slider = QSlider(Qt.Orientation.Horizontal)
    scale_slider.setRange(10, 100)
    scale_slider.setValue(int(_3d_scale))
    def on_scale_changed(v):
        global _3d_scale
        old_scale = _3d_scale
        _3d_scale = float(v)
    # No more manual scaling to prevent Z-compression and unit mismatches

    ok_btn = QPushButton("OK"); ok_btn.clicked.connect(dlg.accept); layout.addWidget(ok_btn)
    dlg.exec()

def open_settings_dialog(mw, context):
    global _enabled
    dlg = PluginSettingsDialog(mw, _enabled)
    if dlg.exec():
        if _enabled != dlg.enabled:
            _enabled = dlg.enabled
            save_settings()
            if _enabled:
                enable_plugin(mw, context)
            else:
                disable_plugin(mw)
            status = "Enabled" if _enabled else "Disabled"
            mw.statusBar().showMessage(f"{PLUGIN_NAME}: {status}")

def initialize(context):
    global _mw, _settings_file, _enabled, _context
    mw = context.get_main_window()
    _mw = mw
    _context = context
    
    # Path to the settings file within the plugin directory
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    _settings_file = os.path.join(plugin_dir, "3d_molecule_on_2d.json")
    
    load_settings()
    
    # Register Setting Menu
    context.add_menu_action("Settings/3D Molecule on 2D...", 
                          lambda: open_settings_dialog(mw, context))

    # Register Toolbar Actions (Standard registration)
    context.add_toolbar_action(on_cleanup_triggered, "Clean Up 3D", tooltip="Sync 2D layout to 3D")
    context.add_toolbar_action(lambda: None, "Rotate 3D", tooltip="Rotate molecule in 3D")
    context.add_toolbar_action(show_settings_dialog, "3D on 2D Settings...", tooltip="Fine-tune visuals")
    # Register Save/Load handlers for project file persistence
    context.register_save_handler(save_state)
    context.register_load_handler(load_state)

    def startup_fix():
        # First capture the actions that were just registered
        configure_actions()
        if _enabled:
            # enable_plugin will activate patches and configure actions properly
            enable_plugin(mw, context)
        else:
            # disable_plugin will hide the actions we just found
            disable_plugin(mw)

    # Avoid immediate UI interference, let the app finish startup
    QTimer.singleShot(0, startup_fix)

def enable_plugin(mw, context):
    global _rotate_tool_handler
    if not _rotate_tool_handler:
        _rotate_tool_handler = RotateToolHandler(mw)
    
    toggle_monkey_patches(True, mw)
    patch_export_logic(True)
    
    tb = getattr(mw, 'plugin_toolbar', None)
    if not tb: return
    
    configure_actions()
    
    # Ensure actions are in the toolbar (they might have been removed by disable_plugin)
    if _toolbar_actions_objs:
        tb.show()
        for act in _toolbar_actions_objs:
            if act not in tb.actions():
                tb.addAction(act)
    
    if mw.scene:
        for item in mw.scene.items():
            # AtomItemとBondItemの両方に対し、Movableフラグを確実に立てる
            if type(item).__name__ in ["AtomItem", "BondItem"]:
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        mw.scene.update()

def disable_plugin(mw):
    toggle_monkey_patches(False, mw)
    patch_export_logic(False)
    if _rotate_tool_handler:
        _rotate_tool_handler.set_active(False)
    
    # Remove from toolbar
    tb = getattr(mw, 'plugin_toolbar', None)
    if tb:
        for act in _toolbar_actions_objs:
            tb.removeAction(act)
        # If toolbar is now essentially empty (only our actions were removed), hide it
        # Note: tb.actions() might still have separators or other plugins' actions
        if not any(not a.isSeparator() for a in tb.actions()):
            tb.hide()
    
    if mw.scene:
        mw.scene.update()

def configure_actions():
    global _mw, _toolbar_actions_objs
    mw = _mw
    tb = getattr(mw, 'plugin_toolbar', None)
    if not tb: return
    
    # Filter for our actions in the toolbar
    found_actions = [a for a in tb.actions() if a.text() in ["Clean Up 3D", "Rotate 3D", "3D on 2D Settings..."]]
    
    # Only update our global list if we found them (to avoid clearing it when plugin is disabled)
    if found_actions:
        _toolbar_actions_objs = found_actions
    
    actions = _toolbar_actions_objs
    if not actions: return

    rotate_act = next((a for a in actions if a.text() == "Rotate 3D"), None)
    cleanup_act = next((a for a in actions if a.text() == "Clean Up 3D"), None)
    settings_act = next((a for a in actions if a.text() == "3D on 2D Settings..."), None)
    
    # Set Checkable and connect toggled
    if rotate_act:
        rotate_act.setCheckable(True)
        try: rotate_act.toggled.disconnect()
        except: pass
        rotate_act.toggled.connect(on_rotate_toggled)
        if _rotate_tool_handler:
            _rotate_tool_handler.rotate_act = rotate_act
    
    # Mode Integration: Add Rotate 3D to main tool group for exclusivity
    if rotate_act and hasattr(mw, "tool_group"):
        if rotate_act not in mw.tool_group.actions():
            mw.tool_group.addAction(rotate_act)
    
    # Visual Grouping (Separators)
    if rotate_act and cleanup_act and not any(a.isSeparator() for a in tb.actions()):
         # This is a bit naive but tries to add separators
         pass

    tb.show()

def on_rotate_toggled(checked):
    global _mw, _rotate_tool_handler
    mw = _mw
    if checked:
        # Ensure other tools in the group are unchecked
        if hasattr(mw, "tool_group"):
            for act in mw.tool_group.actions():
                if act.text() != "Rotate 3D" and act.isChecked():
                    act.setChecked(False)
        if _rotate_tool_handler:
            _rotate_tool_handler.set_active(True)
    else:
        if _rotate_tool_handler:
            _rotate_tool_handler.set_active(False)
        # Default back to select mode if we are turning off rotation
        if hasattr(mw, "mode_actions") and "select" in mw.mode_actions:
            mw.mode_actions["select"].setChecked(True)
            mw.set_mode("select")
    
    # --- Undo/Redo Sync ---
    def on_undo_redo_changed():
        global _mw
        mw = _mw
        # Delay slightly to allow the scene to update its items
        def run_restore():
            if _rotate_tool_handler:
                _rotate_tool_handler.ensure_z_coords(force=True)
            if mw.scene:
                _, _, all_bonds = find_molecules(mw.scene)
                for bond in all_bonds:
                    if hasattr(bond, "update_position"):
                        bond.update_position()
                    if hasattr(bond.atom1, "z_3d") and hasattr(bond.atom2, "z_3d"):
                        bond.setZValue((bond.atom1.z_3d + bond.atom2.z_3d) / 2.0)
                update_molecule_z_ranges(mw.scene)
                mw.scene.update()
        
        QTimer.singleShot(100, run_restore)

    try:
        # Check if undo_stack is a list or a QUndoStack
        stack = getattr(mw, "undo_stack", getattr(mw, "undoStack", None))
        if hasattr(stack, "indexChanged"):
            stack.indexChanged.connect(lambda idx: on_undo_redo_changed())
        else:
            # Fallback to monkeypatching if it's a list or doesn't have indexChanged
            orig_undo = getattr(mw, "undo", None)
            if orig_undo and callable(orig_undo) and not hasattr(orig_undo, "_is_patched"):
                def patched_undo(*args, **kwargs):
                    res = orig_undo(*args, **kwargs)
                    on_undo_redo_changed()
                    return res
                patched_undo._is_patched = True
                mw.undo = patched_undo
            
            orig_redo = getattr(mw, "redo", None)
            if orig_redo and callable(orig_redo) and not hasattr(orig_redo, "_is_patched"):
                def patched_redo(*args, **kwargs):
                    res = orig_redo(*args, **kwargs)
                    on_undo_redo_changed()
                    return res
                patched_redo._is_patched = True
                mw.redo = patched_redo
    except Exception as e:
        print(f"[{PLUGIN_NAME}] Warning: Could not hook into undo/redo system: {e}")

# --- Persistence ---

def save_state():
    global _mw, _depth_cue_strength, _3d_scale
    mw = _mw
    if not mw: return {}
    state = {
        "depth_cue_strength": _depth_cue_strength,
        "3d_scale": _3d_scale,
        "embed_without_h": _embed_without_h
    }
    if mw.scene and hasattr(mw.scene, 'data'):
        z_data = {}
        for aid, atom_data in mw.scene.data.atoms.items():
            item = atom_data.get('item')
            if item and not sip_isdeleted_safe(item) and hasattr(item, "z_3d"):
                # Save raw pixels for Z to match X/Y persistence in the core app.
                # This ensures that rotations and positions are saved 1:1 without scaling artifacts.
                z_data[str(aid)] = item.z_3d
        state["z_data"] = z_data
    return state

def load_state(data):
    global _mw, _depth_cue_strength, _3d_scale, _enabled, _context
    mw = _mw
    if not mw or not data: return

    _depth_cue_strength = data.get("depth_cue_strength", 0.8)
    _3d_scale = data.get("3d_scale", 50.0)
    _embed_without_h = data.get("embed_without_h", False)

    # If the file has 3D data, we temporarily enable the plugin
    has_3d_data = "z_data" in data and len(data["z_data"]) > 0
    if has_3d_data and not _enabled:
        QTimer.singleShot(0, lambda: enable_plugin(mw, _context))

    z_data_to_restore = data.get("z_data", {})

    # Ensure Z coordinates are present (either from z_data or the RDKit molecule)
    def finalized_restore():
        if not mw.scene: return
        
        # Phase 1: Restore Z from project data if available
        if z_data_to_restore:
            for aid_str, z in z_data_to_restore.items():
                try:
                    aid = int(aid_str)
                    atom_data = None
                    if hasattr(mw.scene, 'data') and mw.scene.data:
                        atom_data = mw.scene.data.atoms.get(aid)
                    
                    if atom_data and 'item' in atom_data:
                        item = atom_data['item']
                        if item and not sip_isdeleted_safe(item):
                            item.z_3d = z
                            item.setZValue(z)
                except Exception: continue
        
        # Phase 2: If no Z was restored, fallback to RDKit (but only if rotated state is also reset)
        if _rotate_tool_handler:
            _rotate_tool_handler.ensure_z_coords()
            
        # Phase 3: Refresh all bonds and Z-order
        _, _, all_bonds = find_molecules(mw.scene)
        for bond in all_bonds:
            #bond.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            if hasattr(bond, "update_position"): bond.update_position()
            if hasattr(bond.atom1, "z_3d") and hasattr(bond.atom2, "z_3d"):
                bond.setZValue((bond.atom1.z_3d + bond.atom2.z_3d) / 2.0)
        
        update_molecule_z_ranges(mw.scene)
        mw.scene.update()

    QTimer.singleShot(200, finalized_restore)

def patched_on_calculation_finished(self, result):
    """Hook to trigger plugin sync after 3D conversion finishes."""
    from moleditpy.modules.main_window_compute import MainWindowCompute
    if hasattr(MainWindowCompute, "_original_on_calculation_finished"):
        MainWindowCompute._original_on_calculation_finished(self, result)
    
    global _enabled, _mw, _plugin_triggered_conversion
    if _enabled and _mw == self and _plugin_triggered_conversion:
        # Reset flag IMMEDIATELY to prevent loop re-entry
        _plugin_triggered_conversion = False
        # Defer slightly for core app UI cleanup
        QTimer.singleShot(700, lambda: on_cleanup_triggered(allow_trigger=False))

def toggle_monkey_patches(active, mw=None):
    from moleditpy.modules.atom_item import AtomItem
    from moleditpy.modules.bond_item import BondItem
    global _original_atom_paint, _original_bond_paint, _show_depth_cues
    
    _show_depth_cues = active
    if active:
        if _original_atom_paint is None:
            _original_atom_paint = AtomItem.paint
            AtomItem.paint = patched_atom_paint
        if _original_bond_paint is None:
            from moleditpy.modules.bond_item import BondItem
            _original_bond_paint = BondItem.paint
            BondItem.paint = patched_bond_paint
            
        # Patch MainWindowCompute to detect conversion finish
        from moleditpy.modules.main_window_compute import MainWindowCompute
        if not hasattr(MainWindowCompute, "_original_on_calculation_finished"):
            MainWindowCompute._original_on_calculation_finished = MainWindowCompute.on_calculation_finished
            MainWindowCompute.on_calculation_finished = patched_on_calculation_finished
 
            # Veto ItemPositionChange if it doesn't come from programmatic setPos
            if not hasattr(BondItem, "_original_itemChange"):
                BondItem._original_itemChange = BondItem.itemChange
                def bond_item_change_guarded(self, change, value):
                    return BondItem._original_itemChange(self, change, value)
                BondItem.itemChange = bond_item_change_guarded

    else:
        from moleditpy.modules.main_window_compute import MainWindowCompute
        if hasattr(MainWindowCompute, "_original_on_calculation_finished"):
            MainWindowCompute.on_calculation_finished = MainWindowCompute._original_on_calculation_finished
            delattr(MainWindowCompute, "_original_on_calculation_finished")

        if _original_atom_paint is not None:
            AtomItem.paint = _original_atom_paint
            _original_atom_paint = None
        if _original_bond_paint is not None:
            BondItem.paint = _original_bond_paint
            _original_bond_paint = None

def patched_to_rdkit_mol(self, use_2d_stereo=True):
    mol = self._original_to_rdkit_mol(use_2d_stereo)
    # Only apply Z/XYZ overrides if an export is explicitly in progress
    if _export_in_progress and mol and mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        conf.Set3D(True)
        # Use the plugin's _3d_scale to accurately convert back to angstroms
        scale = _3d_scale if abs(_3d_scale) > 1e-4 else 50.0
        for i in range(mol.GetNumAtoms()):
            aid = get_original_id(mol.GetAtomWithIdx(i))
            if aid is not None and aid in self.atoms:
                item = self.atoms[aid].get("item")
                if item and not sip_isdeleted_safe(item) and hasattr(item, "z_3d"):
                    # Use current visual position (including rotation) for all coordinates
                    pos_item = item.pos()
                    ax = pos_item.x() / scale
                    ay = -pos_item.y() / scale
                    az = item.z_3d / scale
                    conf.SetAtomPosition(i, Point3D(ax, ay, az))
    return mol

def patched_save_as_mol(self, *args, **kwargs):
    global _export_in_progress
    from moleditpy.modules.main_window_molecular_parsers import MainWindowMolecularParsers
    _export_in_progress = True
    try:
        if hasattr(MainWindowMolecularParsers, "_original_save_as_mol") and MainWindowMolecularParsers._original_save_as_mol:
             return MainWindowMolecularParsers._original_save_as_mol(self, *args, **kwargs)
        return self.save_as_mol(*args, **kwargs) # Fallback (should not happen if patched)
    finally:
        _export_in_progress = False

def patch_export_logic(active=True):
    """Monkey patch MolecularData and Parsers for 3D-aware Mol export."""
    from moleditpy.modules.molecular_data import MolecularData
    from moleditpy.modules.main_window_molecular_parsers import MainWindowMolecularParsers
    
    if active:
        # Patch to_rdkit_mol (data layer)
        if not hasattr(MolecularData, "_original_to_rdkit_mol"):
            MolecularData._original_to_rdkit_mol = MolecularData.to_rdkit_mol
            MolecularData.to_rdkit_mol = patched_to_rdkit_mol
            
        # Patch save_as_mol (UI/Export layer) to trigger the flag
        if not hasattr(MainWindowMolecularParsers, "_original_save_as_mol"):
            MainWindowMolecularParsers._original_save_as_mol = MainWindowMolecularParsers.save_as_mol
            MainWindowMolecularParsers.save_as_mol = patched_save_as_mol
    else:
        # Restore to_rdkit_mol
        if hasattr(MolecularData, "_original_to_rdkit_mol"):
            MolecularData.to_rdkit_mol = MolecularData._original_to_rdkit_mol
            delattr(MolecularData, "_original_to_rdkit_mol")
            
        # Restore save_as_mol
        if hasattr(MainWindowMolecularParsers, "_original_save_as_mol"):
            MainWindowMolecularParsers.save_as_mol = MainWindowMolecularParsers._original_save_as_mol
            delattr(MainWindowMolecularParsers, "_original_save_as_mol")

# --- Patched Paint Methods ---

def patched_atom_paint(self, painter, option, widget):
    global _show_depth_cues, _depth_cue_strength, _original_atom_paint
    from moleditpy.modules import atom_item
    
    if _show_depth_cues and hasattr(self, "z_3d") and self.scene():
        # Use per-molecule range if available, else scene range
        z_min = getattr(self, "mol_z_min", None)
        z_max = getattr(self, "mol_z_max", None)
        
        if z_min is None or z_max is None:
            z_min, z_max = get_scene_z_range(self.scene())
            
        z_range = z_max - z_min
        if _show_depth_cues and z_range > 1e-4:
            depth_factor = (self.z_3d - z_min) / z_range
            white_factor = (1.0 - depth_factor) * _depth_cue_strength
            
            if white_factor > 0.05:
                # Shift colors in CPK_COLORS temporarily
                sym = self.symbol
                orig_color = atom_item.CPK_COLORS.get(sym, atom_item.CPK_COLORS["DEFAULT"])
                faded_color = blend_with_white(orig_color, white_factor)
                
                # Temporary Swap
                had_key = sym in atom_item.CPK_COLORS
                atom_item.CPK_COLORS[sym] = faded_color
                try:
                    _original_atom_paint(self, painter, option, widget)
                finally:
                    if had_key: atom_item.CPK_COLORS[sym] = orig_color
                    else: atom_item.CPK_COLORS.pop(sym, None)
                return

    _original_atom_paint(self, painter, option, widget)

def patched_bond_paint(self, painter, option, widget):
    global _show_depth_cues, _depth_cue_strength, _original_bond_paint
    if _original_bond_paint is None: return
    
    if _show_depth_cues and hasattr(self.atom1, "z_3d") and hasattr(self.atom2, "z_3d") and self.scene():
        z1, z2 = self.atom1.z_3d, self.atom2.z_3d
        avg_z = (z1 + z2) / 2.0
        
        # Bond's range is from its atoms
        z_min = getattr(self.atom1, "mol_z_min", None)
        z_max = getattr(self.atom1, "mol_z_max", None)

        if z_min is None or z_max is None:
            z_min, z_max = get_scene_z_range(self.scene())
            
        z_range = z_max - z_min
        if _show_depth_cues and z_range > 1e-4:
            depth_factor = (avg_z - z_min) / z_range
            white_factor = (1.0 - depth_factor) * _depth_cue_strength
            
            if white_factor > 0.05:
                # Temporary Swap Bond Color in Settings to influence internal drawing
                win = self.scene().views()[0].window() if self.scene() and self.scene().views() else None
                if win and hasattr(win, "settings"):
                    orig_hex = win.settings.get("bond_color_2d", "#222222")
                    faded_color = blend_with_white(QColor(orig_hex), white_factor)
                    win.settings["bond_color_2d"] = faded_color.name()
                    try:
                        _original_bond_paint(self, painter, option, widget)
                    finally:
                        win.settings["bond_color_2d"] = orig_hex
                    return

    _original_bond_paint(self, painter, option, widget)

def get_scene_z_range(scene):
    zs = []
    if hasattr(scene, 'data') and hasattr(scene.data, 'atoms'):
        for atom_data in scene.data.atoms.values():
            item = atom_data.get('item')
            if item and not sip_isdeleted_safe(item) and hasattr(item, "z_3d"):
                zs.append(item.z_3d)
    if not zs: return -5.0, 5.0
    return min(zs), max(zs)

def find_molecules(scene):
    """Find connected components of AtomItems in the scene."""
    all_atoms = []
    all_bonds = []
    if not scene: return [], [], []
    for item in scene.items():
        if sip_isdeleted_safe(item): continue
        cls_name = type(item).__name__
        if cls_name == "AtomItem":
            all_atoms.append(item)
        elif cls_name == "BondItem":
            all_bonds.append(item)
    
    if not all_atoms: return [], [], []

    adj = {atom: [] for atom in all_atoms}
    for bond in all_bonds:
        if bond.atom1 in adj and bond.atom2 in adj:
            adj[bond.atom1].append(bond.atom2)
            adj[bond.atom2].append(bond.atom1)

    visited = set()
    molecules = []
    for atom in all_atoms:
        if atom not in visited:
            mol_atoms = []
            stack = [atom]
            visited.add(atom)
            while stack:
                curr = stack.pop()
                mol_atoms.append(curr)
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            molecules.append(mol_atoms)
    return molecules, all_atoms, all_bonds

def update_molecule_z_ranges(scene):
    """Calculate and store Z-range for each molecule on its atoms."""
    try:
        molecules, _, _ = find_molecules(scene)
    except Exception:
        return
    for mol_atoms in molecules:
        zs = [getattr(a, "z_3d", 0.0) for a in mol_atoms]
        if not zs: continue
        z_min, z_max = min(zs), max(zs)
        for a in mol_atoms:
            a.mol_z_min = z_min
            a.mol_z_max = z_max

class LocalCalculationWorker(QObject):
    finished = pyqtSignal(object)
    status = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, mol_block, embed_without_h, atom_ids):
        super().__init__()
        self.mol_block = mol_block
        self.embed_without_h = embed_without_h
        self.atom_ids = atom_ids

    def run(self):
        try:
            self.status.emit("Calculating 3D structure...")
            
            # 1. Convert 2D data to RDKit mol
            mol = Chem.MolFromMolBlock(self.mol_block, removeHs=True)
            if not mol:
                self.error.emit("Failed to create molecule structure.")
                return

            # Ensure original IDs are preserved
            # Map by index as a fallback (MolFromMolBlock usually preserves order)
            for i, aid in enumerate(self.atom_ids):
                if i < mol.GetNumAtoms():
                    mol.GetAtomWithIdx(i).SetIntProp("_original_atom_id", aid)

            # Preserve explicit stereo information (E/Z)
            explicit_stereo = {}
            mol_lines = self.mol_block.split("\n")
            for line in mol_lines:
                if line.startswith("M  CFG"):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            bond_idx = int(parts[3]) - 1
                            cfg_value = int(parts[4])
                            if cfg_value == 1: explicit_stereo[bond_idx] = Chem.BondStereo.STEREOZ
                            elif cfg_value == 2: explicit_stereo[bond_idx] = Chem.BondStereo.STEREOE
                        except: continue

            def apply_stereo(m, stereo_info):
                for bidx, stype in stereo_info.items():
                    if bidx < m.GetNumBonds():
                        b = m.GetBondWithIdx(bidx)
                        if b.GetBondType() == Chem.BondType.DOUBLE:
                            begin = b.GetBeginAtom()
                            end = b.GetEndAtom()
                            bn = [n.GetIdx() for n in begin.GetNeighbors() if n.GetIdx() != end.GetIdx()]
                            en = [n.GetIdx() for n in end.GetNeighbors() if n.GetIdx() != begin.GetIdx()]
                            if bn and en:
                                b.SetStereoAtoms(bn[0], en[0])
                                b.SetStereo(stype)

            # 2. Embedding process
            params = AllChem.ETKDGv2()
            params.randomSeed = 42
            params.enforceChirality = True

            if self.embed_without_h:
                self.status.emit("Performing RDKit embedding (skeleton first)...")
                apply_stereo(mol, explicit_stereo)
                res = AllChem.EmbedMolecule(mol, params)
                if res == -1:
                    params.useRandomCoords = True
                    res = AllChem.EmbedMolecule(mol, params)
                
                if res != -1:
                    self.status.emit("Placing hydrogens on 3D skeleton...")
                    mol = Chem.AddHs(mol, addCoords=True)
            else:
                self.status.emit("Performing RDKit standard embedding...")
                mol = Chem.AddHs(mol)
                apply_stereo(mol, explicit_stereo)
                res = AllChem.EmbedMolecule(mol, params)
                if res == -1:
                    params.useRandomCoords = True
                    res = AllChem.EmbedMolecule(mol, params)

            if res == -1:
                self.error.emit("3D embedding failed.")
                return

            # 3. Final Cleanup: Remove hydrogens if requested BEFORE optimization
            if self.embed_without_h:
                self.status.emit("Removing auxiliary hydrogens...")
                mol = Chem.RemoveHs(mol)

            # 4. Optimization
            self.status.emit("Optimizing structure (MMFF94s)...")
            try:
                AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s")
            except:
                self.status.emit("Optimizing structure (UFF fallback)...")
                try: AllChem.UFFOptimizeMolecule(mol)
                except: pass

            self.status.emit("3D structure ready.")
            self.finished.emit(mol)
            
        except Exception as e:
            self.error.emit(str(e))

def start_local_embedding(mw, embed_without_h=False):
    """
    Start local embedding in a background thread.
    """
    global _active_worker
    if _active_worker:
        return

    mol_block = mw.data.to_mol_block()
    if not mol_block:
        return

    atom_ids = list(mw.data.atoms.keys())
    
    # Setup thread and worker
    thread = QThread()
    worker = LocalCalculationWorker(mol_block, embed_without_h, atom_ids)
    worker.moveToThread(thread)
    
    _active_worker = (thread, worker)
    
    # Connect signals
    def update_ui_status(msg):
        mw.statusBar().showMessage(msg)
        if hasattr(mw, "_calculating_text_actor"):
            try:
                # Update text in the 3D window if it exists
                mw._calculating_text_actor.SetInput(msg)
                mw.plotter.render()
            except: pass

    thread.started.connect(worker.run)
    worker.status.connect(update_ui_status)
    worker.error.connect(lambda msg: on_embedding_error(mw, msg))
    worker.finished.connect(lambda mol: on_embedding_finished(mw, mol))
    
    # Cleanup
    worker.finished.connect(thread.quit)
    worker.error.connect(thread.quit)
    thread.finished.connect(thread.deleteLater)
    worker.finished.connect(worker.deleteLater)
    worker.error.connect(worker.deleteLater)
    
    def clear_active():
        global _active_worker
        _active_worker = None
    thread.finished.connect(clear_active)
    
    # Show "Calculating..." overlay like the main app
    if hasattr(mw, "plotter") and mw.plotter:
        try:
            mw.plotter.clear()
            bg_color_hex = mw.settings.get("background_color", "#919191")
            from PyQt6.QtGui import QColor
            bg_qcolor = QColor(bg_color_hex)
            luminance = bg_qcolor.toHsl().lightness() if bg_qcolor.isValid() else 255
            text_color = "black" if luminance > 128 else "white"
            
            text_actor = mw.plotter.add_text(
                "Calculating...",
                position="lower_right",
                font_size=15,
                color=text_color,
                name="calculating_text"
            )
            mw._calculating_text_actor = text_actor
            mw.plotter.render()
        except: pass

    thread.start()

def on_embedding_error(mw, msg):
    # Remove overlay
    if hasattr(mw, "plotter") and mw.plotter and hasattr(mw, "_calculating_text_actor"):
        try:
            mw.plotter.remove_actor(mw._calculating_text_actor)
            mw.plotter.render()
        except: pass
    mw.statusBar().showMessage(f"Smart 3D Error: {msg}")

def on_embedding_finished(mw, mol):
    # Remove overlay
    if hasattr(mw, "plotter") and mw.plotter and hasattr(mw, "_calculating_text_actor"):
        try:
            mw.plotter.remove_actor(mw._calculating_text_actor)
            mw.plotter.render()
        except: pass
    
    mw.current_mol = mol
    # Update 3D viewer
    if hasattr(mw, "draw_molecule_3d"):
        mw.draw_molecule_3d(mol)
    
    # Perform Sync
    sync_to_3d_layout(mw, mol)
    mw.statusBar().showMessage("Smart 3D: Local Embedding and Sync completed.")
    mw.scene.update()

# --- Logic ---

def sync_to_3d_layout(mw, mol):
    if not mol or mol.GetNumConformers() == 0: return
    conf = mol.GetConformer()
    scale = _3d_scale
    
    # 1. Map original_id to 3D coordinates
    proj_coords = {}
    for i in range(mol.GetNumAtoms()):
        aid = get_original_id(mol.GetAtomWithIdx(i))
        if aid is not None:
            p = conf.GetAtomPosition(i)
            proj_coords[aid] = (p.x * scale, -p.y * scale, p.z * scale)

    # 2. Find molecules in scene
    molecules, all_atoms, all_bonds = find_molecules(mw.scene)
    
    # 3. Targeted Sync based on selection
    selected_items = mw.scene.selectedItems()
    target_mol_indices = []
    
    if selected_items:
        for i, mol_atoms in enumerate(molecules):
            if any(a.isSelected() for a in mol_atoms):
                target_mol_indices.append(i)
                continue
            for bond in all_bonds:
                if bond.isSelected() and (bond.atom1 in mol_atoms or bond.atom2 in mol_atoms):
                    target_mol_indices.append(i)
                    break
    
    # 4. Sync target molecules
    for i, mol_atoms in enumerate(molecules):
        if selected_items and i not in target_mol_indices:
            continue
            
        mapped_data = [] # List of (atom_item, px, py, pz)
        for a in mol_atoms:
            # ROBUST MAPPING: Use a.atom_id directly
            aid = getattr(a, "atom_id", None)
            if aid in proj_coords:
                mapped_data.append((a, *proj_coords[aid]))
        
        if not mapped_data: continue
        
        # Calculate 2D center
        cur_2d_x = sum(a.pos().x() for a, *proj in mapped_data) / len(mapped_data)
        cur_2d_y = sum(a.pos().y() for a, *proj in mapped_data) / len(mapped_data)
        
        # Calculate Projected center
        proj_x = sum(px for a, px, py, pz in mapped_data) / len(mapped_data)
        proj_y = sum(py for a, px, py, pz in mapped_data) / len(mapped_data)
        
        dx, dy = cur_2d_x - proj_x, cur_2d_y - proj_y
        
        for atom_item, px, py, pz in mapped_data:
            atom_item.setPos(QPointF(px + dx, py + dy))
            atom_item.z_3d = pz
            atom_item.setZValue(pz)

    # 5. Update ALL bonds (ensure position and ZValue) and Z-ranges
    for bond in all_bonds:
        if hasattr(bond, "update_position"):
            bond.update_position()
        if hasattr(bond.atom1, "z_3d") and hasattr(bond.atom2, "z_3d"):
            bond.setZValue((bond.atom1.z_3d + bond.atom2.z_3d) / 2.0)
            
    update_molecule_z_ranges(mw.scene)
    mw.scene.update()

class RotateToolHandler(QObject):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.active = False
        self.is_dragging = False
        self.rotate_act = None
        self.target_atoms = None
        self.mw.view_2d.viewport().installEventFilter(self)
        
    def set_active(self, state):
        self.active = state
        if state:
            self.mw.set_mode("plugin_rotate_3d")
            self.ensure_z_coords()
        else:
            self.target_atoms = None
            self.mw.scene.mode = "select"
            if hasattr(self.mw, "activate_select_mode"): self.mw.activate_select_mode()
        
        # Recalculate ranges for depth cues
        if self.mw.scene:
            update_molecule_z_ranges(self.mw.scene)

    def ensure_z_coords(self, force=False):
        if not self.mw.scene: return
        atoms_needing_z = []
        has_nonzero_z = False
        
        all_atoms = [i for i in self.mw.scene.items() if type(i).__name__ == "AtomItem" and not sip_isdeleted_safe(i)]
        for item in all_atoms:
            z = getattr(item, "z_3d", 0.0)
            if abs(z) > 1e-4: has_nonzero_z = True
            atoms_needing_z.append(item)
        
        mol = self.mw.current_mol
        
        # If forced or items are flat but RDKit has 3D data, restore them
        if (force or not has_nonzero_z) and mol and mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            scale = _3d_scale
            
            # Map items to RDKit indices using _original_atom_id
            # Create a lookup: original_id -> rdkit_index
            id_to_idx = {}
            for i in range(mol.GetNumAtoms()):
                aid = get_original_id(mol.GetAtomWithIdx(i))
                if aid is not None:
                    id_to_idx[aid] = i
            
            for item in all_atoms:
                aid = item.atom_id
                if aid in id_to_idx:
                    idx = id_to_idx[aid]
                    item.z_3d = conf.GetAtomPosition(idx).z * scale
                    item.setZValue(item.z_3d)
                else:
                    # Fallback for atoms without mapping
                    item.z_3d = getattr(item, "z_3d", 0.0)
                    item.setZValue(item.z_3d)
        
        # Refresh ranges regardless
        update_molecule_z_ranges(self.mw.scene)

    def eventFilter(self, obj, event):
        if not self.active: return False
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                # Check for item at click position
                item = self.mw.view_2d.itemAt(event.position().toPoint())
                
                self.is_dragging = True
                self.last_pos = event.position().toPoint()
                
                # Identify clicked molecule
                molecules, _, _ = find_molecules(self.mw.scene)
                self.target_atoms = None
                clicked_item = item
                
                # Handle clicking on atoms or bonds
                clicked_atom = None
                if type(clicked_item).__name__ == "AtomItem":
                    clicked_atom = clicked_item
                elif type(clicked_item).__name__ == "BondItem":
                    clicked_atom = clicked_item.atom1
                
                if clicked_atom:
                    for mol in molecules:
                        if clicked_atom in mol:
                            self.target_atoms = mol
                            break
                
                return True
        elif event.type() == QEvent.Type.MouseMove:
            if self.is_dragging:
                curr_pos = event.position().toPoint()
                # dx: horizontal mouse movement -> Ry (yaw)
                # dy: vertical mouse movement -> Rx (pitch)
                dx = (curr_pos.x() - self.last_pos.x()) * 0.01
                dy = -(curr_pos.y() - self.last_pos.y()) * 0.01
                self.rotate_molecule(dx, dy)
                self.last_pos = curr_pos
                return True
        elif event.type() == QEvent.Type.MouseButtonRelease:
            self.is_dragging = False
            self.target_atoms = None
            return True
        return False

    def rotate_molecule(self, dx, dy):
        if not self.mw.scene or not hasattr(self.mw.scene, 'data'):
            return

        # 1-3. Find all items in scene for bond updates
        _, all_atoms, all_bonds = find_molecules(self.mw.scene)
        
        if not self.target_atoms: return

        # 5. Rotate the target molecule around its own COG
        Rx = np.array([[1, 0, 0], [0, np.cos(dy), -np.sin(dy)], [0, np.sin(dy), np.cos(dy)]])
        Ry = np.array([[np.cos(dx), 0, np.sin(dx)], [0, 1, 0], [-np.sin(dx), 0, np.cos(dx)]])
        R = Rx @ Ry

        for mol_atoms in [self.target_atoms]:
            points = np.array([[a.pos().x(), a.pos().y(), getattr(a, "z_3d", 0.0)] for a in mol_atoms])
            center = np.mean(points, axis=0)
            
            new_pts = (points - center) @ R.T + center
            
            for i, atom in enumerate(mol_atoms):
                atom.setPos(QPointF(new_pts[i, 0], new_pts[i, 1]))
                atom.z_3d = new_pts[i, 2]
                atom.setZValue(atom.z_3d)

        # 6. Update all bond positions
        for bond in all_bonds:
            if hasattr(bond, "update_position"):
                bond.update_position()
            # Depth sort bonds based on average Z
            if hasattr(bond.atom1, "z_3d") and hasattr(bond.atom2, "z_3d"):
                bond.setZValue((bond.atom1.z_3d + bond.atom2.z_3d) / 2.0)
                
        update_molecule_z_ranges(self.mw.scene)
        self.mw.scene.update()


def sip_isdeleted_safe(obj):
    try: return sip.isdeleted(obj)
    except: return False
