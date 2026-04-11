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
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
                             QGraphicsItem, QCheckBox, QFrame, QSpacerItem, 
                             QSizePolicy)
from PyQt6.QtCore import Qt, QPointF, QEvent, QObject, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QColor
import logging

# Metadata
PLUGIN_NAME = "3D Molecule on 2D"
PLUGIN_VERSION = "2.3.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = "Integrated 3D depth cues, rotation, and 3D-aware Mol export. Refactored for V3 API."


_enabled = True
_show_depth_cues = True
_rotate_tool_handler = None
_depth_cue_strength = 0.8  # 0.0 to 1.0
_3d_scale = 50.0           # Strictly matches 1.0 / ANGSTROM_PER_PIXEL (1.0 / 0.02 = 50)
_embed_without_h = True    # New option: Default to True per user request
_force_direct_mode = False # Force 2D-coordinate fallback
_active_worker = None
_mw = None
_context = None
_settings_file = os.path.splitext(os.path.abspath(__file__))[0] + ".json"
_plugin_menu_action = None  # QAction reference for Plugin menu entry

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
    # Ensure factor is in [0, 1]
    f = max(0.0, min(1.0, float(factor)))
    # Ensure color is a QColor object
    c = QColor(color)
    r = int(c.red() + (255 - c.red()) * f)
    g = int(c.green() + (255 - c.green()) * f)
    b = int(c.blue() + (255 - c.blue()) * f)
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

        # 1. General
        self.chk_enable = QCheckBox("Enable Plugin")
        self.chk_enable.setChecked(self.enabled)
        self.chk_enable.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.chk_enable)

        def add_separator():
            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setFrameShadow(QFrame.Shadow.Sunken)
            layout.addWidget(line)

        add_separator()

        # 2. Visuals
        layout.addWidget(QLabel("<b>Visual Cues</b>"))
        self.chk_depth_cues = QCheckBox("Show 3D Depth Cues (Whiting)")
        self.chk_depth_cues.setChecked(_show_depth_cues)
        layout.addWidget(self.chk_depth_cues)

        sld_layout = QHBoxLayout()
        sld_layout.addWidget(QLabel("  Strength:"))
        self.sld_strength = QSlider(Qt.Orientation.Horizontal)
        self.sld_strength.setRange(0, 100)
        self.sld_strength.setValue(int(_depth_cue_strength * 100))
        self.lbl_strength = QLabel(f"{self.sld_strength.value()}%")
        self.sld_strength.valueChanged.connect(lambda v: self.lbl_strength.setText(f"{v}%"))
        sld_layout.addWidget(self.sld_strength)
        sld_layout.addWidget(self.lbl_strength)
        layout.addLayout(sld_layout)

        add_separator()

        # 3. Embedding & Conversion
        layout.addWidget(QLabel("<b>3D Embedding</b>"))
        self.chk_embed_no_h = QCheckBox("Embed without Hydrogens (Cleaner)")
        self.chk_embed_no_h.setChecked(_embed_without_h)
        layout.addWidget(self.chk_embed_no_h)

        self.chk_force_direct = QCheckBox("Force Direct Conversion")
        self.chk_force_direct.setChecked(_force_direct_mode)
        self.chk_force_direct.setEnabled(_embed_without_h)
        self.chk_force_direct.setToolTip("Uses 2D coordinates directly. Only available if 'Embed without H' is active.")
        layout.addWidget(self.chk_force_direct)
        self.chk_embed_no_h.toggled.connect(self.chk_force_direct.setEnabled)

        add_separator()

        # 4. Geometry
        layout.addWidget(QLabel("<b>Dimensions</b>"))
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("  3D-to-2D Scale:"))
        self.sld_scale = QSlider(Qt.Orientation.Horizontal)
        self.sld_scale.setRange(10, 200)
        self.sld_scale.setValue(int(_3d_scale))
        self.lbl_scale = QLabel(f"{self.sld_scale.value()}")
        self.sld_scale.valueChanged.connect(lambda v: self.lbl_scale.setText(f"{v}"))
        scale_layout.addWidget(self.sld_scale)
        scale_layout.addWidget(self.lbl_scale)
        layout.addLayout(scale_layout)

        spacer = QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        layout.addItem(spacer)

        from PyQt6.QtWidgets import QDialogButtonBox
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        self.setLayout(layout)

    def accept(self):
        global _embed_without_h, _force_direct_mode, _depth_cue_strength, _3d_scale, _show_depth_cues
        self.enabled = self.chk_enable.isChecked()
        _embed_without_h = self.chk_embed_no_h.isChecked()
        _force_direct_mode = self.chk_force_direct.isChecked()
        _show_depth_cues = self.chk_depth_cues.isChecked()
        _depth_cue_strength = self.sld_strength.value() / 100.0
        _3d_scale = float(self.sld_scale.value())
        save_settings()
        super().accept()

def load_settings():
    global _enabled, _depth_cue_strength, _3d_scale, _embed_without_h, _force_direct_mode, _show_depth_cues
    try:
        if os.path.exists(_settings_file):
            with open(_settings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                _enabled = data.get('enabled', True)
                _show_depth_cues = data.get('show_depth_cues', True)
                _depth_cue_strength = data.get('depth_cue_strength', 0.8)
                _3d_scale = data.get('3d_scale', 50.0)
                _embed_without_h = data.get('embed_without_h', True)
                _force_direct_mode = data.get('force_direct_mode', False)
    except Exception as e:
        print(f"[{PLUGIN_NAME}] Error loading settings: {e}")

def save_settings():
    try:
        data = {
            'enabled': _enabled,
            'show_depth_cues': _show_depth_cues,
            'depth_cue_strength': _depth_cue_strength,
            '3d_scale': _3d_scale,
            'embed_without_h': _embed_without_h,
            'force_direct_mode': _force_direct_mode
        }
        with open(_settings_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[{PLUGIN_NAME}] Error saving settings: {e}")

def on_cleanup_triggered(*args, allow_trigger=True, **kwargs):
    """
    Smart 3D Trigger with Recursion Guard.
    """
    global _last_cleanup_trigger_time, _is_syncing
    mw = _mw
    context = _context
    if not mw or not mw.scene: return
    if _is_syncing: return
    
    _is_syncing = True
    try:
        # 0. Detect molecules in the scene
        molecules, all_atoms, all_bonds = find_molecules(mw.scene)
        if not molecules:
            context.show_status_message("No molecules in scene.")
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
                
                # Check 3: If Embed without H is enabled, but molecule was embedded With H
                if not needs_3d_refresh and _embed_without_h:
                    # If it has hydrogens but lacks the "skeleton-first" marker
                    if mol.GetNumAtoms() > mol.GetNumHeavyAtoms():
                        if not mol.HasProp("_pme_skeleton_embedded"):
                            needs_3d_refresh = True
                            status_msg = "Standard H-embedding detected. Re-embedding without H..."
        
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
            if _embed_without_h or _force_direct_mode:
                context.show_status_message(f"Smart 3D: {status_msg} Local Embedding starting (threaded)...")
                start_local_embedding(mw, _embed_without_h, _force_direct_mode)
            else:
                context.show_status_message(f"Smart 3D: {status_msg} Triggering Main Conversion...")
                if hasattr(mw, "trigger_conversion"):
                    global _plugin_triggered_conversion
                    _plugin_triggered_conversion = True
                    # # [DIRECT ACCESS] to core calculation trigger
                    mw.compute_manager.trigger_conversion()
            return # Wait for conversion to finish
        
        # 3. Perform Sync
        if mol and mol.GetNumConformers() > 0:
            if needs_3d_refresh:
                context.show_status_message("Smart 3D: Syncing whatever available...")
            else:
                context.show_status_message("Smart 3D: Syncing layout to 3D...")
            sync_to_3d_layout(mw, mol)
        else:
            context.show_status_message("Smart 3D: Conversion needed for synchronization.")
    finally:
        _is_syncing = False
    
    mw.scene.update()


    if _rotate_tool_handler and hasattr(_rotate_tool_handler, "rotate_act") and _rotate_tool_handler.rotate_act:
        _rotate_tool_handler.rotate_act.setChecked(True)

def show_settings_dialog(*args, **kwargs):
    if _mw:
        open_settings_dialog(_mw, _context)
    # This consolidates toolbar and menu actions to the same unified dialog.

def open_settings_dialog(mw, context):
    global _enabled

    dlg = PluginSettingsDialog(mw, _enabled)
    if dlg.exec():
        changed = False
        if _enabled != dlg.enabled:
            _enabled = dlg.enabled
            changed = True
            save_settings()
            if _enabled:
                enable_plugin(mw, context)
            else:
                disable_plugin(mw)
            status = "Enabled" if _enabled else "Disabled"
            context.show_status_message(f"{PLUGIN_NAME}: {status}")
        # Hot-reload any toolbar state so the actions reflect the current setting
        refresh_plugin_toolbar(mw, context)
        if not changed:
            save_settings()

def initialize(context):
    global _mw, _context

    mw = context.get_main_window()
    _mw = mw
    _context = context
    
    # Path to the settings file within the plugin directory
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    os.path.join(plugin_dir, "3d_molecule_on_2d.json")
    
    load_settings()
    
    # Register Setting Menu (always  Eneeded to re-enable the plugin)
    context.add_menu_action("Settings/3D Molecule on 2D...",
                            lambda: open_settings_dialog(mw, context))

    # Register Save/Load handlers for project file persistence
    context.register_save_handler(save_state)
    context.register_load_handler(load_state)

    # Only register toolbar actions if the plugin is enabled at startup.
    # This prevents the toolbar from flashing visible then hiding via QTimer.
    if _enabled:
        context.add_toolbar_action(on_cleanup_triggered, "Clean Up 3D", tooltip="Sync 2D layout to 3D")
        context.add_toolbar_action(lambda: None, "Rotate 3D", tooltip="Rotate molecule in 3D")
        context.add_toolbar_action(show_settings_dialog, "3D on 2D Settings...", tooltip="Fine-tune visuals")

    def startup_fix():

        # Locate and store the Plugin menu action reference, then set initial enabled state
        _find_menu_action(mw, _PLUGIN_MENU_ACTION_TEXT)
        if _enabled:
            enable_plugin(mw, context)

    QTimer.singleShot(0, startup_fix)

def enable_plugin(mw, context):
    global _rotate_tool_handler

    if not _rotate_tool_handler:
        _rotate_tool_handler = RotateToolHandler(mw)

    toggle_monkey_patches(True, mw)
    patch_export_logic(True)

    # Re-register toolbar actions with the plugin_manager if they were
    # never registered (plugin was disabled at startup) or were removed.
    pm = getattr(mw, 'plugin_manager', None)
    if pm and hasattr(pm, 'toolbar_actions'):
        already = any(a.get('text', None) in _OWN_ACTION_TEXTS for a in pm.toolbar_actions)
        if not already:
            context.add_toolbar_action(on_cleanup_triggered, "Clean Up 3D", tooltip="Sync 2D layout to 3D")
            context.add_toolbar_action(lambda: None, "Rotate 3D", tooltip="Rotate molecule in 3D")
            context.add_toolbar_action(show_settings_dialog, "3D on 2D Settings...", tooltip="Fine-tune visuals")

    refresh_plugin_toolbar(mw, context)
    configure_actions()

    if mw.scene:
        for item in mw.scene.items():
            if type(item).__name__ in ["AtomItem", "BondItem"]:
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        mw.scene.update()

_OWN_ACTION_TEXTS = {"Clean Up 3D", "Rotate 3D", "3D on 2D Settings..."}

_PLUGIN_MENU_ACTION_TEXT = "3D Molecule on 2D..."

def _find_menu_action(mw, text):
    """Recursively find a QAction by text in the menu bar."""
    def _search(menu):
        for a in menu.actions():
            if a.text() == text:
                return a
            if a.menu():
                found = _search(a.menu())
                if found:
                    return found
        return None
    for a in mw.menuBar().actions():
        if a.menu():
            found = _search(a.menu())
            if found:
                return found
    return None

def disable_plugin(mw):

    toggle_monkey_patches(False, mw)
    patch_export_logic(False)
    if _rotate_tool_handler:
        _rotate_tool_handler.set_active(False)

    # Remove from plugin_manager's registered list so _add_plugin_toolbar_actions
    # won't re-add them if the toolbar is rebuilt while disabled.
    pm = getattr(mw, 'plugin_manager', None)
    if pm and hasattr(pm, 'toolbar_actions'):
        pm.toolbar_actions = [a for a in pm.toolbar_actions
                              if a.get('text', None) not in _OWN_ACTION_TEXTS]

    # Remove from toolbar  Esearch by text so stale object refs don't matter
    tb = getattr(getattr(mw, 'init_manager', None), 'plugin_toolbar', None)
    if tb:
        for act in list(tb.actions()):
            if act.text() in _OWN_ACTION_TEXTS:
                tb.removeAction(act)
        # Hide toolbar if nothing non-separator remains
        if not any(not a.isSeparator() for a in tb.actions()):
            tb.hide()

    if mw.scene:
        mw.scene.update()

def refresh_plugin_toolbar(mw, context):
    tb = getattr(getattr(mw, 'init_manager', None), 'plugin_toolbar', None)
    if not tb:
        return

    # Rebuild the toolbar from the plugin_manager's registered list
    if hasattr(mw, 'init_manager') and hasattr(mw.init_manager, '_add_plugin_toolbar_actions'):
        mw.init_manager._add_plugin_toolbar_actions()

    tb.show()
    configure_actions()

def configure_actions():
    global _toolbar_actions_objs

    mw = _mw
    tb = getattr(getattr(mw, 'init_manager', None), 'plugin_toolbar', None)
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
    next((a for a in actions if a.text() == "3D on 2D Settings..."), None)
    
    # Set Checkable and connect toggled
    if rotate_act:
        rotate_act.setCheckable(True)
        try: rotate_act.toggled.disconnect()
        except Exception as _e:
            logging.warning("[3d_molecule_on_2d.py:474] silenced: %s", _e)
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
        if hasattr(mw.init_manager, "mode_actions") and "select" in mw.init_manager.mode_actions:
            mw.init_manager.mode_actions["select"].setChecked(True)
            mw.ui_manager.set_mode("select")
    
    # --- Undo/Redo Sync ---
    def on_undo_redo_changed():

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
        stack = getattr(mw.edit_actions_manager, "undo_stack", getattr(mw.edit_actions_manager, "undoStack", None))
        if hasattr(stack, "indexChanged"):
            stack.indexChanged.connect(lambda idx: on_undo_redo_changed())
        else:
            # Fallback to monkeypatching if it's a list or doesn't have indexChanged
            orig_undo = getattr(mw.edit_actions_manager, "undo", None)
            if orig_undo and callable(orig_undo) and not hasattr(orig_undo, "_is_patched"):
                def patched_undo(*args, **kwargs):
                    res = orig_undo(*args, **kwargs)
                    on_undo_redo_changed()
                    return res
                patched_undo._is_patched = True
                mw.edit_actions_manager.undo = patched_undo
            
            orig_redo = getattr(mw.edit_actions_manager, "redo", None)
            if orig_redo and callable(orig_redo) and not hasattr(orig_redo, "_is_patched"):
                def patched_redo(*args, **kwargs):
                    res = orig_redo(*args, **kwargs)
                    on_undo_redo_changed()
                    return res
                patched_redo._is_patched = True
                mw.edit_actions_manager.redo = patched_redo
    except Exception as e:
        _context.show_status_message(f"[{PLUGIN_NAME}] Warning: Could not hook into undo/redo system.")
        print(f"[{PLUGIN_NAME}] Warning: Could not hook into undo/redo system: {e}")

# --- Persistence ---

def save_state():

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
            item = atom_data.get('item', None)
            if item and not sip_isdeleted_safe(item) and hasattr(item, "z_3d"):
                # Save raw pixels for Z to match X/Y persistence in the core app.
                # This ensures that rotations and positions are saved 1:1 without scaling artifacts.
                z_data[str(aid)] = item.z_3d
        state["z_data"] = z_data
    return state

def load_state(data):

    mw = _mw
    if not mw or not data: return

    data.get("depth_cue_strength", 0.8)
    data.get("3d_scale", 50.0)
    data.get("embed_without_h", False)

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
                        atom_data = mw.scene.data.atoms.get(aid, None)
                    
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
    from moleditpy.ui.compute_logic import ComputeManager as MainWindowCompute
    if hasattr(MainWindowCompute, "_original_on_calculation_finished"):
        MainWindowCompute._original_on_calculation_finished(self, result)
    

    if _enabled and _mw == self and _plugin_triggered_conversion:
        # Reset flag IMMEDIATELY to prevent loop re-entry
        pass
        # Defer slightly for core app UI cleanup
        QTimer.singleShot(700, lambda: on_cleanup_triggered(allow_trigger=False))

def toggle_monkey_patches(active, mw=None):
    from moleditpy.ui.atom_item import AtomItem
    from moleditpy.ui.bond_item import BondItem
    global _original_atom_paint, _original_bond_paint
    if active:
        if _original_atom_paint is None:
            _original_atom_paint = AtomItem.paint
            AtomItem.paint = patched_atom_paint
        if _original_bond_paint is None:
            _original_bond_paint = BondItem.paint
            BondItem.paint = patched_bond_paint
            
        # Patch MainWindowCompute to detect conversion finish
        from moleditpy.ui.compute_logic import ComputeManager as MainWindowCompute
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
        from moleditpy.ui.compute_logic import ComputeManager as MainWindowCompute
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
                item = self.atoms[aid].get("item", None)
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
    from moleditpy.ui.io_logic import IOManager as MainWindowMolecularParsers
    _export_in_progress = True
    try:
        if hasattr(MainWindowMolecularParsers, "_original_save_as_mol") and MainWindowMolecularParsers._original_save_as_mol:
             return MainWindowMolecularParsers._original_save_as_mol(self, *args, **kwargs)
        return self.save_as_mol(*args, **kwargs) # Fallback (should not happen if patched)
    finally:
        _export_in_progress = False

def patch_export_logic(active=True):
    """Monkey patch MolecularData and Parsers for 3D-aware Mol export."""
    from moleditpy.core.molecular_data import MolecularData
    from moleditpy.ui.io_logic import IOManager as MainWindowMolecularParsers
    
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

    from moleditpy.ui import atom_item
    
    if _show_depth_cues and getattr(self, "z_3d", None) is not None and self.scene():
        # Use per-molecule range if available, else scene range
        z_min = getattr(self, "mol_z_min", None)
        z_max = getattr(self, "mol_z_max", None)
        
        if z_min is None or z_max is None:
            z_min, z_max = get_scene_z_range(self.scene())
            
        z_range = z_max - z_min
        if _show_depth_cues and z_range > 1e-4:
            # depth_factor: 1.0 (nearest) to 0.0 (farthest)
            depth_factor = (self.z_3d - z_min) / z_range
            # DISTANT is whiter: factor=1.0 at z_min, factor=0.0 at z_max
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
            # depth_factor: 1.0 (nearest) to 0.0 (farthest)
            depth_factor = (avg_z - z_min) / z_range
            # DISTANT is whiter: factor=1.0 at z_min, factor=0.0 at z_max
            white_factor = (1.0 - depth_factor) * _depth_cue_strength
            
            if white_factor > 0.05:
                # Temporary Swap Bond Color in Settings to influence internal drawing
                win = self.scene().views()[0].window() if self.scene() and self.scene().views() else None
                if win and hasattr(win.init_manager, "settings"):
                    orig_hex = win.init_manager.settings.get("bond_color_2d", "#222222")
                    faded_color = blend_with_white(QColor(orig_hex), white_factor)
                    win.init_manager.settings["bond_color_2d"] = faded_color.name()
                    try:
                        _original_bond_paint(self, painter, option, widget)
                    finally:
                        win.init_manager.settings["bond_color_2d"] = orig_hex
                    return

    _original_bond_paint(self, painter, option, widget)

def get_scene_z_range(scene):
    zs = []
    if hasattr(scene, 'data') and hasattr(scene.data, 'atoms'):
        for atom_data in scene.data.atoms.values():
            item = atom_data.get('item', None)
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
        
        # Calculate XY Footprint for proportional depth cue scaling
        xs = [a.pos().x() for a in mol_atoms]
        ys = [a.pos().y() for a in mol_atoms]
        max(max(xs)-min(xs), max(ys)-min(ys), 100.0)
        
        actual_depth = z_max - z_min
        # User Feedback: Whiting should match molecule dimension (depth) and be visible.
        # We use the actual depth as the range for maximum sensitivity.
        # But we must avoid division by zero for perfectly flat mols.
        ref_z_range = max(actual_depth, 20.0) 
        
        # User Feedback: "the top must be z=0" -> Reference point for NO whiting.
        # We set mol_z_max = z_max. At z_max, depth_factor=1.0, so white_factor=0.0.
        for a in mol_atoms:
            a.mol_z_max = z_max
            a.mol_z_min = z_max - ref_z_range

class LocalCalculationWorker(QObject):
    finished = pyqtSignal(object)
    status = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, mol_block, embed_without_h, force_direct_mode, atom_ids):
        super().__init__()
        self.mol_block = mol_block
        self.embed_without_h = embed_without_h
        self.force_direct_mode = force_direct_mode
        self.atom_ids = atom_ids

    def run(self):
        try:
            self.status.emit("Calculating 3D structure...")
            res = -1
            
            # 1. Convert 2D data to RDKit mol
            # We use removeHs=False to maintain a stable 1:1 mapping with self.atom_ids
            mol = Chem.MolFromMolBlock(self.mol_block, removeHs=False)
            if not mol:
                self.error.emit("Failed to create molecule structure.")
                return

            # Robust mapping: index -> original ID (stashed on atoms)
            for i, aid in enumerate(self.atom_ids):
                if i < mol.GetNumAtoms():
                    mol.GetAtomWithIdx(i).SetIntProp("_original_atom_id", aid)
            
            # Prepare state for embedding
            if self.embed_without_h:
                mol = Chem.RemoveHs(mol)
                # RemoveHs preserves Properties, so _original_atom_id is still there.

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
            # Convert once as a whole molecule; fragment separation is handled later in sync analysis.

            if self.force_direct_mode:
                self.status.emit("Force Direct Mode: Skipping RDKit embedding...")
                res = -1
            elif self.embed_without_h:
                self.status.emit("Performing RDKit embedding...")
                apply_stereo(mol, explicit_stereo)
                res = AllChem.EmbedMolecule(mol, params)
                if res == -1:
                    params.useRandomCoords = True
                    res = AllChem.EmbedMolecule(mol, params)
                
                if res != -1:
                    self.status.emit("Placing hydrogens on 3D skeleton...")
                    mol = Chem.AddHs(mol, addCoords=True)
                    mol.SetIntProp("_pme_skeleton_embedded", 1)
            else:
                self.status.emit("Performing RDKit standard embedding...")
                mol = Chem.AddHs(mol)
                apply_stereo(mol, explicit_stereo)
                res = AllChem.EmbedMolecule(mol, params)
                if res == -1:
                    params.useRandomCoords = True
                    res = AllChem.EmbedMolecule(mol, params)

            # Check if embedding actually succeeded in adding a conformer
            if res == -1 or mol.GetNumConformers() == 0:
                self.status.emit("Embedding failed. Using direct coordinate fallback...")
                
                # RECOVERY: Re-parse from mol_block to get fresh 2D coordinates.
                # Hs are kept here to ensure _original_atom_id mapping is easy.
                mol = Chem.MolFromMolBlock(self.mol_block, removeHs=False)
                if mol is None:
                    self.error.emit("3D embedding and fallback both failed.")
                    return
                
                # Restore original IDs
                for i, aid in enumerate(self.atom_ids):
                    if i < mol.GetNumAtoms():
                        mol.GetAtomWithIdx(i).SetIntProp("_original_atom_id", aid)

                # If we are NOT in "Embed without H" mode, add them now for proper 3D placement
                if not self.embed_without_h:
                    mol = Chem.AddHs(mol)

                # Need 2D coordinates for the "direct" method
                if mol.GetNumConformers() == 0:
                    AllChem.Compute2DCoords(mol)
                
                conf = mol.GetConformer()
                
                # 1. Try to break planarity using wedge/dash stereo bonds (like main app)
                applied_stereo = False
                for b in mol.GetBonds():
                    bdir = b.GetBondDir()
                    if bdir in [Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH]:
                        idx = b.GetEndAtomIdx()
                        cp = conf.GetAtomPosition(idx)
                        offset = 1.5 if bdir == Chem.BondDir.BEGINWEDGE else -1.5
                        conf.SetAtomPosition(idx, Point3D(cp.x, cp.y, cp.z + offset))
                        applied_stereo = True
                
                # 2. If no stereo info, apply small random Z-jitter (±0.1 ÁE
                if not applied_stereo:
                    rng = np.random.default_rng(seed=42)
                    for i in range(mol.GetNumAtoms()):
                        cp = conf.GetAtomPosition(i)
                        jitter = (rng.random() - 0.5) * 0.2
                        conf.SetAtomPosition(i, Point3D(cp.x, cp.y, cp.z + jitter))
                
                res = 0 # Proceed to optimization
                
                # If "Embed without H" is ON, we must remove them now for CLEAN optimization
                if self.embed_without_h:
                    mol = Chem.RemoveHs(mol)

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
                except Exception as _e:
                    logging.warning("[3d_molecule_on_2d.py:1051] silenced: %s", _e)

            # 5. Final Push
            self.status.emit("3D structure ready.")
            self.finished.emit(mol)
            
        except Exception as e:
            self.error.emit(str(e))

def start_local_embedding(mw, embed_without_h=False, force_direct_mode=False):
    """
    Start local embedding in a background thread.
    """
    global _active_worker

    if _active_worker:
        return

    mol_block = mw.state_manager.data.to_mol_block()
    if not mol_block:
        return

    atom_ids = list(mw.state_manager.data.atoms.keys())
    
    # Setup thread and worker
    thread = QThread()
    worker = LocalCalculationWorker(mol_block, embed_without_h, force_direct_mode, atom_ids)
    worker.moveToThread(thread)
    
    _active_worker = (thread, worker)
    
    # Connect signals
    def update_ui_status(msg):
        mw.statusBar().showMessage(msg)
        if hasattr(mw, "_calculating_text_actor"):
            try:
                # Update text in the 3D window if it exists
                actor = mw._calculating_text_actor
                if hasattr(actor, "SetInput"):
                    actor.SetInput(msg)
                elif hasattr(actor, "SetText"):
                    actor.SetText(1, msg)  # CornerAnnotation: 1 = lower-right
                mw.plotter.render()
            except Exception as _e:
                logging.warning("[3d_molecule_on_2d.py:1089] silenced: %s", _e)

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
            bg_color_hex = mw.init_manager.settings.get("background_color", "#919191")
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
        except Exception as _e:
            logging.warning("[3d_molecule_on_2d.py:1127] silenced: %s", _e)

    thread.start()

def on_embedding_error(mw, msg):
    # Remove overlay
    if hasattr(mw, "plotter") and mw.plotter and hasattr(mw, "_calculating_text_actor"):
        try:
            mw.plotter.remove_actor(mw._calculating_text_actor)
            mw.plotter.render()
        except Exception as _e:
            logging.warning("[3d_molecule_on_2d.py:1137] silenced: %s", _e)
    mw.statusBar().showMessage(f"Smart 3D Error: {msg}")

def on_embedding_finished(mw, mol):
    # Remove overlay
    if hasattr(mw, "plotter") and mw.plotter and hasattr(mw, "_calculating_text_actor"):
        try:
            mw.plotter.remove_actor(mw._calculating_text_actor)
            mw.plotter.render()
        except Exception as _e:
            logging.warning("[3d_molecule_on_2d.py:1146] silenced: %s", _e)
    
    mw.current_mol = mol
    
    # CRITICAL APP STATE SYNC: Re-establish atom mapping and chiral labels before drawing.
    if hasattr(mw.compute_manager, "create_atom_id_mapping"):
        mw.compute_manager.create_atom_id_mapping()
    if hasattr(mw.view_3d_manager, "update_chiral_labels"):
        mw.view_3d_manager.update_chiral_labels()
    
    # Update 3D viewer
    if hasattr(mw.view_3d_manager, "draw_molecule_3d"):
        mw.view_3d_manager.draw_molecule_3d(mol)
    
    # Perform Sync
    sync_to_3d_layout(mw, mol)
    # Ensure Z-ranges are calculated before re-painting
    update_molecule_z_ranges(mw.scene)
    
    # Ensure the RDKit molecule is fully recognized as 3D by the app
    if mol and mol.GetNumConformers() > 0:
        mol.UpdatePropertyCache(False)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    # CRITICAL APP STATE SYNC: Notify main application that 3D conversion is finished
    if _context:
        _context.push_undo_checkpoint()
    if hasattr(mw.ui_manager, "_enable_3d_features"):
        mw.ui_manager._enable_3d_features(True)
    if hasattr(mw, "plotter") and mw.plotter:
        try:
            mw.plotter.reset_camera()
            mw.plotter.render()
        except Exception as _e:
            logging.warning("[3d_molecule_on_2d.py:1179] silenced: %s", _e)
    if hasattr(mw.view_3d_manager, "setup_3d_hover"):
        mw.view_3d_manager.setup_3d_hover()
    if hasattr(mw, "init_manager") and hasattr(mw.init_manager, "view_2d"):
        mw.init_manager.view_2d.setFocus()
    
    mw.statusBar().showMessage("Smart 3D: Local Embedding and Sync completed.")
    mw.scene.update()

    # Re-enable and activate Rotate tool

    if _rotate_tool_handler and hasattr(_rotate_tool_handler, "rotate_act") and _rotate_tool_handler.rotate_act:
        try:
            _rotate_tool_handler.rotate_act.setEnabled(True)
            # Force a small delay to ensure the scene is ready for rotation
            QTimer.singleShot(100, lambda: _rotate_tool_handler.rotate_act.setChecked(True))
        except Exception as _e:
            logging.warning("[3d_molecule_on_2d.py:1195] silenced: %s", _e)

# --- Logic ---

def sync_to_3d_layout(mw, mol):
    if not mol or mol.GetNumConformers() == 0: return
    conf = mol.GetConformer()
    scale = _3d_scale

    # 2. Find molecules in scene
    molecules, all_atoms, all_bonds = find_molecules(mw.scene)

    # Build RDKit fragments as independent coordinate maps: aid -> (x, y, z)
    rdkit_frags = []
    try:
        from rdkit.Chem import rdmolops
        frag_atom_indices = rdmolops.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    except Exception:
        frag_atom_indices = [tuple(range(mol.GetNumAtoms()))]

    for frag in frag_atom_indices:
        fmap = {}
        for idx in frag:
            aid = get_original_id(mol.GetAtomWithIdx(idx))
            if aid is None:
                continue
            p = conf.GetAtomPosition(idx)
            fmap[aid] = (p.x * scale, -p.y * scale, p.z * scale)
        if fmap:
            rdkit_frags.append(fmap)

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
    
    # 4. Sync target molecules with best-matching RDKit fragment
    used_frag_indices = set()
    for i, mol_atoms in enumerate(molecules):
        if selected_items and i not in target_mol_indices:
            continue

        scene_atom_ids = {getattr(a, "atom_id", None) for a in mol_atoms if getattr(a, "atom_id", None) is not None}
        best_frag_idx = None
        best_score = -1
        for frag_idx, fmap in enumerate(rdkit_frags):
            if frag_idx in used_frag_indices:
                continue
            score = len(scene_atom_ids.intersection(fmap.keys()))
            if score > best_score:
                best_score = score
                best_frag_idx = frag_idx

        if best_frag_idx is None and rdkit_frags:
            best_frag_idx = 0
        if best_frag_idx is None:
            continue

        proj_coords = rdkit_frags[best_frag_idx]
        used_frag_indices.add(best_frag_idx)

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
        self.mw.init_manager.view_2d.viewport().installEventFilter(self)
        
    def set_active(self, state):
        self.active = state
        if state:
            self.mw.ui_manager.set_mode("plugin_rotate_3d")
            self.ensure_z_coords()
        else:
            self.target_atoms = None
            self.mw.scene.mode = "select"
            if hasattr(self.mw.ui_manager, "activate_select_mode"): self.mw.ui_manager.activate_select_mode()
        
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
                item = self.mw.init_manager.view_2d.itemAt(event.position().toPoint())
                
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
