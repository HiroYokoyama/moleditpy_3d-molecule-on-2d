import os
import json
import numpy as np
try:
    from rdkit import Chem
    from rdkit.Geometry import Point3D
except ImportError:
    pass
try:
    from PyQt6 import sip
except ImportError:
    import sip
from PyQt6.QtWidgets import QMenu, QToolBar, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGraphicsScene, QGraphicsItem, QCheckBox
from PyQt6.QtCore import Qt, QPointF, QEvent, QObject, QTimer
from PyQt6.QtGui import QColor, QPen, QIcon, QAction, QActionGroup, QPainter, QBrush

# Metadata
PLUGIN_NAME = "3D Molecule on 2D"
PLUGIN_VERSION = "1.1.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = "Integrated 3D depth cues, rotation, and 3D-aware Mol export."

# Global state
_enabled = True
_show_depth_cues = True
_rotate_tool_handler = None
_depth_cue_strength = 0.8  # 0.0 to 1.0
_3d_scale = 50.0           # Matches ANGSTROM_PER_PIXEL (0.02) exactly
_mw = None
_context = None
_settings_file = os.path.splitext(os.path.abspath(__file__))[0] + ".json"

# Store original paint methods
_original_atom_paint = None
_original_bond_paint = None
_original_save_as_mol = None
_toolbar_actions_objs = []
_export_in_progress = False

def blend_with_white(color, factor):
    """Linearly interpolate color towards white."""
    if factor <= 0.0: return color
    f = min(1.0, factor)
    r = int(color.red() + (255 - color.red()) * f)
    g = int(color.green() + (255 - color.green()) * f)
    b = int(color.blue() + (255 - color.blue()) * f)
    return QColor(r, g, b)

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
        from PyQt6.QtWidgets import QDialogButtonBox
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        self.setLayout(layout)

    def accept(self):
        self.enabled = self.chk_enable.isChecked()
        super().accept()

def load_settings():
    global _enabled
    try:
        if os.path.exists(_settings_file):
            import json
            with open(_settings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                _enabled = data.get('enabled', True)
    except Exception as e:
        print(f"[{PLUGIN_NAME}] Error loading settings: {e}")

def save_settings():
    try:
        import json
        data = {'enabled': _enabled}
        with open(_settings_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[{PLUGIN_NAME}] Error saving settings: {e}")

def on_cleanup_triggered(*args, **kwargs):
    # Safety: We NEVER call mw.trigger_conversion() here anymore to avoid modifying 
    # the user's saved 3D coordinates. We just sync to what exists.
    # Note: args[0] may be a bool (checked state) if called from QAction.triggered
    global _mw
    mw = _mw
    if not mw: return
    mol = mw.current_mol
    if not mol: return

    has_3d_conf = mol and mol.GetNumConformers() > 0
    if has_3d_conf:
        # Check if the conformer is effectively 2D (flat)
        is_flat = True
        conf = mol.GetConformer()
        zs = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
        if zs and (max(zs) - min(zs)) > 1e-4:
            is_flat = False
            
        mw.statusBar().showMessage("Syncing layout to existing 3D data...")
        sync_to_3d_layout(mw, mol)
        if is_flat:
             mw.statusBar().showMessage("Note: Source is 2D (flat). Layout synced to Z=0.")
    else:
        mw.statusBar().showMessage("No 3D data available. Use main Conversion tool first.")
    
    # Force bonds to be non-movable again just in case
    for item in mw.scene.items():
        if type(item).__name__ == "BondItem" and not sip_isdeleted_safe(item):
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
    
    mw.scene.update()

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
        scale_value_label.setText(f"{_3d_scale:.1f}")
        
        # Dyna-scale existing values in the scene for immediate visual feedback
        if mw.scene and abs(old_scale) > 1e-4:
            factor = _3d_scale / old_scale
            molecules, _, all_bonds = find_molecules(mw.scene)
            
            for mol_atoms in molecules:
                # Scale positions relative to molecule center
                pts = np.array([[a.pos().x(), a.pos().y()] for a in mol_atoms])
                center = np.mean(pts, axis=0)
                new_pts = (pts - center) * factor + center
                
                for i, atom in enumerate(mol_atoms):
                    atom.setPos(QPointF(new_pts[i, 0], new_pts[i, 1]))
                    if hasattr(atom, "z_3d"):
                        atom.z_3d *= factor
                        atom.setZValue(atom.z_3d)
                        
            for bond in all_bonds:
                if hasattr(bond, "update_position"):
                    bond.update_position()
                if hasattr(bond.atom1, "z_3d") and hasattr(bond.atom2, "z_3d"):
                    bond.setZValue((bond.atom1.z_3d + bond.atom2.z_3d) / 2.0)
            
            update_molecule_z_ranges(mw.scene)
            mw.scene.update()
    scale_slider.valueChanged.connect(on_scale_changed)
    scale_layout.addWidget(scale_label); scale_layout.addWidget(scale_slider); scale_layout.addWidget(scale_value_label)
    layout.addLayout(scale_layout)

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
                    bond.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
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
        "3d_scale": _3d_scale
    }
    if mw.scene and hasattr(mw.scene, 'data'):
        z_data = {}
        for aid, atom_data in mw.scene.data.atoms.items():
            item = atom_data.get('item')
            if item and not sip_isdeleted_safe(item) and hasattr(item, "z_3d"):
                # Store unscaled Z to make it setting-independent
                z_data[str(aid)] = item.z_3d / _3d_scale
        state["z_data"] = z_data
    return state

def load_state(data):
    global _depth_cue_strength, _3d_scale, _mw, _enabled, _context
    mw = _mw
    if not mw: return
    if data:
        _depth_cue_strength = data.get("depth_cue_strength", 0.8)
        _3d_scale = data.get("3d_scale", 50.0)
        
        # If the file has 3D data, we temporary enable the plugin if it's currently disabled.
        has_3d_data = "z_data" in data and len(data["z_data"]) > 0
        if has_3d_data and not _enabled:
            # Temporary enable (do not change global _enabled setting)
            print(f"[{PLUGIN_NAME}] 3D data found in project. Temporarily enabling plugin.")
            # Delay slightly to ensure UI is ready
            QTimer.singleShot(0, lambda: enable_plugin(mw, _context))

        if "z_data" in data and mw.scene and hasattr(mw.scene, 'data'):
            z_map = data["z_data"]
            for aid_str, z in z_map.items():
                try:
                    aid = int(aid_str)
                    # Robust lookup: Try data dict first
                    atom_data = None
                    if hasattr(mw.scene, 'data') and mw.scene.data:
                        atom_data = mw.scene.data.atoms.get(aid)
                    
                    item = None
                    if atom_data and 'item' in atom_data:
                        item = atom_data['item']
                    else:
                        # Fallback: Search the scene for an AtomItem with this ID
                        for i in mw.scene.items():
                            if type(i).__name__ == "AtomItem" and getattr(i, "atom_id", None) == aid:
                                item = i
                                break
                    
                    if item and not sip_isdeleted_safe(item):
                        # Re-scale based on the current _3d_scale
                        item.z_3d = z * _3d_scale
                        # White is back: Higher Z is closer
                        item.setZValue(item.z_3d)
                except Exception: continue

    # Ensure Z coordinates are present (either from z_data or the RDKit molecule)
    def finalized_restore():
        if _rotate_tool_handler:
            _rotate_tool_handler.ensure_z_coords()
        if mw.scene:
            # Refresh all bonds and Z-order
            _, _, all_bonds = find_molecules(mw.scene)
            for bond in all_bonds:
                bond.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                if hasattr(bond, "update_position"): bond.update_position()
                if hasattr(bond.atom1, "z_3d") and hasattr(bond.atom2, "z_3d"):
                    bond.setZValue((bond.atom1.z_3d + bond.atom2.z_3d) / 2.0)
            
            update_molecule_z_ranges(mw.scene)
            mw.scene.update()

    QTimer.singleShot(500, finalized_restore)

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
            # Ensure bonds are never movable
            if not hasattr(BondItem, "_original_init"):
                BondItem._original_init = BondItem.__init__
                def bond_init_fix(self, *args, **kwargs):
                    BondItem._original_init(self, *args, **kwargs)
                    self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                BondItem.__init__ = bond_init_fix
            
            # Also patch setFlag and setFlags to ignore ItemIsMovable for bonds
            if not hasattr(BondItem, "_original_setFlag"):
                BondItem._original_setFlag = BondItem.setFlag
                def bond_set_flag_guarded(self, flag, enabled=True):
                    if flag == QGraphicsItem.GraphicsItemFlag.ItemIsMovable and enabled:
                        # Force disable movement even if requested
                        BondItem._original_setFlag(self, flag, False)
                        return
                    BondItem._original_setFlag(self, flag, enabled)
                BondItem.setFlag = bond_set_flag_guarded
 
            if not hasattr(BondItem, "_original_setFlags"):
                BondItem._original_setFlags = BondItem.setFlags
                def bond_set_flags_guarded(self, flags):
                    # Mask out ItemIsMovable
                    safe_flags = flags & ~QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                    BondItem._original_setFlags(self, safe_flags)
                BondItem.setFlags = bond_set_flags_guarded
 
            # Final guard: Override flags() to always return non-movable
            if not hasattr(BondItem, "_original_flags"):
                BondItem._original_flags = BondItem.flags
                def bond_flags_guarded(self):
                    f = BondItem._original_flags(self)
                    return f & ~QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                BondItem.flags = bond_flags_guarded
 
            # Veto ItemPositionChange if it doesn't come from programmatic setPos
            if not hasattr(BondItem, "_original_itemChange"):
                BondItem._original_itemChange = BondItem.itemChange
                def bond_item_change_guarded(self, change, value):
                    return BondItem._original_itemChange(self, change, value)
                BondItem.itemChange = bond_item_change_guarded

            # Also force update all existing bonds in the scene
            if mw and mw.scene:
                for item in mw.scene.items():
                    if type(item).__name__ == "BondItem" and not sip_isdeleted_safe(item):
                        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
    else:
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
            atom_rd = mol.GetAtomWithIdx(i)
            if atom_rd.HasProp("_original_atom_id"):
                aid = atom_rd.GetIntProp("_original_atom_id")
                if aid in self.atoms:
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

# --- Logic ---

def sync_to_3d_layout(mw, mol):
    if not mol or mol.GetNumConformers() == 0: return
    conf = mol.GetConformer()
    scale = _3d_scale
    
    # 1. Map original_id to 3D coordinates
    proj_coords = {}
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        p = conf.GetAtomPosition(i)
        oid = None
        if atom.HasProp("_original_atom_id"): oid = int(atom.GetProp("_original_atom_id"))
        elif atom.HasProp("original_id"): oid = int(atom.GetProp("original_id"))
        if oid is not None:
            proj_coords[oid] = (p.x * scale, -p.y * scale, p.z * scale)

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
        bond.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
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
                atom = mol.GetAtomWithIdx(i)
                if atom.HasProp("_original_atom_id"):
                    try:
                        id_to_idx[atom.GetIntProp("_original_atom_id")] = i
                    except Exception:
                        try:
                            id_to_idx[int(atom.GetProp("_original_atom_id"))] = i
                        except Exception: pass
            
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
