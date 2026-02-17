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
from PyQt6.QtWidgets import QMenu, QToolBar, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGraphicsScene, QGraphicsItem
from PyQt6.QtCore import Qt, QPointF, QEvent, QObject, QTimer
from PyQt6.QtGui import QColor, QPen, QIcon, QAction, QActionGroup, QPainter, QBrush

# Metadata
PLUGIN_NAME = "3D Molecule on 2D"
PLUGIN_VERSION = "0.0.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = "Integrated 3D depth cues and rotation."

# Global state
_show_depth_cues = True
_rotate_tool_handler = None
_depth_cue_strength = 0.8  # 0.0 to 1.0
_3d_scale = 45.0           # Default is 1.5x of previous 30.0

# Store original paint methods
_original_atom_paint = None
_original_bond_paint = None

def blend_with_white(color, factor):
    """Linearly interpolate color towards white."""
    if factor <= 0.0: return color
    f = min(1.0, factor)
    r = int(color.red() + (255 - color.red()) * f)
    g = int(color.green() + (255 - color.green()) * f)
    b = int(color.blue() + (255 - color.blue()) * f)
    return QColor(r, g, b)

def initialize(context):
    global _rotate_tool_handler, _original_atom_paint, _original_bond_paint
    
    mw = context.get_main_window()
    _rotate_tool_handler = RotateToolHandler(mw)
    
    def on_rotate_toggled(checked):
        if checked:
            # Ensure other tools in the group are unchecked
            if hasattr(mw, "tool_group"):
                for act in mw.tool_group.actions():
                    if act.text() != "Rotate 3D" and act.isChecked():
                        act.setChecked(False)
            _rotate_tool_handler.set_active(True)
        else:
            _rotate_tool_handler.set_active(False)
            # Default back to select mode if we are turning off rotation
            if hasattr(mw, "mode_actions") and "select" in mw.mode_actions:
                mw.mode_actions["select"].setChecked(True)
                mw.set_mode("select")

    def on_cleanup_triggered():
        global _show_depth_cues
        mol = mw.current_mol
        
        all_atoms = [i for i in mw.scene.items() if type(i).__name__ == "AtomItem" and not sip_isdeleted_safe(i)]
        if not all_atoms: return

        # Check if RDKit mol has 3D data
        has_3d_conf = mol and mol.GetNumConformers() > 0
        
        # Check if all atoms in scene have mapping to RDKit mol
        mapped_oids = set()
        if mol:
            for a in mol.GetAtoms():
                try:
                    if a.HasProp("_original_atom_id"): mapped_oids.add(a.GetIntProp("_original_atom_id"))
                    elif a.HasProp("original_id"): mapped_oids.add(a.GetIntProp("original_id"))
                except: pass
        
        scene_oids = {a.atom_id for a in all_atoms}
        # If all scene atoms are mapped to the current RDKit model, we don't need re-conversion
        all_mapped = scene_oids.issubset(mapped_oids) and len(scene_oids) > 0
        
        if has_3d_conf and all_mapped:
            # We have 3D data and mapping, just sync positions
            mw.statusBar().showMessage("Syncing 2D layout to existing 3D data...")
            sync_to_3d_layout(mw, mol)
        else:
            # Missing 3D data or new atoms added that aren't in the model yet
            mw.statusBar().showMessage("Generating 3D coordinates...")
            mw.trigger_conversion()
            # Wait for conversion to complete (conversion uses a thread internally usually)
            QTimer.singleShot(1100, lambda: sync_to_3d_layout(mw, mw.current_mol))
        
        # Force bonds to be non-movable again just in case
        for item in mw.scene.items():
            if type(item).__name__ == "BondItem" and not sip_isdeleted_safe(item):
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        
        mw.scene.update()

    def show_settings_dialog():
        global _depth_cue_strength
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
                
                # Sync back to RDKit conformer if it exists
                # This ensures the new scale is reflected in the saved data
                mol = mw.current_mol
                if mol and mol.GetNumConformers() > 0:
                    conf = mol.GetConformer()
                    aid_to_idx = {}
                    for i in range(mol.GetNumAtoms()):
                        atom_rd = mol.GetAtomWithIdx(i)
                        if atom_rd.HasProp("_original_atom_id"):
                            try: aid_to_idx[atom_rd.GetIntProp("_original_atom_id")] = i
                            except: pass
                    
                    for mol_atoms in molecules:
                        for atom_item in mol_atoms:
                            aid = atom_item.atom_id
                            if aid in aid_to_idx:
                                idx = aid_to_idx[aid]
                                pos = atom_item.pos()
                                # Store relative coordinates (unscaled, flip Y)
                                conf.SetAtomPosition(idx, Point3D(pos.x() / _3d_scale, -pos.y() / _3d_scale, atom_item.z_3d / _3d_scale))

                update_molecule_z_ranges(mw.scene)
                mw.scene.update()
        scale_slider.valueChanged.connect(on_scale_changed)
        scale_layout.addWidget(scale_label); scale_layout.addWidget(scale_slider); scale_layout.addWidget(scale_value_label)
        layout.addLayout(scale_layout)

        ok_btn = QPushButton("OK"); ok_btn.clicked.connect(dlg.accept); layout.addWidget(ok_btn)
        dlg.exec()

    # --- Standard Registration ---
    
    context.add_toolbar_action(lambda: None, "Rotate 3D", tooltip="Rotate molecule in 3D")
    context.add_toolbar_action(on_cleanup_triggered, "Clean Up 3D", tooltip="Sync 2D layout to 3D")
    context.add_toolbar_action(show_settings_dialog, "3D on 2D Setting", tooltip="Settings")
    
    # --- Post-Processing (Refine toolbar UI) ---
    
    def configure_actions():
        tb = getattr(mw, 'plugin_toolbar', None)
        if not tb: return
        
        actions = list(tb.actions())
        rotate_act = next((a for a in actions if a.text() == "Rotate 3D"), None)
        cleanup_act = next((a for a in actions if a.text() == "Clean Up 3D"), None)
        settings_act = next((a for a in actions if a.text() == "3D on 2D Setting"), None)
        
        # Set Checkable and connect toggled
        if rotate_act:
            rotate_act.setCheckable(True)
            rotate_act.toggled.connect(on_rotate_toggled)
            _rotate_tool_handler.rotate_act = rotate_act
        
        # Mode Integration: Add Rotate 3D to main tool group for exclusivity
        if rotate_act and hasattr(mw, "tool_group"):
            mw.tool_group.addAction(rotate_act)
        
        # Visual Grouping (Separators)
        if rotate_act and cleanup_act and not any(a.isSeparator() for a in tb.actions() if tb.actionAt(tb.actionGeometry(cleanup_act).topLeft()) == a):
            mw.plugin_toolbar.insertSeparator(cleanup_act)
        if cleanup_act and settings_act:
            mw.plugin_toolbar.insertSeparator(settings_act)
            
        tb.show()

    QTimer.singleShot(0, configure_actions)
    
    # --- Undo/Redo Sync ---
    def on_undo_redo_changed():
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

    # Persistence
    def save_state():
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
        global _depth_cue_strength, _3d_scale
        if data:
            _depth_cue_strength = data.get("depth_cue_strength", 0.8)
            _3d_scale = data.get("3d_scale", 45.0)
            
            if "z_data" in data and mw.scene and hasattr(mw.scene, 'data'):
                z_map = data["z_data"]
                for aid_str, z in z_map.items():
                    try:
                        aid = int(aid_str)
                        atom_data = mw.scene.data.atoms.get(aid)
                        if atom_data and 'item' in atom_data:
                            item = atom_data['item']
                            if item and not sip_isdeleted_safe(item):
                                # Re-scale based on the current _3d_scale
                                item.z_3d = z * _3d_scale
                                # White is back: Higher Z is closer
                                item.setZValue(item.z_3d)
                    except ValueError: continue

        # Ensure Z coordinates are present (either from z_data or the RDKit molecule)
        # We wait a bit for the scene to be fully loaded and populated
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

    context.register_save_handler(save_state)
    context.register_load_handler(load_state)

    # Initial patches
    toggle_monkey_patches(True)

def toggle_monkey_patches(active):
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
                    if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
                        # If the scene is in select mode and we are NOT in the middle of a sync/rotation,
                        # dragging the bond directly should be blocked.
                        # However, since ItemIsMovable is False, this normally shouldn't happen.
                        # If it does, returning current pos effectively blocks the drag.
                        pass
                    return BondItem._original_itemChange(self, change, value)
                BondItem.itemChange = bond_item_change_guarded

            # Also force update all existing bonds in the scene
            if mw.scene:
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
    for item in scene.items():
        if sip_isdeleted_safe(item): continue
        cls_name = type(item).__name__
        if cls_name == "AtomItem":
            all_atoms.append(item)
        elif cls_name == "BondItem":
            all_bonds.append(item)
    
    if not all_atoms: return []

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
    molecules, _, _ = find_molecules(scene)
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

    # 2. Find molecules in scene and identify item->oid mapping
    molecules, all_atoms, all_bonds = find_molecules(mw.scene)
    item_to_oid = {}
    if hasattr(mw.scene, 'data'):
        for oid, atom_data in mw.scene.data.atoms.items():
            item = atom_data.get('item')
            if item: item_to_oid[item] = oid

    # 3. Targeted Sync based on selection
    selected_items = mw.scene.selectedItems()
    target_mol_indices = []
    
    if selected_items:
        for i, mol_atoms in enumerate(molecules):
            # Check if any atom of this molecule is selected
            if any(a.isSelected() for a in mol_atoms):
                target_mol_indices.append(i)
                continue
            # Check if any bond of this molecule is selected
            for bond in all_bonds:
                if bond.isSelected() and (bond.atom1 in mol_atoms or bond.atom2 in mol_atoms):
                    target_mol_indices.append(i)
                    break
    
    # 3. Sync target molecules
    for i, mol_atoms in enumerate(molecules):
        if selected_items and i not in target_mol_indices:
            continue
            
        mapped_data = [] # List of (atom_item, px, py, pz)
        for a in mol_atoms:
            oid = item_to_oid.get(a)
            if oid in proj_coords:
                mapped_data.append((a, *proj_coords[oid]))
        
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

    # 4. Update ALL bonds (ensure position and ZValue) and Z-ranges
    for bond in all_bonds:
        # Fix: Ensure bonds are NOT movable
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
                    item.z_3d = 0.0
                    item.setZValue(item.z_3d)
        
        # Refresh ranges regardless
        update_molecule_z_ranges(self.mw.scene)

    def eventFilter(self, obj, event):
        if not self.active: return False
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                # Check for item at click position
                item = self.mw.view_2d.itemAt(event.position().toPoint())
                # if item is None:
                #     # Clicking on empty space -> Switch to select mode
                #     if self.rotate_act:
                #         self.rotate_act.setChecked(False)
                #     return False
                
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
                
        # 7. Sync back to RDKit conformer for persistence and undo/redo
        mol = self.mw.current_mol
        if mol and mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            # Build mapping aid -> idx
            aid_to_idx = {}
            for i in range(mol.GetNumAtoms()):
                atom_rd = mol.GetAtomWithIdx(i)
                if atom_rd.HasProp("_original_atom_id"):
                    try: aid_to_idx[atom_rd.GetIntProp("_original_atom_id")] = i
                    except: pass
            
            for atom_item in self.target_atoms:
                aid = atom_item.atom_id
                if aid in aid_to_idx:
                    idx = aid_to_idx[aid]
                    pos = atom_item.pos()
                    # Map back from screen to RDKit (divide by scale, flip Y)
                    conf.SetAtomPosition(idx, Point3D(pos.x() / _3d_scale, -pos.y() / _3d_scale, atom_item.z_3d / _3d_scale))

        update_molecule_z_ranges(self.mw.scene)
        self.mw.scene.update()


def sip_isdeleted_safe(obj):
    try: return sip.isdeleted(obj)
    except: return False
