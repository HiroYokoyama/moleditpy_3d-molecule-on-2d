import numpy as np
try:
    from PyQt6 import sip
except ImportError:
    import sip
from PyQt6.QtWidgets import QMenu, QToolBar, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGraphicsScene
from PyQt6.QtCore import Qt, QPointF, QEvent, QObject, QTimer
from PyQt6.QtGui import QColor, QPen, QIcon, QAction, QActionGroup, QPainter, QBrush

# Metadata
PLUGIN_NAME = "3D Molecule on 2D"
PLUGIN_VERSION = "0.0.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = "Integrated 3D depth cues and rotation."

# Global state
_show_depth_cues = False
_rotate_tool_handler = None
_depth_cue_strength = 0.8  # 0.0 to 1.0

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
    
    # --- Action Callbacks ---
    
    def on_proj_toggled(checked):
        toggle_monkey_patches(checked)
        mw.scene.update()
        mw.statusBar().showMessage(f"Depth Cues {'Enabled' if checked else 'Disabled'}")

    def on_rotate_toggled(checked):
        _rotate_tool_handler.set_active(checked)

    def on_cleanup_triggered():
        global _show_depth_cues
        mol = mw.current_mol
        if mol and mol.GetNumConformers() > 0:
            mw.statusBar().showMessage("Applying existing 3D coordinates...")
            sync_to_3d_layout(mw, mol)
        else:
            mw.statusBar().showMessage("Generating 3D coordinates...")
            mw.trigger_conversion()
            QTimer.singleShot(1000, lambda: sync_to_3d_layout(mw, mw.current_mol))
        
        # Ensure projection is on
        tb = getattr(mw, 'plugin_toolbar', None)
        if tb:
            for act in tb.actions():
                if act.text() == "3D Proj View":
                    act.setChecked(True)
                    # Note: setChecked fires toggled signal
                    break

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
        ok_btn = QPushButton("OK"); ok_btn.clicked.connect(dlg.accept); layout.addWidget(ok_btn)
        dlg.exec()

    # --- Standard Registration ---
    
    context.add_toolbar_action(lambda: None, "3D Proj View", tooltip="Toggle depth cues")
    context.add_toolbar_action(lambda: None, "Rotate 3D", tooltip="Rotate molecule in 3D")
    context.add_toolbar_action(on_cleanup_triggered, "Clean Up 3D", tooltip="Sync 2D layout to 3D")
    context.add_toolbar_action(show_settings_dialog, "⚙", tooltip="Settings")
    
    # --- Post-Processing (Refine toolbar UI) ---
    
    def configure_actions():
        tb = getattr(mw, 'plugin_toolbar', None)
        if not tb: return
        
        actions = list(tb.actions())
        proj_act = next((a for a in actions if a.text() == "3D Proj View"), None)
        rotate_act = next((a for a in actions if a.text() == "Rotate 3D"), None)
        cleanup_act = next((a for a in actions if a.text() == "Clean Up 3D"), None)
        settings_act = next((a for a in actions if a.text() == "⚙"), None)
        
        # Set Checkable and connect toggled
        if proj_act:
            proj_act.setCheckable(True)
            proj_act.toggled.connect(on_proj_toggled)
        if rotate_act:
            rotate_act.setCheckable(True)
            rotate_act.toggled.connect(on_rotate_toggled)
        
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

    # Persistence
    def save_state():
        state = {
            "show_depth_cues": _show_depth_cues,
            "depth_cue_strength": _depth_cue_strength
        }
        if mw.scene and hasattr(mw.scene, 'data'):
            z_data = {}
            for aid, atom_data in mw.scene.data.atoms.items():
                item = atom_data.get('item')
                if item and not sip_isdeleted_safe(item) and hasattr(item, "z_3d"):
                    z_data[str(aid)] = item.z_3d
            state["z_data"] = z_data
        return state

    def load_state(data):
        global _show_depth_cues, _depth_cue_strength
        if not data: return
        _show_depth_cues = data.get("show_depth_cues", False)
        _depth_cue_strength = data.get("depth_cue_strength", 0.8)
        
        def restore_checks():
            tb = getattr(mw, 'plugin_toolbar', None)
            if tb:
                for act in tb.actions():
                    if act.text() == "3D Proj View":
                        act.setChecked(_show_depth_cues)
                        toggle_monkey_patches(_show_depth_cues)
                        break
        QTimer.singleShot(100, restore_checks)

        if "z_data" in data and mw.scene and hasattr(mw.scene, 'data'):
            z_map = data["z_data"]
            for aid_str, z in z_map.items():
                aid = int(aid_str)
                atom_data = mw.scene.data.atoms.get(aid)
                if atom_data and 'item' in atom_data:
                    item = atom_data['item']
                    if item and not sip_isdeleted_safe(item):
                        item.z_3d = z
            mw.scene.update()

    context.register_save_handler(save_state)
    context.register_load_handler(load_state)

    _rotate_tool_handler = RotateToolHandler(mw)

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
            _original_bond_paint = BondItem.paint
            BondItem.paint = patched_bond_paint
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
        z_min, z_max = get_scene_z_range(self.scene())
        z_range = z_max - z_min if z_max != z_min else 1.0
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
        z_min, z_max = get_scene_z_range(self.scene())
        z_range = z_max - z_min if z_max != z_min else 1.0
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

# --- Logic ---

def sync_to_3d_layout(mw, mol):
    if not mol or mol.GetNumConformers() == 0: return
    conf = mol.GetConformer()
    scale = 30.0
    
    scene_positions = []
    for atom_data in mw.scene.data.atoms.values():
        item = atom_data.get('item')
        if item and not sip_isdeleted_safe(item):
            scene_positions.append(item.pos())
    if not scene_positions: return
    avg_x = sum(p.x() for p in scene_positions) / len(scene_positions)
    avg_y = sum(p.y() for p in scene_positions) / len(scene_positions)
    
    proj_coords = {}
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        p = conf.GetAtomPosition(i)
        oid = None
        if atom.HasProp("_original_atom_id"): oid = int(atom.GetProp("_original_atom_id"))
        elif atom.HasProp("original_id"): oid = int(atom.GetProp("original_id"))
        if oid is not None:
            proj_coords[oid] = (p.x * scale, -p.y * scale, p.z * scale)

    if proj_coords:
        p_avg_x = sum(c[0] for c in proj_coords.values()) / len(proj_coords)
        p_avg_y = sum(c[1] for c in proj_coords.values()) / len(proj_coords)
        dx, dy = avg_x - p_avg_x, avg_y - p_avg_y
        for oid, (px, py, pz) in proj_coords.items():
            atom_data = mw.scene.data.atoms.get(oid)
            if atom_data and 'item' in atom_data:
                item = atom_data['item']
                if item and not sip_isdeleted_safe(item):
                    item.setPos(QPointF(px + dx, py + dy))
                    item.z_3d = pz
                    # White is back: Higher Z is closer
                    item.setZValue(pz)
    for item in mw.scene.items():
        if type(item).__name__ == "BondItem" and not sip_isdeleted_safe(item):
            if hasattr(item, "update_position"): item.update_position()
            # Depth sort bonds in sync logic too
            if hasattr(item.atom1, "z_3d") and hasattr(item.atom2, "z_3d"):
                item.setZValue((item.atom1.z_3d + item.atom2.z_3d) / 2.0)
    mw.scene.update()

class RotateToolHandler(QObject):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.active = False
        self.is_dragging = False
        self.mw.view_2d.viewport().installEventFilter(self)
        
    def set_active(self, state):
        self.active = state
        if state:
            self.mw.set_mode("plugin_rotate_3d")
            self.ensure_z_coords()
        else:
            self.mw.scene.mode = "select"
            if hasattr(self.mw, "activate_select_mode"): self.mw.activate_select_mode()

    def ensure_z_coords(self):
        if not self.mw.scene: return
        atoms_needing_z = []
        has_nonzero_z = False
        for item in self.mw.scene.items():
            if type(item).__name__ == "AtomItem":
                z = getattr(item, "z_3d", 0.0)
                if abs(z) > 1e-4: has_nonzero_z = True
                atoms_needing_z.append(item)
        if not has_nonzero_z:
            mol = self.mw.current_mol
            if mol and mol.GetNumConformers() > 0:
                conf = mol.GetConformer()
                scale = 30.0
                for aid, atom_data in self.mw.scene.data.atoms.items():
                    item = atom_data.get('item')
                    if item and not sip_isdeleted_safe(item):
                        if aid < mol.GetNumAtoms():
                            item.z_3d = conf.GetAtomPosition(aid).z * scale
                        else: item.z_3d = 0.0
            else:
                for item in atoms_needing_z: item.z_3d = 0.0

    def eventFilter(self, obj, event):
        if not self.active: return False
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                self.is_dragging = True
                self.last_pos = event.position().toPoint()
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
            return True
        return False

    def rotate_molecule(self, dx, dy):
        if not self.mw.scene or not hasattr(self.mw.scene, 'data'):
            return

        # 1. Collect all atoms and bonds in the scene to build adjacency
        all_atoms = []
        all_bonds = []
        for item in self.mw.scene.items():
            if sip_isdeleted_safe(item): continue
            cls_name = type(item).__name__
            if cls_name == "AtomItem":
                all_atoms.append(item)
            elif cls_name == "BondItem":
                all_bonds.append(item)
        
        if not all_atoms: return

        # 2. Build adjacency list for connected component analysis
        adj = {atom: [] for atom in all_atoms}
        for bond in all_bonds:
            if bond.atom1 in adj and bond.atom2 in adj:
                adj[bond.atom1].append(bond.atom2)
                adj[bond.atom2].append(bond.atom1)

        # 3. Find connected components (molecules)
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

        # 4. Filter molecules: keep only those with at least one selected item
        # If nothing is selected, we keep ALL molecules.
        any_selected = any(a.isSelected() for a in all_atoms) or any(b.isSelected() for b in all_bonds)
        
        target_molecules = []
        if any_selected:
            # Check if molecule contains any selected atom or if any of its internal bonds are selected
            for mol_atoms in molecules:
                mol_set = set(mol_atoms)
                has_selection = any(a.isSelected() for a in mol_atoms)
                if not has_selection:
                    # Check bonds connected to these atoms
                    for bond in all_bonds:
                        if bond.isSelected() and (bond.atom1 in mol_set or bond.atom2 in mol_set):
                            has_selection = True
                            break
                if has_selection:
                    target_molecules.append(mol_atoms)
        else:
            target_molecules = molecules

        if not target_molecules: return

        # 5. Rotate each target molecule around its own COG
        Rx = np.array([[1, 0, 0], [0, np.cos(dy), -np.sin(dy)], [0, np.sin(dy), np.cos(dy)]])
        Ry = np.array([[np.cos(dx), 0, np.sin(dx)], [0, 1, 0], [-np.sin(dx), 0, np.cos(dx)]])
        R = Rx @ Ry

        for mol_atoms in target_molecules:
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
                
        self.mw.scene.update()


def sip_isdeleted_safe(obj):
    try: return sip.isdeleted(obj)
    except: return False
