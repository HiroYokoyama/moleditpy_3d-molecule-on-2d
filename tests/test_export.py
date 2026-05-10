import sys
import os
import importlib

# Add core moleditpy source to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python_molecular_editor/moleditpy/src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from moleditpy.core.molecular_data import MolecularData

def test_3d_mol_export():
    # Dynamically import the plugin
    plugin = importlib.import_module('3d_molecule_on_2d')

    # Mock an Atom Item with visual UI coordinates and a custom z_3d property
    class MockItem:
        def __init__(self, x, y, z, atom_id):
            self.x = x
            self.y = y
            self.z_3d = z
            self.atom_id = atom_id
        def pos(self):
            class Pos:
                def __init__(self, xx, yy):
                    self._x = xx
                    self._y = yy
                def x(self): return self._x
                def y(self): return self._y
            return Pos(self.x, self.y)

    # 1. Patch the export logic so the header is properly bypassed
    plugin.patch_export_logic(True)

    # 2. Build mock molecular data
    data = MolecularData()
    id1 = data.add_atom('C', (0, 0))
    id2 = data.add_atom('C', (10, 0))
    data.add_bond(id1, id2, 1)

    # 3. Simulate UI items with 3D depth attached
    data.atoms[id1]['item'] = MockItem(0, 0, 10.0, id1)
    data.atoms[id2]['item'] = MockItem(10, 0, 20.0, id2)

    # 4. Generate the MOL Block
    mol_block = data.to_mol_block()

    print("--- Test: 3D Mol on 2D Export ---")
    print("MOL Block Output:\n")
    print(mol_block)

    # 5. Verify the Dimensionality flag is strictly at columns 21-22
    lines = mol_block.split('\n')
    assert len(lines) > 1, "MOL block has too few lines"
    dim_flag = lines[1][20:22]
    assert dim_flag == '3D', f"Expected '3D' at columns 21-22, but got '{dim_flag}'"

    # 6. Verify Z Coordinates are not zero
    assert '0.2000' in mol_block, "Z-coordinate 0.2000 missing from MOL block"
    assert '0.4000' in mol_block, "Z-coordinate 0.4000 missing from MOL block"

    print("\n--- ALL TESTS PASSED: 3D Depth Coordinates Exported Correctly ---")

if __name__ == "__main__":
    test_3d_mol_export()
