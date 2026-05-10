# 3D Molecule on 2D Plugin Tests

This directory contains standalone tests specific to the `moleditpy_3d-molecule-on-2d` plugin.

## Available Tests

### `test_export.py`
Verifies that the `3d_molecule_on_2d` plugin correctly exports 3D molecular data to a `.mol` file format, explicitly bypassing bugs in the core `io_logic.py` export routine without actually modifying the main application.

Specifically, it tests:
1. **Header Dimensionality Formatting**: Checks that the `'3D'` dimensionality flag is perfectly aligned to columns 21-22 in the MOL V2000 specification.
2. **Data Model Injection**: Checks that `z_3d` values extracted from UI representations (`item.pos()` and `item.z_3d`) are correctly scaled and mapped into the RDKit conformer.
3. **Non-Zero Coordinate Persistence**: Validates that the final generated MOL block does not incorrectly wipe Z-coordinates to `0.0000`.

## How to Run
To execute the tests, you can run them directly via Python:

```cmd
cd e:\Research\Calculation\moleditpy\DEV_MAIN\moleditpy_3d-molecule-on-2d\tests
python test_export.py
```
