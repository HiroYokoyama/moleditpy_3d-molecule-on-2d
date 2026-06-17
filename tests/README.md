# 3D Molecule on 2D Plugin Tests

This directory contains standalone and integration tests specific to the `moleditpy_3d-molecule-on-2d` plugin.

## Available Tests

### `test_plugin_integration.py`
Verifies the plugin's compatibility contract against the host application's `PluginContext` API.

It operates in two modes:
1. **Stub Mode**: Always runs (local and CI) without requiring GUI, Qt, or RDKit libraries. Stubs check registrations, handlers, namespacing, and metadata format.
2. **Real-Context Mode**: Runs when the main app (`python_molecular_editor`) is detected or provided via the `CI_MAIN_APP_SRC` environment variable. It validates that the actual `PluginContext` API is fully matched.

### `test_export.py`
Verifies that the `3d_molecule_on_2d` plugin correctly exports 3D molecular data to a `.mol` file format, explicitly bypassing bugs in the core `io_logic.py` export routine without actually modifying the main application.

Specifically, it tests:
1. **Header Dimensionality Formatting**: Checks that the `'3D'` dimensionality flag is perfectly aligned to columns 21-22 in the MOL V2000 specification.
2. **Data Model Injection**: Checks that `z_3d` values extracted from UI representations (`item.pos()` and `item.z_3d`) are correctly scaled and mapped into the RDKit conformer.
3. **Non-Zero Coordinate Persistence**: Validates that the final generated MOL block does not incorrectly wipe Z-coordinates to `0.0000`.

## How to Run

To execute all tests (including integration tests), run:

```cmd
cd e:\Research\Calculation\moleditpy\DEV_MAIN\moleditpy_3d-molecule-on-2d
python -m pytest tests/ -v
```
