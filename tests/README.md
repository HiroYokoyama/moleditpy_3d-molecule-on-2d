# 3D Molecule on 2D - Test Suite Guide

This directory contains the unit and integration tests for the `moleditpy_3d-molecule-on-2d` plugin.

## Test Structure

- **`test_core_logic.py`**: Validates the core calculations and workers of the plugin, including:
  - Color blending math used for visual depth cueing.
  - `LocalCalculationWorker` 3D structures generation (via RDKit embedding, hydrogen operations, and coordinate jitter fallbacks).
  - Rigid body coordinates rotation math via `RotateToolHandler`.
- **`test_plugin_integration.py`**: Checks that the plugin correctly integrates with the host application's `PluginContext` (verifying action namespacing, setting persistence, and save/load state handlers).
- **`test_export.py`**: Ensures that the monkey-patched molecular MOL block generation correctly formats 3D coordinate grids with proper dimensionality.

---

## Running Tests

To run the test suite, navigate to the plugin root directory and execute `pytest`. 

### Important: Environment Configuration under Windows
Because `pytest-qt` loads PySide6 by default, it can conflict with PyQt6 imports in the plugin, resulting in `DLL load failed` or symbol collisions. To prevent this, **always** specify the `PYTEST_QT_API=pyqt6` environment variable:

#### PowerShell
```powershell
$env:MOLEDITPY_HEADLESS="1"
$env:QT_QPA_PLATFORM="offscreen"
$env:PYTEST_QT_API="pyqt6"
pytest -v
```

#### CMD
```cmd
set MOLEDITPY_HEADLESS=1
set QT_QPA_PLATFORM=offscreen
set PYTEST_QT_API=pyqt6
pytest -v
```

#### Bash
```bash
MOLEDITPY_HEADLESS=1 QT_QPA_PLATFORM=offscreen PYTEST_QT_API=pyqt6 pytest -v
```
