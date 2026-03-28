#!/usr/bin/env bash
# =============================================================================
# create_project_structure.sh
# 
# Creates the initial directory and file structure for the openSMT project.
# Run from the directory where you want the openSMT folder to be created.
#
# Usage: ./create_project_structure.sh
# =============================================================================
set -e  # Exit on any error
PROJECT_NAME="openSMT"
echo "Creating project structure for ${PROJECT_NAME}..."
# -----------------------------------------------------------------------------
# Create directory structure
# -----------------------------------------------------------------------------
mkdir -p "${PROJECT_NAME}/config"
mkdir -p "${PROJECT_NAME}/hardware/gcode"
mkdir -p "${PROJECT_NAME}/hardware/camera"
mkdir -p "${PROJECT_NAME}/gui"
mkdir -p "${PROJECT_NAME}/utils"
# -----------------------------------------------------------------------------
# Create root level files
# -----------------------------------------------------------------------------
touch "${PROJECT_NAME}/main.py"
# -----------------------------------------------------------------------------
# Create config module files
# -----------------------------------------------------------------------------
touch "${PROJECT_NAME}/config/__init__.py"
touch "${PROJECT_NAME}/config/config_manager.py"
touch "${PROJECT_NAME}/config/default_config.json"
# -----------------------------------------------------------------------------
# Create hardware module files
# -----------------------------------------------------------------------------
touch "${PROJECT_NAME}/hardware/__init__.py"
# G-Code submodule
touch "${PROJECT_NAME}/hardware/gcode/__init__.py"
touch "${PROJECT_NAME}/hardware/gcode/gcode_handler.py"
touch "${PROJECT_NAME}/hardware/gcode/gcode_device.py"
# Camera submodule
touch "${PROJECT_NAME}/hardware/camera/__init__.py"
touch "${PROJECT_NAME}/hardware/camera/camera_manager.py"
touch "${PROJECT_NAME}/hardware/camera/camera_device.py"
touch "${PROJECT_NAME}/hardware/camera/vision_pipeline.py"
# -----------------------------------------------------------------------------
# Create GUI module files
# -----------------------------------------------------------------------------
touch "${PROJECT_NAME}/gui/__init__.py"
touch "${PROJECT_NAME}/gui/main_window.py"
touch "${PROJECT_NAME}/gui/camera_window.py"
touch "${PROJECT_NAME}/gui/hardware_config_widget.py"
touch "${PROJECT_NAME}/gui/pipeline_editor.py"
# -----------------------------------------------------------------------------
# Create utils module files
# -----------------------------------------------------------------------------
touch "${PROJECT_NAME}/utils/__init__.py"
touch "${PROJECT_NAME}/utils/file_watcher.py"
# -----------------------------------------------------------------------------
# Create additional project files
# -----------------------------------------------------------------------------
# Requirements file
cat > "${PROJECT_NAME}/requirements.txt" << 'EOF'
# =============================================================================
# openSMT Python Dependencies
# =============================================================================
# Qt 6 GUI framework
PyQt6>=6.5.0
# Serial communication for G-Code devices
pyserial>=3.5
# Computer vision
opencv-python>=4.8.0
# File system monitoring for config hot-reload
watchdog>=3.0.0
# JSON schema validation (optional but recommended)
jsonschema>=4.19.0
EOF
# .gitignore
cat > "${PROJECT_NAME}/.gitignore" << 'EOF'
# =============================================================================
# openSMT .gitignore
# =============================================================================
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class
# Virtual environments
venv/
env/
.venv/
# IDE settings
.idea/
.vscode/
*.swp
*.swo
# OS files
.DS_Store
Thumbs.db
# Distribution / packaging
dist/
build/
*.egg-info/
# Local configuration overrides (user-specific settings)
config/local_config.json
# Log files
*.log
logs/
EOF
# README placeholder
cat > "${PROJECT_NAME}/README.md" << 'EOF'
# openSMT
Open-source software for SMD pick and place machines.
## License
MIT License - See LICENSE file for details.
## Requirements
- Python 3.11+
- Qt 6
- OpenCV
- See `requirements.txt` for full dependencies
## Installation
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
