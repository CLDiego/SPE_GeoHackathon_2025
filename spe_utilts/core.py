from pathlib import Path

def find_project_root() -> Path:
    current_path = Path(__file__).resolve()
    while current_path != current_path.root:
        if (current_path / 'utils').exists():  # Check if 'utils' directory exists
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("Project root not found")