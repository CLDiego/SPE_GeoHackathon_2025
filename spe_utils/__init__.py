"""
Utilities for the University of Manchester Faculty of Science and Engineering SPE GeoHackathon 2025.

This package provides various utility functions and classes for the workshop, including:
- Data handling utilities for datasets and loaders
- Machine learning utilities for training and evaluation
- Plotting utilities for visualizing data and results
- Core functions for project management and exercise validation

The package automatically imports all submodules and their public members.
"""

import importlib
import pkgutil

# Import version
from .__version__ import __version__

def import_all_modules(package_name):
    """Automatically import all modules and their public members."""
    package = importlib.import_module(package_name)
    results = {}
    
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = f"{package_name}.{name}"
        results[name] = importlib.import_module(full_name)
        
        # If it's a package, recurse into it
        if is_pkg:
            results.update(import_all_modules(full_name))
    
    return results

# Automatically import all modules
modules = import_all_modules(__package__)

# Collect all public members (those not starting with _)
__all__ = ['find_project_root']
for module in modules.values():
    if hasattr(module, '__all__'):
        __all__.extend(module.__all__)
    else:
        __all__.extend([name for name in dir(module)
                       if not name.startswith('_')])

# Remove duplicates while preserving order
__all__ = list(dict.fromkeys(__all__))

print('Faculty of Science and Engineering 🔬')
print('\033[95mThe University of Manchester \033[0m')
print(f'Invoking SPE GeoHackathon 2025 utils version: \033[92m{__version__}\033[0m')