"""
Data utilities and datasets for the SPE GeoHackathon 2025.

This module provides:
- Curated geoscience text collections
- Functions to access and filter text data
- Sample datasets for various exercises
"""

from .texts import *

__all__ = [
    'GEOSCIENCE_TERMS',
    'TOKENIZATION_EXAMPLES', 
    'SIMPLE_PROMPTS',
    'GEOPHYSICS_TEXTS',
    'GEOPHYSICS_CATEGORIES',
    'get_texts_by_category',
    'get_available_categories',
    'get_random_texts'
]