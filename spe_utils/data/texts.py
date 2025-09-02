"""
Geoscience text datasets for NLP and LLM exercises.

This module provides curated text collections for various geoscience domains
including seismic methods, reservoir properties, well logging, and more.
"""

# Geoscience term examples for tokenization demos
GEOSCIENCE_TERMS = [
    "seismic inversion",
    "reservoir characterization", 
    "hydrocarbon exploration",
    "petrophysical analysis",
    "porosity measurement",
    "permeability analysis"
]

# Simple text examples for tokenization
TOKENIZATION_EXAMPLES = [
    "Seismic inversion is a geophysical technique.",
    "Hydrocarbon exploration uses seismic surveys.",
    "Reservoir characterization involves petrophysical analysis.",
    "What is the porosity and permeability of this formation?"
]

# Simple prompts for text generation
SIMPLE_PROMPTS = [
    "The geology of this region",
    "Oil and gas exploration requires",
    "Seismic waves travel through"
]

# Comprehensive geophysics and petroleum engineering texts
GEOPHYSICS_TEXTS = [
    # Seismic Methods
    "Seismic inversion transforms seismic reflection data into quantitative subsurface rock properties.",
    "P-wave velocity depends on rock density and bulk modulus in elastic media.",
    "S-wave velocity is controlled by shear modulus and density of the formation.",
    "Seismic amplitude variation with offset reveals fluid content and lithology changes.",
    "Pre-stack seismic inversion simultaneously estimates multiple elastic parameters from angle stacks.",
    "Post-stack seismic inversion derives acoustic impedance from normal incidence reflectivity.",
    "Seismic interpretation identifies structural features like faults, folds, and stratigraphic boundaries.",
    "Time-lapse seismic monitoring tracks reservoir changes during production and injection.",
    
    # Reservoir Properties
    "Porosity measures the void space available for fluid storage in reservoir rocks.",
    "Permeability quantifies the ability of fluids to flow through porous rock matrices.",
    "Water saturation represents the fraction of pore space occupied by formation water.",
    "Net-to-gross ratio indicates the proportion of reservoir quality rock in a formation.",
    "Reservoir pressure drives hydrocarbon flow from formation to wellbore during production.",
    "Capillary pressure controls fluid distribution at pore scale in reservoir rocks.",
    "Relative permeability curves describe multiphase flow behavior in porous media.",
    "Formation volume factor accounts for fluid expansion from reservoir to surface conditions.",
    
    # Well Logging
    "Gamma ray logs measure natural radioactivity to identify shale and clay content.",
    "Resistivity logs detect hydrocarbon presence by measuring electrical resistance of formations.",
    "Neutron logs respond to hydrogen content, indicating porosity and fluid types.",
    "Density logs measure bulk density to calculate porosity and identify lithology.",
    "Photoelectric factor from density logs helps distinguish different rock types and minerals.",
    "Spontaneous potential logs indicate permeable zones and formation water resistivity.",
    "Caliper logs measure borehole diameter to identify washouts and tight spots.",
    "Nuclear magnetic resonance logs provide pore size distribution and permeability estimates.",
    
    # Drilling and Completion
    "Drilling mud maintains wellbore stability and carries cuttings to surface.",
    "Casing design protects formations and enables safe drilling to target depths.",
    "Hydraulic fracturing creates artificial fractures to enhance reservoir permeability.",
    "Horizontal drilling maximizes contact with thin reservoir layers.",
    "Perforation creates communication pathways between wellbore and reservoir.",
    "Sand control prevents formation sand production that could damage equipment.",
    "Acidizing dissolves formation damage and enhances near-wellbore permeability.",
    "Wellbore trajectory optimization maximizes reservoir contact while avoiding hazards.",
    
    # Production Engineering
    "Artificial lift systems maintain production when reservoir pressure declines.",
    "Nodal analysis optimizes production system performance from reservoir to separator.",
    "Decline curve analysis predicts future production rates and ultimate recovery.",
    "Enhanced oil recovery techniques mobilize residual oil after primary depletion.",
    "Water flooding maintains reservoir pressure and sweeps oil toward producers.",
    "Gas injection improves oil recovery through miscible or immiscible displacement.",
    "Thermal recovery methods reduce oil viscosity in heavy oil reservoirs.",
    "Production optimization balances rate, pressure, and equipment constraints.",
    
    # Geology and Geochemistry
    "Source rock maturation generates hydrocarbons through thermal decomposition of organic matter.",
    "Migration pathways allow hydrocarbons to move from source to reservoir rocks.",
    "Structural traps accumulate hydrocarbons through folding and faulting processes.",
    "Stratigraphic traps form through depositional and diagenetic rock property changes.",
    "Seal integrity prevents hydrocarbon leakage from reservoir to surface.",
    "Basin modeling predicts hydrocarbon generation, migration, and accumulation timing.",
    "Sequence stratigraphy correlates rock units across regional geological frameworks.",
    "Diagenesis modifies reservoir quality through cementation and dissolution processes.",
    
    # Advanced Technologies
    "Machine learning algorithms identify patterns in seismic and well log data.",
    "Digital twins create virtual reservoir models for production optimization.",
    "Microseismic monitoring tracks fracture growth during stimulation operations.",
    "Fiber optic sensing provides distributed measurements along wellbore length.",
    "Cloud computing enables large-scale reservoir simulation and data analytics.",
    "Automated drilling systems improve efficiency and reduce human error.",
    "Real-time optimization adjusts operations based on continuous data streams.",
    "Carbon capture and storage requires geological characterization for safe sequestration."
]

# Category mapping for the comprehensive texts
GEOPHYSICS_CATEGORIES = [
    "Seismic Methods", "Seismic Methods", "Seismic Methods", "Seismic Methods", 
    "Seismic Methods", "Seismic Methods", "Seismic Methods", "Seismic Methods",
    "Reservoir Properties", "Reservoir Properties", "Reservoir Properties", "Reservoir Properties",
    "Reservoir Properties", "Reservoir Properties", "Reservoir Properties", "Reservoir Properties",
    "Well Logging", "Well Logging", "Well Logging", "Well Logging",
    "Well Logging", "Well Logging", "Well Logging", "Well Logging",
    "Drilling & Completion", "Drilling & Completion", "Drilling & Completion", "Drilling & Completion",
    "Drilling & Completion", "Drilling & Completion", "Drilling & Completion", "Drilling & Completion",
    "Production Engineering", "Production Engineering", "Production Engineering", "Production Engineering",
    "Production Engineering", "Production Engineering", "Production Engineering", "Production Engineering",
    "Geology & Geochemistry", "Geology & Geochemistry", "Geology & Geochemistry", "Geology & Geochemistry",
    "Geology & Geochemistry", "Geology & Geochemistry", "Geology & Geochemistry", "Geology & Geochemistry",
    "Advanced Technologies", "Advanced Technologies", "Advanced Technologies", "Advanced Technologies",
    "Advanced Technologies", "Advanced Technologies", "Advanced Technologies", "Advanced Technologies"
]

def get_texts_by_category(category=None):
    """
    Get geophysics texts filtered by category.
    
    Args:
        category (str, optional): Category to filter by. If None, returns all texts.
        
    Returns:
        list: List of texts for the specified category or all texts.
    """
    if category is None:
        return GEOPHYSICS_TEXTS
    
    return [text for text, cat in zip(GEOPHYSICS_TEXTS, GEOPHYSICS_CATEGORIES) 
            if cat == category]

def get_available_categories():
    """
    Get list of available text categories.
    
    Returns:
        list: Unique category names.
    """
    return list(set(GEOPHYSICS_CATEGORIES))

def get_random_texts(n=5, category=None, seed=42):
    """
    Get random selection of geophysics texts.
    
    Args:
        n (int): Number of texts to return.
        category (str, optional): Category to sample from.
        seed (int): Random seed for reproducibility.
        
    Returns:
        list: Random selection of texts.
    """
    import random
    random.seed(seed)
    
    texts = get_texts_by_category(category)
    return random.sample(texts, min(n, len(texts)))