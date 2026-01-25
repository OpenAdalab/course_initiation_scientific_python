# Day 3 - Afternoon Session
## Final Integration Project: Discovering a Resonance in Dimuon Events

**Duration:** 3.5 hours
**Format:** Team project (pairs)

---

## Project Overview (15 min)

### Scenario

You have simulated data from a particle detector recording muon pairs from proton-proton collisions. Your goal is to **identify a resonance peak** (like J/ψ or Z boson) in the invariant mass spectrum.

This project brings together everything learned in the course:

- Data loading and exploration (Day 1)
- NumPy/Pandas operations (Days 1-2)
- Visualization (Day 2)
- Functions and classes (Day 2)
- Error handling and testing (Day 3)

### Provided Materials

📓 **Open the project notebook:** [day3_afternoon_exercises.ipynb](https://github.com/OpenAdalab/course_initiation_scientific_python/blob/main/notebooks/day3_afternoon_exercises.ipynb)

The notebook includes:

- **CSV data file** with event data (event_id, muon kinematics, weights)
- **Monte Carlo simulation** file (same format, includes true_mass column)
- **Detector efficiency map** (NumPy array)
- **Configuration file** (cuts, binning, etc.)

### Project Structure

```
project/
├── data/
│   ├── events_data.csv        # Real data events
│   ├── events_mc.csv          # Monte Carlo simulation
│   └── efficiency_map.npy     # Detector efficiency
├── config.json                # Analysis configuration
├── analysis.py                # Your analysis code
└── test_analysis.py           # Your tests
```

---

## Part 1: Data Loading and Exploration (45 min)

### Beginner Level

1. Load data into Pandas DataFrame
2. Inspect data: check for NaN values, show basic statistics
3. Create simple plots: pT distributions, η-φ distributions
4. Calculate invariant mass for each event

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/events_data.csv')

# Basic inspection
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(f"Missing values:\n{df.isnull().sum()}")

# Simple invariant mass calculation (simplified formula)
# M² = 2 * pT1 * pT2 * (cosh(η1-η2) - cos(φ1-φ2))
def calculate_mass_simple(row):
    deta = row['mu1_eta'] - row['mu2_eta']
    dphi = row['mu1_phi'] - row['mu2_phi']
    m2 = 2 * row['mu1_pt'] * row['mu2_pt'] * (np.cosh(deta) - np.cos(dphi))
    return np.sqrt(m2)

df['mass'] = df.apply(calculate_mass_simple, axis=1)
```

### Advanced Level

1. Load multiple files and concatenate efficiently
2. Implement lazy loading for large datasets
3. Create comprehensive data quality checks
4. Calculate invariant mass with full relativistic kinematics
5. Generate correlation plots between variables

```python
import pandas as pd
import numpy as np
from pathlib import Path

def load_all_data(data_dir, pattern='*.csv'):
    """Load and concatenate all CSV files matching pattern."""
    data_path = Path(data_dir)
    all_files = list(data_path.glob(pattern))

    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        df['source_file'] = f.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def calculate_mass_relativistic(mu1_pt, mu1_eta, mu1_phi,
                                 mu2_pt, mu2_eta, mu2_phi,
                                 mu_mass=0.105):
    """Calculate invariant mass with full 4-momentum."""
    # Convert to 4-momentum components
    def get_4momentum(pt, eta, phi, mass):
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        E = np.sqrt(pt**2 * np.cosh(eta)**2 + mass**2)
        return px, py, pz, E

    px1, py1, pz1, E1 = get_4momentum(mu1_pt, mu1_eta, mu1_phi, mu_mass)
    px2, py2, pz2, E2 = get_4momentum(mu2_pt, mu2_eta, mu2_phi, mu_mass)

    # Invariant mass
    E_tot = E1 + E2
    px_tot = px1 + px2
    py_tot = py1 + py2
    pz_tot = pz1 + pz2

    m2 = E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2
    return np.sqrt(np.maximum(m2, 0))

def data_quality_report(df):
    """Generate comprehensive data quality report."""
    report = {
        'n_events': len(df),
        'n_missing': df.isnull().sum().to_dict(),
        'pt_range': (df[['mu1_pt', 'mu2_pt']].min().min(),
                     df[['mu1_pt', 'mu2_pt']].max().max()),
        'eta_range': (df[['mu1_eta', 'mu2_eta']].min().min(),
                      df[['mu1_eta', 'mu2_eta']].max().max()),
    }
    return report
```

---

## Part 2: Event Selection and Cuts (45 min)

### Beginner Level

1. Apply basic cuts: pT > 20 GeV, |η| < 2.4
2. Use boolean indexing to filter events
3. Count events before/after cuts
4. Visualize effect of cuts on distributions

```python
# Define cuts
pt_min = 20.0  # GeV
eta_max = 2.4

# Apply cuts
mask_pt = (df['mu1_pt'] > pt_min) & (df['mu2_pt'] > pt_min)
mask_eta = (np.abs(df['mu1_eta']) < eta_max) & (np.abs(df['mu2_eta']) < eta_max)

# Combined selection
mask_all = mask_pt & mask_eta

print(f"Events before cuts: {len(df)}")
print(f"Events after pT cut: {mask_pt.sum()}")
print(f"Events after η cut: {mask_eta.sum()}")
print(f"Events after all cuts: {mask_all.sum()}")

# Apply selection
df_selected = df[mask_all].copy()
```

### Advanced Level

1. Implement configurable cut system reading from config file
2. Apply detector efficiency corrections using 2D maps
3. Optimize cut values by maximizing significance S/√B
4. Create cut-flow table
5. Compare data and Monte Carlo after cuts

```python
import json
import numpy as np

def load_config(config_path):
    """Load analysis configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def apply_configurable_cuts(df, config):
    """Apply cuts from configuration file."""
    mask = pd.Series(True, index=df.index)

    for cut_name, cut_def in config['cuts'].items():
        variable = cut_def['variable']
        if cut_def['type'] == 'min':
            mask &= df[variable] > cut_def['value']
        elif cut_def['type'] == 'max':
            mask &= df[variable] < cut_def['value']
        elif cut_def['type'] == 'abs_max':
            mask &= np.abs(df[variable]) < cut_def['value']

    return mask

def apply_efficiency_correction(df, efficiency_map, eta_bins, pt_bins):
    """Apply 2D efficiency correction."""
    # Find bin indices
    eta_idx = np.digitize(df['mu1_eta'], eta_bins) - 1
    pt_idx = np.digitize(df['mu1_pt'], pt_bins) - 1

    # Clip to valid range
    eta_idx = np.clip(eta_idx, 0, len(eta_bins) - 2)
    pt_idx = np.clip(pt_idx, 0, len(pt_bins) - 2)

    # Look up efficiency
    efficiency = efficiency_map[eta_idx, pt_idx]

    # Apply correction (weight by 1/efficiency)
    df['weight_corrected'] = df['weight'] / efficiency

    return df

def create_cutflow_table(df, cuts_sequence):
    """Create cut-flow table showing events passing each cut."""
    cutflow = [('Initial', len(df))]

    current_mask = pd.Series(True, index=df.index)
    for cut_name, cut_mask in cuts_sequence:
        current_mask &= cut_mask
        cutflow.append((cut_name, current_mask.sum()))

    # Convert to DataFrame
    cutflow_df = pd.DataFrame(cutflow, columns=['Cut', 'Events'])
    cutflow_df['Efficiency'] = cutflow_df['Events'] / cutflow_df['Events'].iloc[0]
    cutflow_df['Relative Eff'] = cutflow_df['Events'] / cutflow_df['Events'].shift(1)

    return cutflow_df
```

---

## Part 3: Functions and Classes (45 min)

### Beginner Level

Create functions:

```python
def calculate_invariant_mass(pt1, eta1, phi1, pt2, eta2, phi2):
    """
    Calculate invariant mass of a muon pair.

    Parameters:
    -----------
    pt1, eta1, phi1 : float
        Kinematics of first muon
    pt2, eta2, phi2 : float
        Kinematics of second muon

    Returns:
    --------
    float : Invariant mass in GeV
    """
    deta = eta1 - eta2
    dphi = phi1 - phi2
    m2 = 2 * pt1 * pt2 * (np.cosh(deta) - np.cos(dphi))
    return np.sqrt(m2)

def apply_cuts(df, pt_min=20.0, eta_max=2.4):
    """
    Apply kinematic cuts to DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input data with mu1_pt, mu2_pt, mu1_eta, mu2_eta columns
    pt_min : float
        Minimum pT cut (GeV)
    eta_max : float
        Maximum |η| cut

    Returns:
    --------
    pd.DataFrame : Filtered data
    """
    mask = (
        (df['mu1_pt'] > pt_min) &
        (df['mu2_pt'] > pt_min) &
        (np.abs(df['mu1_eta']) < eta_max) &
        (np.abs(df['mu2_eta']) < eta_max)
    )
    return df[mask].copy()

def plot_mass_spectrum(masses, bins=50, range=(60, 120), label='Data'):
    """
    Plot invariant mass spectrum.

    Parameters:
    -----------
    masses : array-like
        Invariant mass values
    bins : int
        Number of histogram bins
    range : tuple
        (min, max) for histogram range
    label : str
        Legend label
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    counts, bin_edges, _ = ax.hist(masses, bins=bins, range=range,
                                    histtype='step', linewidth=2, label=label)

    # Add error bars
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    errors = np.sqrt(counts)
    ax.errorbar(bin_centers, counts, yerr=errors, fmt='none',
                capsize=2, color='black')

    ax.set_xlabel(r'$m_{\mu\mu}$ (GeV/c²)', fontsize=12)
    ax.set_ylabel('Events', fontsize=12)
    ax.set_title('Dimuon Invariant Mass Spectrum', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax
```

Create class:

```python
class MuonPair:
    """A pair of muons from a collision event."""

    def __init__(self, pt1, eta1, phi1, pt2, eta2, phi2):
        """Initialize muon pair with kinematics."""
        self.mu1 = {'pt': pt1, 'eta': eta1, 'phi': phi1}
        self.mu2 = {'pt': pt2, 'eta': eta2, 'phi': phi2}

    def invariant_mass(self):
        """Calculate invariant mass of the pair."""
        deta = self.mu1['eta'] - self.mu2['eta']
        dphi = self.mu1['phi'] - self.mu2['phi']
        m2 = 2 * self.mu1['pt'] * self.mu2['pt'] * (np.cosh(deta) - np.cos(dphi))
        return np.sqrt(m2)

    def passes_cuts(self, pt_min=20.0, eta_max=2.4):
        """Check if pair passes kinematic cuts."""
        pt_ok = (self.mu1['pt'] > pt_min) and (self.mu2['pt'] > pt_min)
        eta_ok = (abs(self.mu1['eta']) < eta_max) and (abs(self.mu2['eta']) < eta_max)
        return pt_ok and eta_ok
```

### Advanced Level

Create function library with error handling:

```python
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV with validation.

    Parameters:
    -----------
    filepath : str
        Path to CSV file

    Returns:
    --------
    pd.DataFrame : Validated data

    Raises:
    -------
    FileNotFoundError : if file doesn't exist
    ValueError : if required columns missing or data invalid
    """
    logger.info(f"Loading data from {filepath}")

    df = pd.read_csv(filepath)

    required_columns = ['event_id', 'mu1_pt', 'mu1_eta', 'mu1_phi',
                        'mu2_pt', 'mu2_eta', 'mu2_phi']

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate physics constraints
    if (df[['mu1_pt', 'mu2_pt']] < 0).any().any():
        raise ValueError("Negative pT values found")

    logger.info(f"Loaded {len(df)} valid events")
    return df

def calculate_significance(signal: float, background: float) -> float:
    """
    Calculate statistical significance.

    Parameters:
    -----------
    signal : float
        Number of signal events
    background : float
        Number of background events

    Returns:
    --------
    float : Significance S/√B (or S/√(S+B) for low statistics)
    """
    if background <= 0:
        logger.warning("Zero or negative background, using S/√(S+B)")
        return signal / np.sqrt(signal + background) if (signal + background) > 0 else 0

    return signal / np.sqrt(background)
```

Create class hierarchy:

```python
import numpy as np
from abc import ABC, abstractmethod

class Particle(ABC):
    """Abstract base class for particles."""

    def __init__(self, pt: float, eta: float, phi: float):
        self.pt = pt
        self.eta = eta
        self.phi = phi

    @property
    @abstractmethod
    def mass(self) -> float:
        """Rest mass in GeV."""
        pass

    @property
    def px(self) -> float:
        return self.pt * np.cos(self.phi)

    @property
    def py(self) -> float:
        return self.pt * np.sin(self.phi)

    @property
    def pz(self) -> float:
        return self.pt * np.sinh(self.eta)

    @property
    def energy(self) -> float:
        p2 = self.pt**2 * np.cosh(self.eta)**2
        return np.sqrt(p2 + self.mass**2)

    def __repr__(self):
        return f"{self.__class__.__name__}(pt={self.pt:.2f}, eta={self.eta:.2f})"


class Muon(Particle):
    """Muon particle."""

    MASS = 0.105  # GeV

    def __init__(self, pt: float, eta: float, phi: float, charge: int = -1):
        super().__init__(pt, eta, phi)
        self.charge = charge

    @property
    def mass(self) -> float:
        return self.MASS


class DimuonEvent:
    """Event containing a muon pair."""

    def __init__(self, event_id: int, mu1: Muon, mu2: Muon):
        self.event_id = event_id
        self.mu1 = mu1
        self.mu2 = mu2

    @property
    def invariant_mass(self) -> float:
        """Calculate invariant mass from 4-momenta."""
        E = self.mu1.energy + self.mu2.energy
        px = self.mu1.px + self.mu2.px
        py = self.mu1.py + self.mu2.py
        pz = self.mu1.pz + self.mu2.pz

        m2 = E**2 - px**2 - py**2 - pz**2
        return np.sqrt(max(m2, 0))

    def passes_selection(self, pt_min: float = 20.0, eta_max: float = 2.4) -> bool:
        """Check if event passes selection cuts."""
        pt_ok = (self.mu1.pt > pt_min) and (self.mu2.pt > pt_min)
        eta_ok = (abs(self.mu1.eta) < eta_max) and (abs(self.mu2.eta) < eta_max)
        return pt_ok and eta_ok


class AnalysisPipeline:
    """Orchestrates the full analysis workflow."""

    def __init__(self, config: Dict):
        self.config = config
        self.steps = []
        self.results = {}

    def add_step(self, name: str, function):
        """Add an analysis step."""
        self.steps.append({'name': name, 'function': function})

    def run(self, data):
        """Execute all analysis steps."""
        result = data
        for step in self.steps:
            logger.info(f"Running: {step['name']}")
            result = step['function'](result)
            self.results[step['name']] = result
        return result
```

---

## Part 4: Error Handling and Validation (30 min)

### Beginner Level

1. Handle missing data files gracefully
2. Validate that invariant masses are positive
3. Check array dimensions match
4. Add try/except around file operations

```python
def safe_load_data(filepath):
    """Load data with error handling."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {len(df)} events")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{filepath}' is empty")
        return None

def validate_masses(masses):
    """Check that all masses are positive."""
    masses = np.asarray(masses)

    if np.any(masses < 0):
        n_negative = np.sum(masses < 0)
        print(f"Warning: {n_negative} negative mass values found")
        # Replace with NaN
        masses = np.where(masses < 0, np.nan, masses)

    return masses
```

### Advanced Level

1. Implement comprehensive input validation
2. Create custom exceptions
3. Add logging at different levels
4. Validate physics constraints

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dimuon_analysis')

class InvalidEventError(Exception):
    """Raised when event data is invalid."""
    pass

class EfficiencyMapError(Exception):
    """Raised when efficiency map is invalid."""
    pass

def validate_event_data(df):
    """Comprehensive data validation."""
    logger.info("Validating event data...")

    # Check for required columns
    required = ['mu1_pt', 'mu1_eta', 'mu1_phi', 'mu2_pt', 'mu2_eta', 'mu2_phi']
    missing = set(required) - set(df.columns)
    if missing:
        raise InvalidEventError(f"Missing columns: {missing}")

    # Check for NaN values
    n_nan = df[required].isnull().sum().sum()
    if n_nan > 0:
        logger.warning(f"Found {n_nan} NaN values")

    # Physics validation
    if (df[['mu1_pt', 'mu2_pt']] < 0).any().any():
        raise InvalidEventError("Negative pT values")

    if (np.abs(df[['mu1_eta', 'mu2_eta']]) > 10).any().any():
        raise InvalidEventError("Unphysical eta values (|η| > 10)")

    logger.info("Data validation passed")
    return True
```

---

## Part 5: Testing (30 min)

### Beginner Level

Write tests using pytest:

```python
# test_analysis.py
import pytest
import numpy as np
from analysis import calculate_invariant_mass, apply_cuts, MuonPair

def test_invariant_mass_known_value():
    """Test mass calculation with Z boson kinematics."""
    # Two back-to-back muons at rest in CM frame
    mass = calculate_invariant_mass(
        pt1=45.6, eta1=0, phi1=0,
        pt2=45.6, eta2=0, phi2=np.pi
    )
    assert mass == pytest.approx(91.2, rel=0.01)

def test_invariant_mass_positive():
    """Mass should always be positive."""
    mass = calculate_invariant_mass(30, 0.5, 0.1, 25, -0.3, 2.5)
    assert mass >= 0

def test_apply_cuts_reduces_events():
    """Cuts should reduce number of events."""
    df = pd.DataFrame({
        'mu1_pt': [10, 30, 50],
        'mu2_pt': [15, 35, 55],
        'mu1_eta': [0.5, 1.0, 3.0],
        'mu2_eta': [0.3, 0.8, 2.8]
    })

    selected = apply_cuts(df, pt_min=20, eta_max=2.4)
    assert len(selected) < len(df)

def test_muon_pair_passes_cuts():
    """Test MuonPair cut selection."""
    pair = MuonPair(pt1=30, eta1=0.5, phi1=0,
                    pt2=25, eta2=-0.3, phi2=1.0)
    assert pair.passes_cuts(pt_min=20, eta_max=2.4) == True

    pair_fail = MuonPair(pt1=10, eta1=0.5, phi1=0,
                         pt2=25, eta2=-0.3, phi2=1.0)
    assert pair_fail.passes_cuts(pt_min=20, eta_max=2.4) == False
```

### Advanced Level

Comprehensive test suite:

```python
# test_analysis_advanced.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

@pytest.fixture
def sample_events():
    """Generate sample events for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'event_id': range(n),
        'mu1_pt': np.random.exponential(30, n),
        'mu1_eta': np.random.uniform(-2.5, 2.5, n),
        'mu1_phi': np.random.uniform(-np.pi, np.pi, n),
        'mu2_pt': np.random.exponential(30, n),
        'mu2_eta': np.random.uniform(-2.5, 2.5, n),
        'mu2_phi': np.random.uniform(-np.pi, np.pi, n),
    })

@pytest.mark.parametrize("pt1,pt2,expected_pass", [
    (30, 25, True),
    (10, 25, False),
    (30, 15, False),
    (10, 10, False),
])
def test_pt_cuts_parametrized(pt1, pt2, expected_pass):
    """Test pT cuts with various values."""
    pair = MuonPair(pt1=pt1, eta1=0, phi1=0, pt2=pt2, eta2=0, phi2=1)
    assert pair.passes_cuts(pt_min=20) == expected_pass

def test_mass_physical_constraints(sample_events):
    """Mass should always be positive and finite."""
    masses = sample_events.apply(
        lambda r: calculate_invariant_mass(
            r['mu1_pt'], r['mu1_eta'], r['mu1_phi'],
            r['mu2_pt'], r['mu2_eta'], r['mu2_phi']
        ), axis=1
    )
    assert (masses >= 0).all()
    assert np.isfinite(masses).all()

def test_efficiency_map_bounds():
    """Efficiency must be between 0 and 1."""
    eff_map = np.random.uniform(0.8, 1.0, (10, 10))
    assert eff_map.min() >= 0
    assert eff_map.max() <= 1

@patch('analysis.pd.read_csv')
def test_load_data_mocked(mock_read_csv, sample_events):
    """Test data loading with mocked file I/O."""
    mock_read_csv.return_value = sample_events
    df = load_and_validate_data('fake_path.csv')
    assert len(df) == 100
    mock_read_csv.assert_called_once_with('fake_path.csv')
```

---

## Part 6: Final Analysis and Presentation (30 min)

### Both Levels

1. Generate final plots: invariant mass spectrum with identified peak
2. Extract peak position, width, and significance
3. Compare with Monte Carlo expectations
4. Create summary of methodology and results

```python
from scipy.optimize import curve_fit
from scipy.stats import norm

def gaussian_plus_background(x, A, mu, sigma, a, b):
    """Gaussian signal + linear background."""
    return A * norm.pdf(x, mu, sigma) + a * x + b

def fit_mass_peak(masses, bins=50, range=(80, 100)):
    """Fit Gaussian + background to mass spectrum."""
    # Create histogram
    counts, bin_edges = np.histogram(masses, bins=bins, range=range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    errors = np.sqrt(counts)
    errors[errors == 0] = 1  # Avoid division by zero

    # Initial guess
    p0 = [counts.max(), 91.2, 2.5, 0, counts.mean()]

    # Fit
    try:
        popt, pcov = curve_fit(gaussian_plus_background, bin_centers, counts,
                               p0=p0, sigma=errors, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        return {
            'amplitude': (popt[0], perr[0]),
            'mean': (popt[1], perr[1]),
            'sigma': (popt[2], perr[2]),
            'success': True
        }
    except RuntimeError:
        return {'success': False}

def create_final_plot(data_masses, mc_masses, fit_result):
    """Create publication-quality final plot."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Data histogram
    counts_data, bins, _ = ax.hist(data_masses, bins=50, range=(60, 120),
                                    histtype='step', linewidth=2,
                                    color='black', label='Data')

    # MC histogram (normalized to data)
    scale = len(data_masses) / len(mc_masses)
    ax.hist(mc_masses, bins=50, range=(60, 120),
            histtype='stepfilled', alpha=0.3,
            weights=[scale]*len(mc_masses),
            color='blue', label='Monte Carlo')

    # Fit curve
    if fit_result['success']:
        x_fit = np.linspace(60, 120, 200)
        y_fit = gaussian_plus_background(x_fit, *[fit_result[k][0] for k in
                                                   ['amplitude', 'mean', 'sigma']], 0, 0)
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='Fit')

        # Add fit results text
        text = (f"$m = {fit_result['mean'][0]:.2f} \\pm {fit_result['mean'][1]:.2f}$ GeV\n"
                f"$\\sigma = {fit_result['sigma'][0]:.2f} \\pm {fit_result['sigma'][1]:.2f}$ GeV")
        ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(r'$m_{\mu\mu}$ (GeV/c²)', fontsize=14)
    ax.set_ylabel('Events / 1.2 GeV', fontsize=14)
    ax.set_title('Dimuon Invariant Mass Spectrum', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    return fig
```

---

## Deliverables

### Beginner Level

1. Jupyter notebook with documented code
2. Functions for mass calculation and plotting
3. Basic `MuonPair` class
4. 3-5 simple tests
5. Final mass plot with identified peak

### Advanced Level

1. Modular Python code structure
2. Complete function library with type hints and docstrings
3. Full class hierarchy with inheritance
4. Comprehensive test suite (10+ tests)
5. Configuration file system
6. Publication-quality plots
7. Analysis report with systematic uncertainties

---

## Evaluation Criteria

| Category | Weight | Description |
|----------|--------|-------------|
| **Code Quality** | 30% | Organization, readability, documentation |
| **Technical Implementation** | 40% | Correct physics, appropriate data structures, efficiency |
| **Testing and Validation** | 20% | Test coverage, physics validation, reliability |
| **Collaboration** | 10% | Pair programming, task division, code review |

---

## Key Takeaways

!!! success "Course Summary"
    Over these 3 days, you have learned:

    - **Day 1**: Python basics, NumPy arrays, Pandas DataFrames, Matplotlib
    - **Day 2**: Advanced data manipulation, functions, OOP
    - **Day 3**: Error handling, testing, debugging, complete analysis workflow

!!! info "Next Steps"
    - Explore the [Scikit-HEP ecosystem](https://scikit-hep.org/)
    - Learn [uproot](https://uproot.readthedocs.io/) for ROOT file I/O
    - Practice with real LHC open data
    - Join the [HEP Software Foundation](https://hepsoftwarefoundation.org/)

---

## Additional Resources

- [Particle Physics Reference Formulae](https://pdg.lbl.gov/2023/reviews/rpp2023-rev-kinematics.pdf)
- [CERN Open Data Portal](https://opendata.cern.ch/)
- [Scikit-HEP Tutorials](https://scikit-hep.org/tutorials)
- [Python Testing Best Practices](https://realpython.com/pytest-python-testing/)

---

**Navigate:** [← Day 3 Morning](day3_morning.md) | [Home](index.md)
