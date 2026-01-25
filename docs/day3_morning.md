# Day 3 - Morning Session
## Error Handling, Testing, and Debugging

**Duration:** 3.5 hours
**Topics:** Understanding and handling errors, Testing scientific code, Debugging tools

---

## 1. Understanding and Handling Errors (1h 15min)

### Reading Tracebacks

When Python encounters an error, it produces a **traceback** - a roadmap to the problem:

```python
# Example: a buggy analysis function
def calculate_mass_ratio(m1, m2):
    return m1 / m2

def analyze_particle_pair(data):
    result = calculate_mass_ratio(data['mass1'], data['mass2'])
    return result * 100

# This will fail
data = {'mass1': 0.105, 'mass2': 0}
analyze_particle_pair(data)
```

```
Traceback (most recent call last):
  File "analysis.py", line 10, in <module>
    analyze_particle_pair(data)
  File "analysis.py", line 6, in analyze_particle_pair
    result = calculate_mass_ratio(data['mass1'], data['mass2'])
  File "analysis.py", line 3, in calculate_mass_ratio
    return m1 / m2
ZeroDivisionError: float division by zero
```

**Reading the traceback** (bottom to top):

1. **Error type and message**: `ZeroDivisionError: float division by zero`
2. **Exact line**: `return m1 / m2` in `calculate_mass_ratio`
3. **Call chain**: How we got there

### Common Errors in Scientific Computing

| Error Type | Common Cause | Example |
|------------|--------------|---------|
| `ZeroDivisionError` | Division by zero in calculations | `pt / eta` when `eta = 0` |
| `ValueError` | Invalid value for operation | `np.sqrt(-1)` with real arrays |
| `IndexError` | Array index out of bounds | `particles[10]` with 5 particles |
| `KeyError` | Missing dictionary/DataFrame key | `df['mass']` when column is `'Mass'` |
| `TypeError` | Wrong data type | `"10" + 5` (string + int) |
| `FileNotFoundError` | Missing data file | `pd.read_csv('missing.csv')` |
| `MemoryError` | Array too large | `np.zeros((100000, 100000))` |

### Try/Except Blocks for Robust Pipelines

```python
import numpy as np
import pandas as pd

def load_data_safely(filepath):
    """Load data with error handling."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} events from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{filepath}' is empty")
        return None
    except Exception as e:
        print(f"Unexpected error loading {filepath}: {e}")
        return None

# Usage
data = load_data_safely('events.csv')
if data is not None:
    # Continue analysis
    pass
```

### Multiple Exception Handling

```python
def calculate_invariant_mass(E1, E2, p1, p2):
    """
    Calculate invariant mass with validation.

    Returns invariant mass or NaN if calculation fails.
    """
    try:
        # Check for valid inputs
        if E1 < 0 or E2 < 0:
            raise ValueError("Energy cannot be negative")

        E_total = E1 + E2
        p_total = p1 + p2
        m_squared = E_total**2 - p_total**2

        if m_squared < 0:
            raise ValueError("Unphysical: m² < 0")

        return np.sqrt(m_squared)

    except ValueError as e:
        print(f"Physics error: {e}")
        return np.nan
    except TypeError as e:
        print(f"Type error: {e}")
        return np.nan

# Test with edge cases
print(calculate_invariant_mass(50, 45, 49.5, 44.8))  # Valid
print(calculate_invariant_mass(-10, 45, 49.5, 44.8))  # Negative energy
print(calculate_invariant_mass("50", 45, 49.5, 44.8))  # Wrong type
```

### The `finally` Clause

`finally` runs regardless of whether an exception occurred - useful for cleanup:

```python
def process_data_file(filepath):
    """Process file with guaranteed cleanup."""
    file_handle = None
    try:
        file_handle = open(filepath, 'r')
        data = file_handle.read()
        # Process data...
        return data
    except FileNotFoundError:
        print(f"File {filepath} not found")
        return None
    finally:
        # This ALWAYS runs
        if file_handle is not None:
            file_handle.close()
            print("File handle closed")

# Better: use context manager (with statement)
def process_data_file_better(filepath):
    """Process file with context manager."""
    try:
        with open(filepath, 'r') as f:
            data = f.read()
            return data
    except FileNotFoundError:
        print(f"File {filepath} not found")
        return None
```

### Validating Input Data

```python
import numpy as np

def validate_particle_data(pt, eta, phi):
    """
    Validate particle kinematic data.

    Parameters:
    -----------
    pt : float or array
        Transverse momentum (must be >= 0)
    eta : float or array
        Pseudorapidity (typically |eta| < 5)
    phi : float or array
        Azimuthal angle (must be in [-π, π])

    Raises:
    -------
    ValueError : if any validation fails
    """
    pt = np.asarray(pt)
    eta = np.asarray(eta)
    phi = np.asarray(phi)

    # Check for NaN values
    if np.any(np.isnan(pt)) or np.any(np.isnan(eta)) or np.any(np.isnan(phi)):
        raise ValueError("Data contains NaN values")

    # Physical constraints
    if np.any(pt < 0):
        raise ValueError(f"pT must be non-negative, got min={pt.min():.2f}")

    if np.any(np.abs(eta) > 10):
        raise ValueError(f"Unphysical eta value: |eta| > 10")

    if np.any(np.abs(phi) > np.pi):
        raise ValueError(f"phi must be in [-π, π], got range [{phi.min():.2f}, {phi.max():.2f}]")

    return True

# Usage
try:
    validate_particle_data(pt=[30, 45, -5], eta=[0.5, 1.2, 2.0], phi=[0.1, 0.2, 0.3])
except ValueError as e:
    print(f"Validation failed: {e}")
```

### Custom Exceptions

```python
class AnalysisError(Exception):
    """Base exception for analysis errors."""
    pass

class InvalidEventError(AnalysisError):
    """Raised when event data is invalid."""
    def __init__(self, event_id, message):
        self.event_id = event_id
        self.message = message
        super().__init__(f"Event {event_id}: {message}")

class EfficiencyMapError(AnalysisError):
    """Raised when efficiency map is invalid."""
    pass

class PhysicsConstraintError(AnalysisError):
    """Raised when physics constraints are violated."""
    pass

# Usage
def process_event(event):
    """Process a single event with custom exceptions."""
    if event['n_muons'] < 2:
        raise InvalidEventError(
            event['event_id'],
            f"Need 2 muons, found {event['n_muons']}"
        )

    mass = calculate_invariant_mass(event)
    if mass > 1000:  # GeV
        raise PhysicsConstraintError(
            f"Unphysical mass: {mass:.1f} GeV"
        )

    return mass

# Handle specific exceptions
try:
    result = process_event({'event_id': 123, 'n_muons': 1})
except InvalidEventError as e:
    print(f"Skipping invalid event: {e}")
except PhysicsConstraintError as e:
    print(f"Physics problem: {e}")
except AnalysisError as e:
    print(f"Analysis error: {e}")
```

### Exercise 3.1 (35 min)

📓 **Open the companion notebook:** [day3_morning_exercises.ipynb](https://github.com/OpenAdalab/course_initiation_scientific_python/blob/main/notebooks/day3_morning_exercises.ipynb)

| Level | Task |
|-------|------|
| **Beginner** | Handle missing files in data loading, validate energy values, provide user-friendly error messages |
| **Advanced** | Implement custom exceptions for analysis failures, build error recovery for corrupted data |

---

## 2. Testing Scientific Code (1h 15min)

### Why Testing Matters

In scientific computing, bugs can lead to:

- **Wrong results** published in papers
- **Wasted time** debugging downstream issues
- **Irreproducible** analyses

Testing provides:

- **Confidence** that code works correctly
- **Documentation** of expected behavior
- **Safety net** for refactoring

### Introduction to pytest

pytest is Python's most popular testing framework:

```bash
# Install pytest
pip install pytest
```

Basic test structure:

```python
# test_kinematics.py

import numpy as np

def calculate_pt(px, py):
    """Calculate transverse momentum."""
    return np.sqrt(px**2 + py**2)

# Test functions start with 'test_'
def test_calculate_pt_basic():
    """Test pT calculation with simple values."""
    result = calculate_pt(3, 4)
    assert result == 5.0

def test_calculate_pt_zero():
    """Test pT with zero momentum."""
    result = calculate_pt(0, 0)
    assert result == 0.0

def test_calculate_pt_negative():
    """Test pT with negative components."""
    result = calculate_pt(-3, -4)
    assert result == 5.0
```

Run tests:

```bash
pytest test_kinematics.py -v
```

### Testing Numerical Code: Tolerances

Floating-point comparisons require tolerances:

```python
import numpy as np
import pytest

def calculate_mass(E, p):
    """Calculate invariant mass."""
    return np.sqrt(E**2 - p**2)

def test_mass_calculation():
    """Test mass calculation with known values."""
    # Z boson: E=91.2, p≈0 at rest
    mass = calculate_mass(91.2, 0)
    assert mass == 91.2  # Exact comparison OK here

def test_mass_calculation_relativistic():
    """Test with relativistic particle."""
    # Electron: m=0.000511 GeV, E=10 GeV
    m_electron = 0.000511
    E = 10.0
    p = np.sqrt(E**2 - m_electron**2)

    calculated_mass = calculate_mass(E, p)

    # Use pytest.approx for floating-point comparison
    assert calculated_mass == pytest.approx(m_electron, rel=1e-6)

def test_mass_with_numpy_arrays():
    """Test with array inputs."""
    E = np.array([10.0, 20.0, 30.0])
    p = np.array([9.9, 19.8, 29.7])

    masses = calculate_mass(E, p)

    # For arrays, use numpy testing
    expected = np.sqrt(E**2 - p**2)
    np.testing.assert_allclose(masses, expected, rtol=1e-10)
```

### Test Fixtures

Fixtures provide reusable test data:

```python
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_particles():
    """Create sample particle data for testing."""
    return pd.DataFrame({
        'pt': [30.0, 45.0, 25.0, 60.0],
        'eta': [0.5, -1.2, 2.0, 0.1],
        'phi': [0.1, -2.5, 1.5, 3.0],
        'mass': [0.105, 0.105, 0.000511, 0.105]  # muon, muon, electron, muon
    })

@pytest.fixture
def sample_event():
    """Create a sample event for testing."""
    return {
        'event_id': 12345,
        'run_id': 1,
        'n_particles': 4,
        'total_energy': 160.0
    }

def test_particle_selection(sample_particles):
    """Test particle selection with fixture data."""
    selected = sample_particles[sample_particles['pt'] > 30]
    assert len(selected) == 2

def test_eta_cut(sample_particles):
    """Test eta cut."""
    selected = sample_particles[np.abs(sample_particles['eta']) < 1.5]
    assert len(selected) == 3

def test_event_metadata(sample_event):
    """Test event fixture."""
    assert sample_event['event_id'] == 12345
    assert sample_event['n_particles'] == 4
```

### Parametrized Tests

Run the same test with different inputs:

```python
import pytest
import numpy as np

def calculate_delta_r(eta1, phi1, eta2, phi2):
    """Calculate angular separation between two objects."""
    deta = eta1 - eta2
    dphi = phi1 - phi2
    # Handle phi wrap-around
    while dphi > np.pi:
        dphi -= 2 * np.pi
    while dphi < -np.pi:
        dphi += 2 * np.pi
    return np.sqrt(deta**2 + dphi**2)

@pytest.mark.parametrize("eta1,phi1,eta2,phi2,expected", [
    (0, 0, 0, 0, 0.0),                      # Same point
    (1, 0, 0, 0, 1.0),                      # Only eta difference
    (0, 1, 0, 0, 1.0),                      # Only phi difference
    (1, 1, 0, 0, np.sqrt(2)),               # Both different
    (0, 3.0, 0, -3.0, pytest.approx(2*np.pi - 6, abs=0.01)),  # Phi wrap
])
def test_delta_r(eta1, phi1, eta2, phi2, expected):
    """Test ΔR calculation with various inputs."""
    result = calculate_delta_r(eta1, phi1, eta2, phi2)
    assert result == pytest.approx(expected, abs=1e-6)
```

### Testing for Expected Exceptions

```python
import pytest
import numpy as np

def validate_energy(E):
    """Validate that energy is positive."""
    if E < 0:
        raise ValueError(f"Energy must be positive, got {E}")
    return E

def test_validate_energy_negative():
    """Test that negative energy raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        validate_energy(-10)

    assert "must be positive" in str(excinfo.value)

def test_validate_energy_positive():
    """Test that positive energy passes."""
    assert validate_energy(50) == 50
```

### Test Organization

```
my_analysis/
├── analysis/
│   ├── __init__.py
│   ├── kinematics.py
│   └── selection.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py       # Shared fixtures
│   ├── test_kinematics.py
│   └── test_selection.py
└── pytest.ini
```

Example `conftest.py` (shared fixtures):

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def dimuon_events():
    """Generate sample dimuon events for testing."""
    np.random.seed(42)
    n_events = 100

    return pd.DataFrame({
        'event_id': range(n_events),
        'mu1_pt': np.random.exponential(30, n_events),
        'mu1_eta': np.random.uniform(-2.5, 2.5, n_events),
        'mu2_pt': np.random.exponential(30, n_events),
        'mu2_eta': np.random.uniform(-2.5, 2.5, n_events),
    })
```

### Exercise 3.2 (45 min)

📓 **Continue in the notebook:** [day3_morning_exercises.ipynb](https://github.com/OpenAdalab/course_initiation_scientific_python/blob/main/notebooks/day3_morning_exercises.ipynb)

| Level | Task |
|-------|------|
| **Beginner** | Write tests for kinematic functions, test selection cuts with known events |
| **Advanced** | Property-based testing for physical constraints, mock external data sources |

---

## 3. Debugging Tools and Best Practices (1h)

### Jupyter Debugging Features

#### Using print() strategically

```python
def find_z_candidates(muons, mass_window=(70, 110)):
    """Find Z boson candidates from muon pairs."""
    print(f"DEBUG: Input muons shape: {muons.shape}")
    print(f"DEBUG: Mass window: {mass_window}")

    candidates = []
    n_muons = len(muons)
    print(f"DEBUG: Processing {n_muons} muons")

    for i in range(n_muons):
        for j in range(i + 1, n_muons):
            mass = calculate_pair_mass(muons.iloc[i], muons.iloc[j])
            print(f"DEBUG: Pair ({i},{j}) mass = {mass:.2f}")

            if mass_window[0] < mass < mass_window[1]:
                candidates.append((i, j, mass))
                print(f"DEBUG: Found candidate!")

    print(f"DEBUG: Total candidates: {len(candidates)}")
    return candidates
```

#### Using assert for sanity checks

```python
def apply_efficiency_correction(data, efficiency_map):
    """Apply detector efficiency correction."""
    # Sanity checks
    assert efficiency_map.min() >= 0, "Efficiency cannot be negative"
    assert efficiency_map.max() <= 1, "Efficiency cannot exceed 1"
    assert len(data) > 0, "Empty data array"

    corrected = data / efficiency_map

    assert not np.any(np.isnan(corrected)), "NaN values in corrected data"
    assert not np.any(np.isinf(corrected)), "Infinite values in corrected data"

    return corrected
```

### Python Debugger (pdb)

```python
import pdb

def buggy_analysis(events):
    """Analysis with debugger breakpoint."""
    total = 0

    for i, event in enumerate(events):
        # Set breakpoint
        if i == 5:
            pdb.set_trace()  # Debugger stops here

        # Some calculation
        mass = event['mass']
        total += mass

    return total / len(events)

# In Python 3.7+, use builtin breakpoint()
def buggy_analysis_modern(events):
    """Analysis with modern breakpoint."""
    total = 0

    for i, event in enumerate(events):
        if i == 5:
            breakpoint()  # Same as pdb.set_trace()

        mass = event['mass']
        total += mass

    return total / len(events)
```

**Common pdb commands:**

| Command | Action |
|---------|--------|
| `n` | Execute next line |
| `s` | Step into function |
| `c` | Continue to next breakpoint |
| `p variable` | Print variable value |
| `l` | Show current location in code |
| `q` | Quit debugger |
| `h` | Help |

### IPython Magic Commands for Debugging

In Jupyter notebooks:

```python
# After an error, examine the traceback interactively
%debug

# Time a single line
%timeit np.sum(data**2)

# Time a cell
%%timeit
result = 0
for x in data:
    result += x**2

# Profile a function
%prun analyze_events(data)

# Line-by-line profiling (requires line_profiler)
%lprun -f my_function my_function(data)
```

### Profiling for Performance Bottlenecks

```python
import cProfile
import pstats

def profile_analysis():
    """Profile the analysis to find bottlenecks."""

    # Create profiler
    profiler = cProfile.Profile()

    # Run the analysis
    profiler.enable()
    result = run_full_analysis(data)
    profiler.disable()

    # Print statistics
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

    return result

# Or use as context manager
from contextlib import contextmanager
import time

@contextmanager
def timer(description):
    """Context manager for timing code blocks."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{description}: {elapsed:.3f} seconds")

# Usage
with timer("Event selection"):
    selected = apply_cuts(data)

with timer("Mass calculation"):
    masses = calculate_masses(selected)
```

### Memory Profiling

```python
import sys

def check_memory_usage(obj, name="object"):
    """Check memory usage of an object."""
    size_bytes = sys.getsizeof(obj)

    # For numpy arrays
    if hasattr(obj, 'nbytes'):
        size_bytes = obj.nbytes

    # For pandas DataFrames
    if hasattr(obj, 'memory_usage'):
        size_bytes = obj.memory_usage(deep=True).sum()

    # Convert to human readable
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            print(f"{name}: {size_bytes:.2f} {unit}")
            return
        size_bytes /= 1024

# Usage
import numpy as np
import pandas as pd

data = np.random.random((10000, 100))
check_memory_usage(data, "numpy array")

df = pd.DataFrame(data)
check_memory_usage(df, "pandas DataFrame")
```

### Logging for Production Code

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger('particle_analysis')

def analyze_run(run_id, data_path):
    """Analyze a single run with logging."""
    logger.info(f"Starting analysis for run {run_id}")

    try:
        logger.debug(f"Loading data from {data_path}")
        data = load_data(data_path)
        logger.info(f"Loaded {len(data)} events")

        logger.debug("Applying selection cuts")
        selected = apply_cuts(data)
        logger.info(f"Selected {len(selected)} events ({100*len(selected)/len(data):.1f}%)")

        if len(selected) < 10:
            logger.warning(f"Very few events selected for run {run_id}")

        logger.debug("Calculating masses")
        masses = calculate_masses(selected)

        logger.info(f"Run {run_id} complete. Mean mass: {masses.mean():.2f} GeV")
        return masses

    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in run {run_id}")
        raise

# Logging levels: DEBUG < INFO < WARNING < ERROR < CRITICAL
```

### Common Debugging Patterns

```python
# Pattern 1: Isolate the problem
def debug_step_by_step(data):
    """Break complex operation into steps for debugging."""
    # Step 1: Load
    print(f"Step 1: Loaded {len(data)} events")

    # Step 2: Filter
    filtered = data[data['pt'] > 20]
    print(f"Step 2: After pT cut: {len(filtered)} events")

    # Step 3: Calculate
    filtered['mass'] = calculate_mass(filtered)
    print(f"Step 3: Mass range: [{filtered['mass'].min():.1f}, {filtered['mass'].max():.1f}]")

    return filtered

# Pattern 2: Compare with known result
def verify_calculation():
    """Verify calculation against known physics."""
    # Z boson mass should be ~91.2 GeV
    z_mass = calculate_dimuon_mass(muon1, muon2)
    print(f"Calculated Z mass: {z_mass:.2f} GeV (expected ~91.2)")

    # Check if reasonable
    assert 80 < z_mass < 100, f"Z mass outside expected range: {z_mass}"

# Pattern 3: Binary search for bugs
def find_bad_event(events):
    """Use binary search to find problematic event."""
    n = len(events)

    # Check first half
    try:
        process_events(events[:n//2])
        print("Bug is in second half")
        # Continue with second half...
    except:
        print("Bug is in first half")
        # Continue with first half...
```

### Exercise 3.3 (30 min)

📓 **Continue in the notebook:** [day3_morning_exercises.ipynb](https://github.com/OpenAdalab/course_initiation_scientific_python/blob/main/notebooks/day3_morning_exercises.ipynb)

| Level | Task |
|-------|------|
| **Both** | Debug provided analysis code with intentional bugs (matching skill level) |

---

## Key Takeaways

!!! success "What We Learned"
    - **Error handling**: Try/except blocks, custom exceptions, validation
    - **Testing**: pytest basics, fixtures, parametrized tests, tolerances
    - **Debugging**: pdb, profiling, logging, systematic approaches

!!! warning "Best Practices"
    - Always validate input data at function boundaries
    - Write tests for edge cases and physical constraints
    - Use logging instead of print() for production code
    - Profile before optimizing - don't guess where bottlenecks are

!!! info "This Afternoon"
    Day 3 Afternoon: **Final Integration Project**

    Build a complete analysis pipeline to discover a resonance in dimuon events!

---

## Additional Resources

- [Python Exceptions Documentation](https://docs.python.org/3/tutorial/errors.html)
- [pytest Documentation](https://docs.pytest.org/)
- [Python Debugging with pdb](https://realpython.com/python-debugging-pdb/)
- [Python Logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [Effective Python Testing](https://realpython.com/pytest-python-testing/)

---

**Navigate:** [← Day 2 Afternoon](day2_afternoon.md) | [Day 3 Afternoon →](day3_afternoon.md)
