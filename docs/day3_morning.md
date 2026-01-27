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


??? "Optionnal : Custom Exceptions"
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

#### How pytest Works

When you run `pytest`, it automatically:

1. **Discovers tests**: Searches for files named `test_*.py` or `*_test.py`
2. **Collects test functions**: Finds functions starting with `test_`
3. **Runs each test in isolation**: Each test runs independently
4. **Reports results**: Shows passed ✓, failed ✗, or error status

```bash
# Run all tests in current directory
pytest

# Run tests in a specific file
pytest test_kinematics.py

# Run with verbose output (shows each test name)
pytest -v

# Run a specific test function
pytest test_kinematics.py::test_calculate_pt_basic

# Stop at first failure
pytest -x
```

**The test lifecycle:**

```
For each test function:
    1. Setup (prepare any needed resources)
    2. Execute the test function
    3. Assert results (pass/fail)
    4. Teardown (cleanup resources)
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

### OPTIONNAL: Test Fixtures: Reusable Test Setup

#### What is a Test Fixture?

A **fixture** is any consistent, reusable setup that tests need to run reliably. The term comes from manufacturing, where a "fixture" holds a piece in place during work.

In testing, fixtures provide:

- **Test data**: Sample DataFrames, arrays, or dictionaries
- **Resources**: Database connections, temporary files
- **State**: A known starting point for each test

**Without fixtures** (repetitive code):

```python
def test_particle_selection():
    # Setup repeated in every test!
    particles = pd.DataFrame({
        'pt': [30.0, 45.0, 25.0],
        'eta': [0.5, -1.2, 2.0]
    })
    selected = particles[particles['pt'] > 30]
    assert len(selected) == 1

def test_eta_cut():
    # Same setup again!
    particles = pd.DataFrame({
        'pt': [30.0, 45.0, 25.0],
        'eta': [0.5, -1.2, 2.0]
    })
    selected = particles[np.abs(particles['eta']) < 1.5]
    assert len(selected) == 2
```

#### The `@pytest.fixture` Decorator

The `@pytest.fixture` decorator transforms a function into a fixture provider. When a test function has a parameter with the same name as a fixture, pytest automatically:

1. Calls the fixture function
2. Passes its return value to the test

```python
import pytest
import pandas as pd

@pytest.fixture
def particles():
    """This function becomes a fixture."""
    return pd.DataFrame({
        'pt': [30.0, 45.0, 25.0],
        'eta': [0.5, -1.2, 2.0]
    })

# pytest sees 'particles' parameter → calls the fixture → injects the DataFrame
def test_particle_selection(particles):
    selected = particles[particles['pt'] > 30]
    assert len(selected) == 1

def test_eta_cut(particles):
    selected = particles[np.abs(particles['eta']) < 1.5]
    assert len(selected) == 2
```

**Key benefits:**

- **DRY (Don't Repeat Yourself)**: Setup code written once
- **Isolation**: Each test gets a fresh copy of the data
- **Readability**: Test functions focus on assertions, not setup
- **Maintainability**: Change setup in one place

#### Fixture Scope

By default, fixtures run once per test. You can control this with the `scope` parameter:

```python
@pytest.fixture(scope="function")  # Default: run for each test
def fresh_data():
    return create_data()

@pytest.fixture(scope="module")  # Run once per test file
def expensive_data():
    return load_large_dataset()  # Only loaded once!

@pytest.fixture(scope="session")  # Run once for entire test session
def database_connection():
    return connect_to_db()
```

#### Complete Fixture Examples

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
import numpy as np
import pandas as pd

def calculate_pair_mass(mu1, mu2):
    """Calculate invariant mass of two muons (simplified)."""
    # Simplified formula using pT and eta
    E1 = mu1['pt'] * np.cosh(mu1['eta'])
    E2 = mu2['pt'] * np.cosh(mu2['eta'])
    p1 = mu1['pt'] * np.sinh(mu1['eta'])
    p2 = mu2['pt'] * np.sinh(mu2['eta'])
    return np.sqrt((E1 + E2)**2 - (p1 + p2)**2)

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

# Create sample muon data and test
muons = pd.DataFrame({
    'pt': [45.0, 38.0, 25.0, 50.0],
    'eta': [0.5, -0.8, 1.2, 0.1],
    'phi': [0.1, 2.5, -1.0, 1.5]
})

candidates = find_z_candidates(muons)
print(f"\nFound {len(candidates)} Z candidates")
```

#### Using assert for sanity checks

```python
import numpy as np

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

# Test with valid data
data = np.array([100, 200, 150, 180])
efficiency = np.array([0.85, 0.90, 0.78, 0.92])
corrected = apply_efficiency_correction(data, efficiency)
print(f"Original: {data}")
print(f"Corrected: {corrected.round(1)}")

# Test with invalid efficiency (will raise AssertionError)
# bad_efficiency = np.array([0.85, 1.5, 0.78, 0.92])  # 1.5 > 1!
# apply_efficiency_correction(data, bad_efficiency)
```

### Python Debugger (pdb)

```python
import pdb

# Create sample events for testing
events = [
    {'event_id': i, 'mass': 90 + np.random.normal(0, 5)}
    for i in range(10)
]

def buggy_analysis(events):
    """Analysis with debugger breakpoint."""
    total = 0

    for i, event in enumerate(events):
        # Set breakpoint at event 5
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

# Uncomment to test (will open interactive debugger):
# result = buggy_analysis(events)
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
import numpy as np

# Create sample data for testing
data = np.random.random(10000)
```

#### The `%debug` Magic Command

The `%debug` command opens an interactive debugger **after an error has occurred**. This is called "post-mortem debugging" - you can inspect the state of your program at the exact moment it crashed.

**How to use `%debug`:**

1. Run a cell that produces an error
2. In the next cell, type `%debug` and run it
3. An interactive debugger (ipdb) opens at the point of failure
4. Use pdb commands to inspect variables, navigate the call stack, etc.
5. Type `q` to quit the debugger

```python
# Step 1: Run code that will fail
def compute_ratio(a, b):
    """Compute ratio - will fail if b contains zeros."""
    return a / b

def analyze_samples(samples):
    """Analyze samples by computing ratios."""
    ratios = []
    for i, sample in enumerate(samples):
        ratio = compute_ratio(sample['value'], sample['weight'])
        ratios.append(ratio)
    return np.array(ratios)

# This will raise ZeroDivisionError!
samples = [
    {'value': 10, 'weight': 2},
    {'value': 15, 'weight': 3},
    {'value': 20, 'weight': 0},  # Bug: weight is 0!
    {'value': 25, 'weight': 5},
]

# Uncomment to see the error:
# analyze_samples(samples)
```

```python
# Step 2: After the error, run %debug in a new cell
%debug
```

**Inside the debugger, you can:**

```
ipdb> p sample          # Print the current sample
{'value': 20, 'weight': 0}

ipdb> p i               # Print the loop index
2

ipdb> p samples         # Print all samples
[{'value': 10, 'weight': 2}, ...]

ipdb> u                 # Go UP one level in the call stack
ipdb> d                 # Go DOWN one level in the call stack

ipdb> l                 # List code around current line

ipdb> q                 # Quit the debugger
```

#### Automatic Debugging with `%pdb`

You can enable automatic post-mortem debugging so you don't need to type `%debug` manually:

```python
# Enable automatic debugger on exceptions
%pdb on

# Now any error will automatically open the debugger
# analyze_samples(samples)  # Would open debugger automatically

# Disable automatic debugger
%pdb off
```

#### Debugging Commands Summary

| Command | Action |
|---------|--------|
| `p var` | Print variable value |
| `pp var` | Pretty-print variable |
| `l` | List source code around current line |
| `u` | Go UP one level in call stack |
| `d` | Go DOWN one level in call stack |
| `w` | Print full stack trace (where) |
| `q` | Quit debugger |
| `h` | Help |

```python
# Time a single line
%timeit np.sum(data**2)
```

```python
%%timeit
# Time a cell
result = 0
for x in data:
    result += x**2
```

```python
# Profile a function
def analyze_data(arr):
    return np.sqrt(np.sum(arr**2))

%prun analyze_data(data)
```

### OPTIONNAL : Profiling for Performance Bottlenecks

```python
import cProfile
import pstats
import numpy as np
from contextlib import contextmanager
import time

# Define functions to profile
def slow_sum(data):
    """Intentionally slow sum using Python loop."""
    total = 0
    for x in data:
        total += x
    return total

def fast_sum(data):
    """Fast sum using NumPy."""
    return np.sum(data)

def run_full_analysis(data):
    """Run analysis pipeline."""
    result1 = slow_sum(data)
    result2 = fast_sum(data)
    return result1, result2

# Create test data
data = np.random.random(100000)

# Profile the analysis
def profile_analysis():
    """Profile the analysis to find bottlenecks."""
    profiler = cProfile.Profile()

    profiler.enable()
    result = run_full_analysis(data)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions

    return result

profile_analysis()
```

```python
# Timer context manager - very useful!
@contextmanager
def timer(description):
    """Context manager for timing code blocks."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{description}: {elapsed:.3f} seconds")

# Usage example
data = np.random.random(1000000)

with timer("Python loop sum"):
    total = 0
    for x in data:
        total += x

with timer("NumPy sum"):
    total = np.sum(data)
```

### Memory Profiling

```python
import sys
import numpy as np
import pandas as pd

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

# Test with different data structures
data = np.random.random((10000, 100))
check_memory_usage(data, "numpy array (10000x100)")

df = pd.DataFrame(data)
check_memory_usage(df, "pandas DataFrame (10000x100)")

# Compare dtypes impact on memory
small_ints = np.array([1, 2, 3, 4, 5], dtype=np.int8)
large_ints = np.array([1, 2, 3, 4, 5], dtype=np.int64)
check_memory_usage(small_ints, "int8 array")
check_memory_usage(large_ints, "int64 array")
```

### Logging for Production Code

```python
import logging
import numpy as np
import pandas as pd

# Configure logging (in Jupyter, use StreamHandler only)
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see all messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Print to console/notebook
    ]
)

logger = logging.getLogger('particle_analysis')

# Define the helper functions
def load_data(data_path):
    """Simulate loading data."""
    # In real code: return pd.read_csv(data_path)
    np.random.seed(42)
    return pd.DataFrame({
        'pt': np.random.exponential(30, 1000),
        'eta': np.random.uniform(-2.5, 2.5, 1000),
        'mass': np.random.normal(91, 5, 1000)
    })

def apply_cuts(data):
    """Apply selection cuts."""
    return data[(data['pt'] > 20) & (np.abs(data['eta']) < 2.0)]

def calculate_masses(data):
    """Return mass column."""
    return data['mass']

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

# Test the logging
masses = analyze_run(run_id=12345, data_path="events.csv")

# Logging levels: DEBUG < INFO < WARNING < ERROR < CRITICAL
```

### Common Debugging Patterns

```python
import numpy as np
import pandas as pd

# Create sample data for all patterns
np.random.seed(42)
sample_data = pd.DataFrame({
    'pt': np.random.exponential(30, 100),
    'eta': np.random.uniform(-2.5, 2.5, 100),
    'phi': np.random.uniform(-np.pi, np.pi, 100)
})

# Pattern 1: Isolate the problem - break into steps
def calculate_mass(data):
    """Simple mass calculation."""
    return data['pt'] * 0.5  # Simplified

def debug_step_by_step(data):
    """Break complex operation into steps for debugging."""
    # Step 1: Check input
    print(f"Step 1: Loaded {len(data)} events")
    print(f"        Columns: {list(data.columns)}")

    # Step 2: Filter
    filtered = data[data['pt'] > 20].copy()
    print(f"Step 2: After pT cut: {len(filtered)} events")

    # Step 3: Calculate
    filtered['mass'] = calculate_mass(filtered)
    print(f"Step 3: Mass range: [{filtered['mass'].min():.1f}, {filtered['mass'].max():.1f}]")

    return filtered

result = debug_step_by_step(sample_data)
```

```python
# Pattern 2: Compare with known result
def calculate_pt(px, py):
    """Calculate transverse momentum."""
    return np.sqrt(px**2 + py**2)

def verify_calculation():
    """Verify calculation against known values."""
    # Known: 3-4-5 triangle
    pt = calculate_pt(3, 4)
    print(f"Calculated pT: {pt:.2f} (expected 5.0)")
    assert pt == 5.0, f"pT calculation wrong: {pt}"

    # Known: 5-12-13 triangle
    pt = calculate_pt(5, 12)
    print(f"Calculated pT: {pt:.2f} (expected 13.0)")
    assert pt == 13.0, f"pT calculation wrong: {pt}"

    print("All verifications passed!")

verify_calculation()
```

```python
# Pattern 3: Binary search for bugs
def process_events(events):
    """Process events - fails on event with negative pt."""
    for i, row in events.iterrows():
        if row['pt'] < 0:
            raise ValueError(f"Negative pT at index {i}")
    return len(events)

# Create data with one bad event
bad_data = sample_data.copy()
bad_data.loc[73, 'pt'] = -5  # Introduce bug at index 73

def find_bad_event(events):
    """Use binary search to find problematic event."""
    n = len(events)
    print(f"Searching {n} events...")

    if n <= 1:
        print(f"Found bad event at index: {events.index[0]}")
        return events.index[0]

    mid = n // 2
    first_half = events.iloc[:mid]
    second_half = events.iloc[mid:]

    try:
        process_events(first_half)
        print(f"Bug is in second half (indices {mid}-{n-1})")
        return find_bad_event(second_half)
    except ValueError:
        print(f"Bug is in first half (indices 0-{mid-1})")
        return find_bad_event(first_half)

# Find the bad event
bad_index = find_bad_event(bad_data)
print(f"\nBad event details:\n{bad_data.loc[bad_index]}")
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
