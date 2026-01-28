# Day 2 - Afternoon Session
## Functions and Object-Oriented Programming

**Duration:** 3.5 hours
**Topics:** Functions for code organization, Object-oriented programming, Integration exercise

---

## 4. Functions for Code Organization (1h 30min)

### Why Functions?

Functions are fundamental building blocks for:

- **Reusability**: Write once, use many times
- **Readability**: Give meaningful names to operations
- **Testing**: Test individual pieces of code
- **Maintenance**: Fix bugs in one place

```python
# Without functions: repetitive, error-prone
E1, p1 = 50.0, 49.5
mass1 = (E1**2 - p1**2)**0.5
E2, p2 = 45.0, 44.8
mass2 = (E2**2 - p2**2)**0.5

# With functions: clear, reusable
def calculate_mass(E, p):
    """Calculate invariant mass from energy and momentum."""
    return (E**2 - p**2)**0.5

mass1 = calculate_mass(50.0, 49.5)
mass2 = calculate_mass(45.0, 44.8)
```

### Function Anatomy

```python
def function_name(param1, param2, optional_param=default_value):
    """
    Brief description of what the function does.

    Parameters:
    -----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2
    optional_param : type, optional
        Description of optional param (default: default_value)

    Returns:
    --------
    type
        Description of return value

    Raises:
    -------
    ValueError
        When parameter is invalid

    Examples:
    ---------
    >>> function_name(1, 2)
    3
    """
    # Function body
    result = param1 + param2 + optional_param
    return result
```

### Parameters and Arguments

```python
# Positional arguments
def invariant_mass(E1, E2, p1, p2):
    E_total = E1 + E2
    p_total = p1 + p2
    return (E_total**2 - p_total**2)**0.5

mass = invariant_mass(50.0, 45.0, 49.5, 44.8)

# Keyword arguments (more explicit)
mass = invariant_mass(E1=50.0, E2=45.0, p1=49.5, p2=44.8)

# Default values
def apply_cut(data, threshold=20.0, variable='pt'):
    """Apply a minimum threshold cut."""
    return data[data[variable] > threshold]

# Can use default or override
filtered1 = apply_cut(df)  # threshold=20.0, variable='pt'
filtered2 = apply_cut(df, threshold=50.0)  # Override threshold
filtered3 = apply_cut(df, variable='energy')  # Override variable

# *args and **kwargs for flexible functions
def calculate_total_energy(*particles):
    """Sum energies from arbitrary number of particles."""
    return sum(p['energy'] for p in particles)

def configure_analysis(**options):
    """Configure analysis with arbitrary options."""
    for key, value in options.items():
        print(f"Setting {key} = {value}")
```

### Return Values

```python
# Single return value
def get_mean(data):
    return sum(data) / len(data)

# Multiple return values (tuple unpacking)
def get_statistics(data):
    """Return mean and standard deviation."""
    import numpy as np
    return np.mean(data), np.std(data)

mean, std = get_statistics([1, 2, 3, 4, 5])

# Return dictionary for many values
def analyze_distribution(data):
    """Comprehensive statistics."""
    import numpy as np
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'median': np.median(data),
        'min': np.min(data),
        'max': np.max(data),
        'n': len(data)
    }

stats = analyze_distribution(energies)
print(f"Mean: {stats['mean']:.2f}")
```
???+ tip "Type Hints for improving clarity"
    Type hints make code more readable and enable IDE support:

    ```python
    import numpy as np
    from typing import List, Tuple, Dict, Optional, Union
    import pandas as pd

    def calculate_pt(px: float, py: float) -> float:
        """Calculate transverse momentum."""
        return np.sqrt(px**2 + py**2)

    def apply_cuts(
        df: pd.DataFrame,
        pt_min: float = 20.0,
        eta_max: float = 2.5
    ) -> pd.DataFrame:
        """Apply kinematic cuts to DataFrame."""
        mask = (df['pt'] > pt_min) & (np.abs(df['eta']) < eta_max)
        return df[mask]

    def find_pairs(
        particles: List[Dict[str, float]]
    ) -> List[Tuple[int, int]]:
        """Find all particle pairs."""
        pairs = []
        n = len(particles)
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        return pairs

    def get_efficiency(
        pt: float,
        eta: float,
        efficiency_map: Optional[np.ndarray] = None
    ) -> float:
        """Look up detector efficiency."""
        if efficiency_map is None:
            return 0.95  # Default efficiency
        # Look up in map...
        return 0.95
    ```
??? tip "Optionnal : Lambda Functions" 
    Short anonymous functions for simple operations:

    ```python
    import numpy as np
    import pandas as pd

    # Lambda syntax: lambda arguments: expression
    square = lambda x: x**2
    print(square(5))  # 25

    # Useful with apply
    df = pd.DataFrame({'energy': [10, 20, 30, 40]})
    df['energy_gev'] = df['energy'].apply(lambda x: x / 1000)

    # Useful with map
    masses = [0.000511, 0.105, 1.777]  # e, mu, tau masses
    masses_mev = list(map(lambda m: m * 1000, masses))

    # Useful with filter
    energies = [15, 45, 23, 67, 12, 89]
    high_energy = list(filter(lambda e: e > 50, energies))

    # Useful with sorted
    particles = [
        {'name': 'electron', 'mass': 0.000511},
        {'name': 'muon', 'mass': 0.105},
        {'name': 'tau', 'mass': 1.777}
    ]
    sorted_by_mass = sorted(particles, key=lambda p: p['mass'])
    ```
??? tip "Optionnal Functional Programming Concepts"
    ```python
    from functools import reduce

    # map: apply function to all elements
    energies = [10, 20, 30, 40, 50]
    squared = list(map(lambda x: x**2, energies))
    # Same as: [x**2 for x in energies]

    # filter: select elements matching condition
    high = list(filter(lambda x: x > 25, energies))
    # Same as: [x for x in energies if x > 25]

    # reduce: combine elements into single value
    total = reduce(lambda a, b: a + b, energies)
    # Same as: sum(energies)

    # Chaining with comprehensions (preferred in Python)
    result = [x**2 for x in energies if x > 25]
    ```
??? info "Decorators (Advanced)"
    Decorators modify function behavior:

    ```python
    import time
    from functools import wraps

    # Timing decorator
    def timer(func):
        """Measure execution time of a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"{func.__name__} took {elapsed:.4f} seconds")
            return result
        return wrapper

    @timer
    def slow_calculation(n):
        """Simulate slow calculation."""
        total = 0
        for i in range(n):
            total += i**2
        return total

    result = slow_calculation(1000000)

    # Caching decorator (memoization)
    from functools import lru_cache

    @lru_cache(maxsize=128)
    def expensive_calculation(n):
        """Result is cached for repeated calls."""
        print(f"Computing for n={n}")
        return sum(i**2 for i in range(n))

    # First call computes
    result1 = expensive_calculation(1000)
    # Second call uses cache
    result2 = expensive_calculation(1000)
    ```

### Exercise 2.4 (50 min)

📓 **Open the companion notebook:** [day2_afternoon_exercises.ipynb](https://github.com/OpenAdalab/course_initiation_scientific_python/blob/main/notebooks/day2_afternoon_exercises.ipynb)

| Level | Task |
|-------|------|
| **Beginner** | Write `calculate_invariant_mass()`, `apply_selection_cuts()`, `plot_distribution()` |
| **Advanced** | Create function library with timing decorator, caching, array/DataFrame support |

---

## 5. Object-Oriented Programming for Analysis (1h 30min)

### What is OOP?

Object-Oriented Programming organizes code around **objects** that combine:

- **Attributes**: Data (properties)
- **Methods**: Functions that operate on that data

```
Real World          →  Python Code
─────────────────────────────────────
Particle            →  class Particle
  mass = 0.105          self.mass = 0.105
  charge = -1           self.charge = -1
  decay()               def decay(self): ...
```

### Classes and Instances

```python
class Particle:
    """A particle with basic properties."""

    def __init__(self, name, mass, charge):
        """
        Initialize a particle.

        Parameters:
        -----------
        name : str
            Particle name (e.g., 'muon')
        mass : float
            Rest mass in GeV/c²
        charge : int
            Electric charge in units of e
        """
        self.name = name
        self.mass = mass
        self.charge = charge

    def info(self):
        """Print particle information."""
        print(f"{self.name}: m={self.mass} GeV/c², q={self.charge}e")

# Create instances (specific particles)
electron = Particle('electron', 0.000511, -1)
muon = Particle('muon', 0.105, -1)
proton = Particle('proton', 0.938, +1)

# Access attributes
print(electron.mass)  # 0.000511

# Call methods
muon.info()  # muon: m=0.105 GeV/c², q=-1e
```

### Exemple of workflow : combining functions and classes

A common pattern is to use **functions** for data processing steps and **classes** to organize your analysis:

```python
import numpy as np
import pandas as pd

# Functions for processing steps
def load_events(filename):
    """Load event data from file."""
    return pd.read_csv(filename)

def select_muons(df, pt_min=20.0, eta_max=2.4):
    """Select muon candidates passing kinematic cuts."""
    mask = (df['pt'] > pt_min) & (np.abs(df['eta']) < eta_max)
    return df[mask]

def calculate_invariant_mass(mu1, mu2):
    """Calculate invariant mass of two muons."""
    E_total = mu1['E'] + mu2['E']
    px_total = mu1['px'] + mu2['px']
    py_total = mu1['py'] + mu2['py']
    pz_total = mu1['pz'] + mu2['pz']
    m2 = E_total**2 - (px_total**2 + py_total**2 + pz_total**2)
    return np.sqrt(m2) if m2 > 0 else 0

# Class to organize the analysis
class ZBosonAnalysis:
    """Simple Z boson analysis."""

    def __init__(self, name):
        self.name = name
        self.data = None
        self.selected = None
        self.masses = []

    def load(self, filename):
        """Load data using our function."""
        self.data = load_events(filename)
        print(f"Loaded {len(self.data)} events")

    def select(self, pt_min=20.0, eta_max=2.4):
        """Apply selection using our function."""
        self.selected = select_muons(self.data, pt_min, eta_max)
        print(f"Selected {len(self.selected)} muons")

    def find_z_candidates(self):
        """Find Z candidates from muon pairs."""
        # Simplified: pair consecutive muons
        for i in range(0, len(self.selected) - 1, 2):
            mu1 = self.selected.iloc[i]
            mu2 = self.selected.iloc[i + 1]
            mass = calculate_invariant_mass(mu1, mu2)
            if 70 < mass < 110:  # Z mass window
                self.masses.append(mass)
        print(f"Found {len(self.masses)} Z candidates")

    def summary(self):
        """Print analysis summary."""
        print(f"\n=== {self.name} ===")
        print(f"Z candidates: {len(self.masses)}")
        if self.masses:
            print(f"Mean mass: {np.mean(self.masses):.1f} GeV")

# Usage
analysis = ZBosonAnalysis("My Z Analysis")
analysis.load("muons.csv")
analysis.select(pt_min=25.0)
analysis.find_z_candidates()
analysis.summary()
```

This pattern keeps functions simple and testable, while the class provides a clear workflow structure. 
You can write some kind of code in a python file (```.py```), in order to be automated & debuged more easily !



### (Optionnal) Advanced exemples : The Particle Class with 4-Momentum

```python
import numpy as np

class Particle:
    """A particle with 4-momentum."""

    def __init__(self, px, py, pz, E, name='unknown'):
        """
        Initialize particle with 4-momentum.

        Parameters:
        -----------
        px, py, pz : float
            Momentum components (GeV/c)
        E : float
            Energy (GeV)
        name : str
            Particle name
        """
        self.px = px
        self.py = py
        self.pz = pz
        self.E = E
        self.name = name

    @property
    def pt(self):
        """Transverse momentum."""
        return np.sqrt(self.px**2 + self.py**2)

    @property
    def p(self):
        """Total momentum magnitude."""
        return np.sqrt(self.px**2 + self.py**2 + self.pz**2)

    @property
    def mass(self):
        """Invariant mass from 4-momentum."""
        m2 = self.E**2 - self.p**2
        return np.sqrt(m2) if m2 > 0 else 0

    @property
    def eta(self):
        """Pseudorapidity."""
        if self.p == abs(self.pz):
            return float('inf') * np.sign(self.pz)
        return 0.5 * np.log((self.p + self.pz) / (self.p - self.pz))

    @property
    def phi(self):
        """Azimuthal angle."""
        return np.arctan2(self.py, self.px)

    def __repr__(self):
        """String representation."""
        return f"Particle({self.name}, pt={self.pt:.2f}, eta={self.eta:.2f})"

    def __add__(self, other):
        """Add two particles (combine 4-momenta)."""
        return Particle(
            self.px + other.px,
            self.py + other.py,
            self.pz + other.pz,
            self.E + other.E,
            name=f"{self.name}+{other.name}"
        )
```

### Using the Particle Class

```python
# Create two muons from Z decay
mu1 = Particle(px=30, py=20, pz=10, E=50, name='mu+')
mu2 = Particle(px=-25, py=-15, pz=40, E=45, name='mu-')

print(f"Muon 1: pt={mu1.pt:.2f}, eta={mu1.eta:.2f}, phi={mu1.phi:.2f}")
print(f"Muon 2: pt={mu2.pt:.2f}, eta={mu2.eta:.2f}, phi={mu2.phi:.2f}")

# Combine to get Z candidate
z_candidate = mu1 + mu2
print(f"\nZ candidate: mass={z_candidate.mass:.2f} GeV/c²")
```

### The Event Class

```python
class Event:
    """Container for particles in a collision event."""

    def __init__(self, event_id, run_id=0):
        """
        Initialize an event.

        Parameters:
        -----------
        event_id : int
            Unique event identifier
        run_id : int
            Run number
        """
        self.event_id = event_id
        self.run_id = run_id
        self.particles = []
        self.metadata = {}

    def add_particle(self, particle):
        """Add a particle to the event."""
        self.particles.append(particle)

    def n_particles(self):
        """Number of particles in event."""
        return len(self.particles)

    def get_particles(self, name=None):
        """Get particles, optionally filtered by name."""
        if name is None:
            return self.particles
        return [p for p in self.particles if p.name == name]

    def total_energy(self):
        """Total energy of all particles."""
        return sum(p.E for p in self.particles)

    def select_particles(self, pt_min=0, eta_max=float('inf')):
        """Select particles passing kinematic cuts."""
        return [p for p in self.particles
                if p.pt > pt_min and abs(p.eta) < eta_max]

    def get_leading(self, variable='pt'):
        """Get particle with highest value of variable."""
        if not self.particles:
            return None
        return max(self.particles, key=lambda p: getattr(p, variable))

    def __repr__(self):
        return f"Event({self.event_id}, n_particles={self.n_particles()})"
```

### Using the Event Class

```python
# Create an event
event = Event(event_id=12345, run_id=1)

# Add particles
event.add_particle(Particle(30, 20, 10, 50, 'muon'))
event.add_particle(Particle(-25, -15, 40, 45, 'muon'))
event.add_particle(Particle(50, 30, 20, 100, 'jet'))
event.add_particle(Particle(-40, 25, -10, 80, 'jet'))

print(event)
print(f"Total energy: {event.total_energy():.2f} GeV")
print(f"Leading particle: {event.get_leading('pt')}")

# Apply selection
selected = event.select_particles(pt_min=30)
print(f"Particles with pt > 30: {len(selected)}")
```

### Inheritance: Specialized Particles

```python
class Lepton(Particle):
    """Base class for leptons."""

    def __init__(self, px, py, pz, E, charge, name='lepton'):
        super().__init__(px, py, pz, E, name)
        self.charge = charge

    def is_isolated(self, isolation_cone=0.3, max_pt=5.0):
        """Check if lepton is isolated (placeholder)."""
        # In real analysis, check nearby activity
        return True


class Electron(Lepton):
    """Electron with specific properties."""

    MASS = 0.000511  # GeV/c²

    def __init__(self, px, py, pz, E, charge=-1):
        super().__init__(px, py, pz, E, charge, name='electron')
        self.cluster_energy = E  # Calorimeter cluster

    def energy_momentum_match(self, tolerance=0.1):
        """Check E/p consistency for electrons."""
        if self.p == 0:
            return False
        e_over_p = self.E / self.p
        return abs(e_over_p - 1.0) < tolerance


class Muon(Lepton):
    """Muon with specific properties."""

    MASS = 0.105  # GeV/c²

    def __init__(self, px, py, pz, E, charge=-1):
        super().__init__(px, py, pz, E, charge, name='muon')
        self.track_quality = 1.0  # Track quality score

    def is_combined(self):
        """Check if muon has both inner and muon spectrometer tracks."""
        return True  # Placeholder


class Jet:
    """Hadronic jet."""

    def __init__(self, px, py, pz, E, btag_score=0.0):
        self.px = px
        self.py = py
        self.pz = pz
        self.E = E
        self.btag_score = btag_score
        self.name = 'jet'

    @property
    def pt(self):
        return np.sqrt(self.px**2 + self.py**2)

    def is_b_tagged(self, working_point=0.7):
        """Check if jet is b-tagged."""
        return self.btag_score > working_point
```

??? info "Special methods (Advanced)"
    Python magic (dunder) methods are special methods with double underscores ```__``` that enable operator overloading and custom object behavior

    ```python
    class FourVector:
        """Four-vector with special methods."""

        def __init__(self, px, py, pz, E):
            self.px = px
            self.py = py
            self.pz = pz
            self.E = E

        def __repr__(self):
            """String representation for developers."""
            return f"FourVector(px={self.px}, py={self.py}, pz={self.pz}, E={self.E})"

        def __str__(self):
            """String representation for users."""
            return f"4-vector: (E={self.E:.2f}, p=({self.px:.2f}, {self.py:.2f}, {self.pz:.2f}))"

        def __add__(self, other):
            """Add two four-vectors."""
            return FourVector(
                self.px + other.px,
                self.py + other.py,
                self.pz + other.pz,
                self.E + other.E
            )

        def __sub__(self, other):
            """Subtract two four-vectors."""
            return FourVector(
                self.px - other.px,
                self.py - other.py,
                self.pz - other.pz,
                self.E - other.E
            )

        def __mul__(self, scalar):
            """Multiply by scalar."""
            return FourVector(
                self.px * scalar,
                self.py * scalar,
                self.pz * scalar,
                self.E * scalar
            )

        def __eq__(self, other):
            """Check equality."""
            return (self.px == other.px and self.py == other.py and
                    self.pz == other.pz and self.E == other.E)

        def __len__(self):
            """Return number of components."""
            return 4

        @property
        def mass(self):
            """Invariant mass."""
            m2 = self.E**2 - (self.px**2 + self.py**2 + self.pz**2)
            return np.sqrt(m2) if m2 > 0 else 0

    # Usage
    v1 = FourVector(30, 20, 10, 50)
    v2 = FourVector(-25, -15, 40, 45)

    print(v1)           # Uses __str__
    print(repr(v1))     # Uses __repr__
    print(v1 + v2)      # Uses __add__
    print((v1 + v2).mass)  # Combined mass
    ```

??? info "Analysis Workflow with Classes (Advanced)"
    For more complex analyses, you can create a generic `AnalysisChain` class that manages steps dynamically:

    ```python
    class AnalysisChain:
        """Configurable analysis workflow."""

        def __init__(self, name, config=None):
            self.name = name
            self.config = config or {}
            self.steps = []
            self.results = {}

        def add_step(self, name, function):
            """Add an analysis step."""
            self.steps.append({'name': name, 'function': function})

        def run(self, data):
            """Execute all analysis steps."""
            result = data
            for step in self.steps:
                print(f"Running: {step['name']}")
                result = step['function'](result)
                self.results[step['name']] = result
            return result

        def summary(self):
            """Print analysis summary."""
            print(f"\n=== Analysis: {self.name} ===")
            for step_name, result in self.results.items():
                if hasattr(result, '__len__'):
                    print(f"  {step_name}: {len(result)} entries")
                else:
                    print(f"  {step_name}: {result}")

    # Usage example
    def load_data(path):
        """Load data from file."""
        import pandas as pd
        return pd.DataFrame({
            'pt': np.random.exponential(30, 1000),
            'eta': np.random.uniform(-2.5, 2.5, 1000),
            'trigger': np.random.choice([True, False], 1000, p=[0.7, 0.3])
        })

    def apply_trigger(df):
        return df[df['trigger']]

    def apply_kinematic_cuts(df):
        return df[(df['pt'] > 20) & (np.abs(df['eta']) < 2.4)]

    # Build and run analysis
    analysis = AnalysisChain('Z Analysis')
    analysis.add_step('Load', lambda x: load_data(x))
    analysis.add_step('Trigger', apply_trigger)
    analysis.add_step('Kinematics', apply_kinematic_cuts)

    final_data = analysis.run('data.csv')
    analysis.summary()
    ```

### Exercise 2.5 (60 min)

📓 **Continue in the notebook:** [day2_afternoon_exercises.ipynb](https://github.com/OpenAdalab/course_initiation_scientific_python/blob/main/notebooks/day2_afternoon_exercises.ipynb)

| Level | Task |
|-------|------|
| **Beginner** | Create `Particle` class with 4-momentum, `Event` class with selection methods |
| **Advanced** | Full `FourVector` with Lorentz boost, inheritance hierarchy, `AnalysisChain` class |

---

## 6. Integration Exercise (30 min)

📓 **Continue in the notebook:** [day2_afternoon_exercises.ipynb](https://github.com/OpenAdalab/course_initiation_scientific_python/blob/main/notebooks/day2_afternoon_exercises.ipynb)

| Level | Task |
|-------|------|
| **Both** | Combine functions and classes to analyze a complete dataset |

This exercise brings together everything learned today:

1. Load data using functions
2. Create Particle and Event objects
3. Apply selection cuts using class methods
4. Calculate derived quantities
5. Visualize results

---

## Key Takeaways

!!! success "What We Learned"
    - **Functions**: Modular, reusable code with clear interfaces
    - **Type hints**: Self-documenting code
    - **Classes**: Organize data and behavior together
    - **Inheritance**: Share code between related classes
    - **Special methods**: Pythonic object behavior

!!! warning "Design Principles"
    - Keep functions small and focused
    - Use classes when data and behavior belong together
    - Prefer composition over inheritance for flexibility
    - Write docstrings for all public functions and classes

!!! info "Tomorrow"
    Day 3 will cover:

    - Error handling and debugging
    - Testing scientific code
    - Final integration project

---

## Additional Resources

- [Python Functions Tutorial](https://docs.python.org/3/tutorial/controlflow.html#defining-functions)
- [Python Classes Tutorial](https://docs.python.org/3/tutorial/classes.html)
- [Real Python: OOP in Python](https://realpython.com/python3-object-oriented-programming/)
- [Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
- [Fluent Python (Book)](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)

---

**Navigate:** [← Day 2 Morning](day2_morning.md) | [Day 3 Morning →](day3_morning.md)
