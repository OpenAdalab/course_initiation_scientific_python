# Day 1 - Afternoon Session
## Lists, Advanced NumPy, and Pandas DataFrames

**Duration:** 3.5 hours
**Topics:** Python data structures, deep dive into NumPy arrays, introduction to Pandas

---

## 4. Lists and Core Python Data Structures (1 hour)

### Python Lists

Lists are ordered, mutable collections of items. They're fundamental to Python but less efficient than NumPy arrays for numerical work.

#### Creating Lists

```python
# List of particle energies
energies = [45.2, 67.8, 23.1, 89.5, 34.6]

# List of particle types
particles = ['electron', 'muon', 'photon', 'electron']

# Empty list
hits = []

# List with mixed types (allowed but not recommended)
mixed = [42, 'muon', 3.14, True]
```

#### Indexing and Slicing

```python
energies = [45.2, 67.8, 23.1, 89.5, 34.6]

# Indexing (same as NumPy)
energies[0]      # First element: 45.2
energies[-1]     # Last element: 34.6
energies[-2]     # Second from end: 89.5

# Slicing [start:stop:step]
energies[1:3]    # [67.8, 23.1]
energies[:2]     # First two: [45.2, 67.8]
energies[2:]     # From index 2 to end
energies[::2]    # Every second element
energies[::-1]   # Reverse the list
```

#### List Operations

```python
# Append (add to end)
energies.append(55.3)

# Extend (add multiple items)
energies.extend([12.5, 78.9])

# Insert at position
energies.insert(0, 100.0)  # Insert at beginning

# Remove by value
energies.remove(45.2)  # Remove first occurrence

# Remove by index
del energies[0]        # Delete first element
popped = energies.pop()  # Remove and return last element

# Length
n = len(energies)

# Check membership
if 67.8 in energies:
    print("Found!")
```

#### List Comprehensions

List comprehensions provide a concise way to create lists:

```python
# Traditional loop
squared = []
for x in [1, 2, 3, 4, 5]:
    squared.append(x**2)

# List comprehension (more Pythonic)
squared = [x**2 for x in [1, 2, 3, 4, 5]]
# Result: [1, 4, 9, 16, 25]

# With condition
high_energy = [e for e in energies if e > 50]

# Apply function
import math
magnitudes = [math.sqrt(px**2 + py**2) for px, py in momentum_pairs]
```

### Dictionaries

Dictionaries store key-value pairs, perfect for metadata and configurations.

```python
# Detector configuration
detector = {
    'name': 'ATLAS',
    'layers': 3,
    'radius': 1.2,  # meters
    'efficiency': 0.95,
    'active': True
}

# Accessing values
detector['name']           # 'ATLAS'
detector.get('radius')     # 1.2
detector.get('missing', 0)  # Returns 0 if key doesn't exist

# Adding/modifying
detector['temperature'] = 77  # Kelvin
detector['efficiency'] = 0.96

# Keys and values
detector.keys()    # All keys
detector.values()  # All values
detector.items()   # Key-value pairs

# Check if key exists
if 'radius' in detector:
    print(f"Radius: {detector['radius']} m")
```

#### Nested Dictionaries

```python
# Complete detector geometry
detector_config = {
    'barrel': {
        'layers': 3,
        'radii': [1.0, 1.5, 2.0],  # meters
        'z_coverage': 2.5
    },
    'endcap': {
        'layers': 2,
        'z_positions': [3.0, 4.0],
        'r_coverage': 1.5
    },
    'trigger': {
        'threshold': 20.0,  # GeV
        'rate': 1000  # Hz
    }
}

# Access nested values
barrel_layers = detector_config['barrel']['layers']
trigger_threshold = detector_config['trigger']['threshold']
```

### Exercise 1.4 (30 min)

Open the afternoon notebook: [day1_afternoon_exercises.ipynb](../notebooks/day1_afternoon_exercises.ipynb)

**Beginner:**
- Create a list of detector hit positions (x, y coordinates as tuples)
- Calculate distances from the origin using a loop or list comprehension
- Find the hit closest to the origin

**Advanced:**
- Create nested dictionaries for complete detector geometry
- Implement coordinate transformations (Cartesian ↔ Cylindrical)
- Calculate which detector layer each hit corresponds to

---

## 5. Deep Dive into NumPy Arrays (1.5 hours)

### Multi-dimensional Arrays

NumPy truly shines with multi-dimensional data.

```python
import numpy as np

# 1D array (vector)
energies = np.array([10.5, 20.3, 15.7])

# 2D array (matrix)
# Each row: [px, py, pz] for one particle
momenta = np.array([
    [10.5, 5.2, 8.3],
    [20.3, -3.1, 15.7],
    [5.8, 12.4, -6.2]
])

# 3D array (e.g., detector voxels)
detector_grid = np.zeros((10, 10, 10))  # 10x10x10 grid

# Array properties
momenta.shape      # (3, 3) - 3 particles, 3 components
momenta.ndim       # 2 - number of dimensions
momenta.size       # 9 - total number of elements
momenta.dtype      # float64 - data type
```

### Array Creation Methods

```python
# Zeros and ones
zeros = np.zeros((3, 4))      # 3x4 array of zeros
ones = np.ones((2, 5))        # 2x5 array of ones
identity = np.eye(3)          # 3x3 identity matrix

# Range functions
integers = np.arange(0, 10)            # [0, 1, ..., 9]
integers = np.arange(0, 10, 2)         # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)        # [0, 0.25, 0.5, 0.75, 1]

# Random arrays
np.random.seed(42)  # For reproducibility
random = np.random.random((3, 3))      # Uniform [0, 1)
normal = np.random.normal(0, 1, 100)   # Normal distribution
```

### Array Operations (Broadcasting)

Broadcasting allows operations between arrays of different shapes:

```python
# Scalar operations (broadcast to all elements)
energies = np.array([10, 20, 30])
energies * 2        # [20, 40, 60]
energies + 5        # [15, 25, 35]

# Array-array operations (element-wise)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a + b              # [5, 7, 9]
a * b              # [4, 10, 18]

# Broadcasting with different shapes
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
vector = np.array([10, 20, 30])

# Add vector to each row of matrix
matrix + vector    # [[11, 22, 33],
                   #  [14, 25, 36]]
```

### Boolean Indexing and Masking

Powerful technique for event selection:

```python
energies = np.array([15.3, 45.7, 23.1, 67.8, 12.5, 89.2])

# Create boolean mask
high_energy = energies > 50
# Result: [False, False, False, True, False, True]

# Select elements
energies[high_energy]  # [67.8, 89.2]

# Or in one step
energies[energies > 50]

# Multiple conditions (use & and |, not 'and'/'or')
medium = (energies > 20) & (energies < 70)
energies[medium]  # [45.7, 23.1, 67.8]

# Count how many pass cuts
np.sum(high_energy)  # 2 (True = 1, False = 0)
```

### Statistical Operations

```python
data = np.array([[1, 2, 3],
                 [4, 5, 6]])

# Operations on entire array
np.mean(data)      # 3.5
np.std(data)       # ~1.71
np.sum(data)       # 21
np.min(data)       # 1
np.max(data)       # 6

# Operations along specific axis
# axis=0: down columns, axis=1: across rows
np.mean(data, axis=0)  # [2.5, 3.5, 4.5] - column means
np.mean(data, axis=1)  # [2., 5.] - row means

# Useful functions
np.median(data)
np.percentile(data, 75)  # 75th percentile
np.argmax(data)          # Index of maximum (flattened)
np.argmin(data, axis=1)  # Index of minimum in each row
```

### Common NumPy Functions

```python
# Element-wise functions
x = np.array([0, np.pi/4, np.pi/2])
np.sin(x)
np.cos(x)
np.exp(x)
np.log(x + 1)  # Natural logarithm
np.sqrt(x)
np.abs(x)

# Where (conditional selection)
energies = np.array([15, 45, 23, 67, 12])
# If energy > 50, return energy, else return 0
np.where(energies > 50, energies, 0)
# Result: [0, 0, 0, 67, 0]

# Histogramming
data = np.random.normal(50, 10, 1000)
counts, bin_edges = np.histogram(data, bins=20, range=(0, 100))
# counts: number of events in each bin
# bin_edges: edges of bins (length = bins + 1)

# Sorting
sorted_data = np.sort(energies)
sorted_indices = np.argsort(energies)  # Indices that would sort
```

### Exercise 1.5 (50 min)

**Beginner:**

1. Load simulated detector data (will be provided as `.npy` file)
2. Apply energy threshold cuts using boolean masks (e.g., E > 20 GeV)
3. Calculate basic statistics (mean, std, min, max)
4. Plot distributions before and after cuts

**Advanced:**

1. Vectorized calculation of invariant masses for all particle pairs in an event
2. Implement detector acceptance corrections using 2D efficiency maps
3. Use advanced indexing for complex event selection
4. Optimize for performance (compare vectorized vs loop approaches)

---

## 6. Introduction to Pandas DataFrames (1 hour)

### What is a DataFrame?

A DataFrame is a 2D labeled data structure, like a spreadsheet or SQL table. Perfect for event-based particle physics data!

```python
import pandas as pd

# Create DataFrame from dictionary
data = pd.DataFrame({
    'event': [1, 2, 3, 4, 5],
    'energy': [45.2, 67.8, 23.1, 89.5, 34.6],
    'pt': [30.1, 45.6, 20.3, 60.2, 25.1],
    'eta': [-0.5, 1.2, -1.8, 0.3, 2.1],
    'particle': ['e', 'mu', 'e', 'mu', 'e']
})

print(data)
```

Output:
```
   event  energy    pt   eta particle
0      1    45.2  30.1  -0.5        e
1      2    67.8  45.6   1.2       mu
2      3    23.1  20.3  -1.8        e
3      4    89.5  60.2   0.3       mu
4      5    34.6  25.1   2.1        e
```

### DataFrame Properties

```python
# Basic info
data.shape         # (5, 5) - rows, columns
data.columns       # Column names
data.index         # Row indices
data.dtypes        # Data type of each column

# Quick preview
data.head()        # First 5 rows
data.tail(3)       # Last 3 rows
data.info()        # Summary information
data.describe()    # Statistical summary (numerical columns)
```

### Selecting Data

```python
# Select column (returns Series)
data['energy']
data.energy  # Alternative (if column name has no spaces)

# Select multiple columns (returns DataFrame)
data[['energy', 'pt']]

# Select rows by index (iloc - integer location)
data.iloc[0]        # First row
data.iloc[0:3]      # First 3 rows
data.iloc[:, 0:2]   # All rows, first 2 columns

# Select by label (loc)
data.loc[0:2, 'energy':'pt']

# Boolean indexing (like NumPy masks)
high_energy = data[data['energy'] > 50]
electrons = data[data['particle'] == 'e']

# Multiple conditions
high_energy_electrons = data[(data['energy'] > 30) &
                             (data['particle'] == 'e')]
```

### Adding and Modifying Columns

```python
# Add new column (transverse energy)
data['Et'] = data['energy'] / np.cosh(data['eta'])

# Modify existing column
data['energy'] = data['energy'] * 1000  # Convert GeV to MeV

# Conditional modification
data.loc[data['particle'] == 'e', 'charge'] = -1
data.loc[data['particle'] == 'mu', 'charge'] = -1

# Apply function to column
data['energy_sqrt'] = data['energy'].apply(np.sqrt)

# Drop columns
data = data.drop(['energy_sqrt'], axis=1)
```

### Loading and Saving Data

```python
# Load from CSV
df = pd.read_csv('data/events.csv')

# Load with specific options
df = pd.read_csv('data/events.csv',
                 sep=',',           # Delimiter
                 header=0,          # Row number for column names
                 names=['event', 'E', 'p'],  # Custom column names
                 skiprows=5,        # Skip first 5 rows
                 nrows=1000)        # Read only first 1000 rows

# Save to CSV
data.to_csv('output.csv', index=False)

# Load from other formats
df = pd.read_excel('data.xlsx')
df = pd.read_json('data.json')

# For ROOT files (particle physics)
import uproot
tree = uproot.open('data.root')['tree_name']
df = tree.arrays(library='pd')  # Convert to pandas
```

### Basic Operations

```python
# Sort by column
data_sorted = data.sort_values('energy', ascending=False)

# Group by category
by_particle = data.groupby('particle')

# Aggregate functions
by_particle.mean()     # Mean energy, pt for each particle type
by_particle.count()    # Count events per particle
by_particle.sum()
by_particle.agg(['mean', 'std', 'count'])  # Multiple stats

# Filter groups
high_pt_groups = data.groupby('particle').filter(lambda x: x['pt'].mean() > 30)
```

### Merging DataFrames

```python
# Event-level data
events = pd.DataFrame({
    'event': [1, 2, 3],
    'trigger': [True, False, True]
})

# Particle-level data
particles = pd.DataFrame({
    'event': [1, 1, 2, 3, 3],
    'particle': ['e', 'mu', 'e', 'mu', 'e'],
    'pt': [30, 25, 45, 20, 35]
})

# Merge (like SQL join)
merged = pd.merge(particles, events, on='event', how='left')
# Each particle gets trigger info from its event
```

### Handling Missing Data

```python
# Check for missing values
data.isnull()           # Boolean DataFrame
data.isnull().sum()     # Count NaN per column

# Fill missing values
data.fillna(0)                    # Fill with 0
data.fillna(data.mean())          # Fill with column mean

# Drop rows with missing values
data.dropna()                     # Drop any row with NaN
data.dropna(subset=['energy'])    # Drop only if energy is NaN
```

### Exercise 1.6 (35 min)

**Beginner:**

1. Load particle collision data from CSV into a DataFrame
2. Filter events by trigger conditions
3. Calculate and add a new column for transverse momentum (pt)
4. Create simple summary statistics

**Advanced:**

1. Merge multiple DataFrames (event data + detector info)
2. Group by event number and calculate event-level quantities:
   - Total energy per event
   - Number of particles per event
   - Leading (highest pT) particle per event
3. Handle missing data and outliers
4. Create pivot tables for run-by-run statistics

---

## Integration Example

Let's combine lists, NumPy, and Pandas in a realistic analysis:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate simulated collision events
np.random.seed(42)
n_events = 1000

# Create event data
events = pd.DataFrame({
    'event': range(n_events),
    'n_particles': np.random.poisson(5, n_events),
    'trigger': np.random.choice([True, False], n_events, p=[0.7, 0.3])
})

# Apply trigger cut
triggered_events = events[events['trigger']]

# Generate particle-level data for triggered events
particle_data = []
for idx, row in triggered_events.iterrows():
    for i in range(row['n_particles']):
        particle_data.append({
            'event': row['event'],
            'energy': np.random.exponential(30),
            'eta': np.random.uniform(-2.5, 2.5),
            'phi': np.random.uniform(-np.pi, np.pi)
        })

particles = pd.DataFrame(particle_data)

# Calculate transverse energy
particles['Et'] = particles['energy'] / np.cosh(particles['eta'])

# Event selection: at least one high-Et particle
good_events = particles[particles['Et'] > 50]['event'].unique()
final_data = particles[particles['event'].isin(good_events)]

print(f"Started with {n_events} events")
print(f"After trigger: {len(triggered_events)} events")
print(f"After Et cut: {len(good_events)} events")
print(f"Final particles: {len(final_data)}")

# Visualize
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(particles['energy'], bins=50)
plt.xlabel('Energy (GeV)')
plt.ylabel('Particles')

plt.subplot(1, 3, 2)
plt.hist(particles['Et'], bins=50)
plt.xlabel('Et (GeV)')
plt.ylabel('Particles')

plt.subplot(1, 3, 3)
plt.hist2d(particles['eta'], particles['phi'], bins=30)
plt.xlabel('η')
plt.ylabel('φ')
plt.colorbar(label='Particles')

plt.tight_layout()
plt.show()
```

---

## Key Takeaways

!!! success "What We Learned"
    - **Lists**: Flexible but slower for numerical work; use list comprehensions
    - **Dictionaries**: Perfect for metadata, configurations, and structured data
    - **NumPy arrays**: Essential for efficient numerical computing
        - Boolean masking for event selection
        - Vectorized operations (100x faster than loops!)
        - Multi-dimensional arrays for complex data
    - **Pandas DataFrames**: Ideal for tabular particle physics data
        - Easy data loading, filtering, and grouping
        - SQL-like operations (merge, groupby)
        - Seamless integration with NumPy

!!! warning "Performance Tip"
    Always prefer NumPy/Pandas vectorized operations over Python loops when working with large datasets!

!!! info "Tomorrow"
    Day 2 will cover:
    - Advanced NumPy techniques (fancy indexing, broadcasting)
    - Advanced Pandas operations (MultiIndex, time series)
    - Functions and object-oriented programming for analysis code

---

## Additional Resources

- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/index.html)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Uproot Documentation](https://uproot.readthedocs.io/) (for ROOT files)
- [Awkward Array](https://awkward-array.org/) (for jagged/nested arrays in HEP)

---

**Navigate:** [← Day 1 Morning](day1_morning.md) | [Day 2 Morning →](day2_morning.md)
