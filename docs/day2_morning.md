# Day 2 - Morning Session
## Advanced Data Manipulation and Visualization

**Duration:** 3.5 hours
**Topics:** Advanced NumPy operations, Advanced Pandas techniques, Matplotlib and Seaborn visualization

---

## 1. Advanced NumPy Operations (1h 15min)

### Fancy Indexing and Advanced Slicing

NumPy allows powerful indexing beyond simple slices:

```python
import numpy as np

# Create sample array
data = np.arange(20).reshape(4, 5)
print(data)
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]
#  [15 16 17 18 19]]

# Fancy indexing: select specific rows
rows = [0, 2, 3]
print(data[rows])  # Rows 0, 2, and 3

# Select specific elements with arrays
rows = np.array([0, 1, 2])
cols = np.array([1, 3, 4])
print(data[rows, cols])  # Elements (0,1), (1,3), (2,4) → [1, 8, 14]

# Boolean indexing combined with fancy indexing
mask = data > 10
data[mask] = 0  # Set all values > 10 to 0
```

??? "Optionnal : creating 2D meshgrid"
    ### np.meshgrid for 2D Parameter Scans

    `meshgrid` creates coordinate matrices from coordinate vectors - essential for 2D calculations:

    ```python
    # Create 2D grid for detector simulation
    x = np.linspace(-2, 2, 5)   # x positions
    y = np.linspace(-1, 1, 3)   # y positions

    X, Y = np.meshgrid(x, y)
    print("X grid:\n", X)
    print("Y grid:\n", Y)

    # Calculate distance from origin at each point
    R = np.sqrt(X**2 + Y**2)
    print("Distance from origin:\n", R)

    # Example: 2D Gaussian for detector response
    def gaussian_2d(X, Y, x0, y0, sigma):
        return np.exp(-((X-x0)**2 + (Y-y0)**2) / (2*sigma**2))

    response = gaussian_2d(X, Y, 0, 0, 1.0)
    ```

### Linear Algebra Operations

NumPy provides comprehensive linear algebra support:

```python
# Vector operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Dot product
dot_product = np.dot(v1, v2)  # or v1 @ v2
print(f"v1 · v2 = {dot_product}")  # 32

# Cross product
cross_product = np.cross(v1, v2)
print(f"v1 × v2 = {cross_product}")

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = A @ B  # or np.matmul(A, B)

# Transpose
A_T = A.T

# Determinant and inverse
det_A = np.linalg.det(A)
A_inv = np.linalg.inv(A)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
```

### Physics Example: Rotation Matrices

```python
def rotation_matrix_z(theta):
    """Rotation matrix around z-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

# Rotate a 3D vector
vector = np.array([1, 0, 0])
angle = np.pi / 4  # 45 degrees

R = rotation_matrix_z(angle)
rotated = R @ vector
print(f"Original: {vector}")
print(f"Rotated 45°: {rotated}")
```

### Performance: Vectorization vs Loops

Vectorization is crucial for performance in scientific computing:

```python
import time

# Compare loop vs vectorized approach
n = 1_000_000
data = np.random.random(n)

# Method 1: Python loop (SLOW)
def sum_squared_loop(arr):
    total = 0
    for x in arr:
        total += x**2
    return total

# Method 2: Vectorized (FAST)
def sum_squared_vectorized(arr):
    return np.sum(arr**2)

# Timing comparison
start = time.time()
result_loop = sum_squared_loop(data)
time_loop = time.time() - start

start = time.time()
result_vec = sum_squared_vectorized(data)
time_vec = time.time() - start

print(f"Loop: {time_loop:.4f}s")
print(f"Vectorized: {time_vec:.4f}s")
print(f"Speedup: {time_loop/time_vec:.1f}x")
```

### Exercise 2.1 (40 min)

📓 **Open the companion notebook:** [day2_morning_exercises.ipynb](https://github.com/OpenAdalab/course_initiation_scientific_python/blob/main/notebooks/day2_morning_exercises.ipynb)

| Level | Task |
|-------|------|
| **Beginner** | Calculate angular separations between particle pairs, create 2D η-φ histograms |
| **Advanced** | Implement jet clustering with vectorized operations, benchmark performance |

---

## 2. Advanced Pandas Techniques (1h 15min)

### The `map()` Method: Dictionary Lookups

The `map()` method applies a mapping (dictionary or function) to each element of a Series. It's perfect for lookups and simple transformations:

```python
import pandas as pd
import numpy as np

# Sample detector data
df = pd.DataFrame({
    'detector': ['barrel', 'endcap', 'barrel', 'endcap', 'barrel'],
    'energy_raw': [45.2, 52.1, 38.7, 61.3, 42.8]
})

# Apply calibration factors using a dictionary
calibration = {
    'barrel': 1.02,
    'endcap': 1.05
}

df['cal_factor'] = df['detector'].map(calibration)
df['energy_cal'] = df['energy_raw'] * df['cal_factor']

print(df)
#   detector  energy_raw  cal_factor  energy_cal
# 0   barrel        45.2        1.02       46.10
# 1   endcap        52.1        1.05       54.71
# 2   barrel        38.7        1.02       39.47
# ...
```

**When to use `map()`:**

- Looking up values from a dictionary
- Simple element-wise transformations
- Converting codes to labels (e.g., particle IDs to names)

### The `groupby()` Method: Split-Apply-Combine

`groupby()` is one of the most powerful Pandas operations. It follows the **split-apply-combine** pattern:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   SPLIT     │────►│   APPLY     │────►│  COMBINE    │
│  by groups  │     │  function   │     │  results    │
└─────────────┘     └─────────────┘     └─────────────┘
```

```python
# Sample data with multiple runs and detectors
df = pd.DataFrame({
    'run': [1, 1, 1, 2, 2, 2],
    'detector': ['barrel', 'endcap', 'barrel', 'barrel', 'endcap', 'endcap'],
    'energy': [45.2, 52.1, 38.7, 61.3, 42.8, 55.9]
})

# Group by single column
by_run = df.groupby('run')

# Basic aggregations
print(by_run['energy'].mean())    # Mean energy per run
print(by_run['energy'].sum())     # Total energy per run
print(by_run['energy'].count())   # Number of events per run

# Group by multiple columns
by_run_detector = df.groupby(['run', 'detector'])
print(by_run_detector['energy'].mean())
```

### The `agg()` Method: Multiple Aggregations

`agg()` allows applying multiple aggregation functions at once:

```python
# Single column, multiple functions
stats = df.groupby('run')['energy'].agg(['mean', 'std', 'min', 'max', 'count'])
print(stats)
#       mean        std   min   max  count
# run
# 1    45.33   6.70...  38.7  52.1      3
# 2    53.33   9.42...  42.8  61.3      3

# Multiple columns, different functions
summary = df.groupby('run').agg({
    'energy': ['mean', 'std'],    # Mean and std for energy
    'detector': 'count'           # Count of detectors (= number of rows)
})
print(summary)

# Rename columns for clarity
summary.columns = ['energy_mean', 'energy_std', 'n_events']
print(summary)
```

### Common Aggregation Functions

| Function | Description | Example |
|----------|-------------|---------|
| `mean()` | Average value | Mean energy per run |
| `sum()` | Total | Total pT in event |
| `count()` | Number of non-null values | Particles per event |
| `min()` / `max()` | Minimum / Maximum | Leading particle pT |
| `std()` | Standard deviation | Energy spread |
| `first()` / `last()` | First / Last value | First particle in event |

### Finding Specific Rows with `idxmax()` / `idxmin()`

To find which row has the maximum or minimum value:

```python
# Particle data with multiple particles per event
particles = pd.DataFrame({
    'event': [1, 1, 1, 2, 2, 3],
    'particle_type': ['e', 'mu', 'photon', 'e', 'mu', 'e'],
    'pt': [45.2, 32.1, 12.5, 67.8, 23.4, 89.1]
})

# Find index of leading (highest pT) particle in each event
leading_idx = particles.groupby('event')['pt'].idxmax()
print(leading_idx)
# event
# 1    0
# 2    3
# 3    5

# Get the actual rows
leading_particles = particles.loc[leading_idx]
print(leading_particles)
#    event particle_type    pt
# 0      1             e  45.2
# 3      2             e  67.8
# 5      3             e  89.1
```

### The `transform()` Method: Same-Shape Output

Unlike `agg()` which reduces data, `transform()` returns data with the same shape as the input. Useful for adding group-level statistics back to the DataFrame:

```python
# Add mean energy per run as a new column
df['run_mean_energy'] = df.groupby('run')['energy'].transform('mean')

# Normalize energy within each run
df['energy_normalized'] = df.groupby('run')['energy'].transform(
    lambda x: (x - x.mean()) / x.std()
)

print(df)
```

### Combining Datasets with `merge()`

```python
# Event-level information
events = pd.DataFrame({
    'event_id': [1, 2, 3],
    'luminosity': [1.5e33, 1.6e33, 1.4e33],
    'trigger': [True, False, True]
})

# Particle-level information
particles = pd.DataFrame({
    'event_id': [1, 1, 2, 3, 3],
    'particle': ['e', 'mu', 'e', 'mu', 'e'],
    'pt': [30, 25, 45, 20, 35]
})

# Merge: add event info to each particle
merged = pd.merge(particles, events, on='event_id')
print(merged)

# Filter to triggered events only
triggered = merged[merged['trigger'] == True]
```

??? tip "Advanced: Pivot Tables for Summary Statistics"
    Pivot tables provide a spreadsheet-style summary across two dimensions:

    ```python
    # Basic pivot table: mean energy by run and detector
    pivot = pd.pivot_table(
        df,
        values='energy',
        index='run',
        columns='detector',
        aggfunc='mean'
    )
    print(pivot)
    #          barrel  endcap
    # run
    # 1         41.95   52.10
    # 2         61.30   49.35

    # Multiple aggregation functions
    pivot_multi = pd.pivot_table(
        df,
        values='energy',
        index='run',
        columns='detector',
        aggfunc=['mean', 'std', 'count']
    )
    ```

??? tip "Advanced: Multi-level Indexing (MultiIndex)"
    MultiIndex allows hierarchical indexing for complex data structures:

    ```python
    # Create MultiIndex DataFrame
    df_multi = df.set_index(['run', 'detector'])
    print(df_multi)

    # Access data at different levels
    print(df_multi.loc[1])              # All data from run 1
    print(df_multi.loc[(1, 'barrel')])  # Run 1, barrel only

    # Cross-section selection with xs
    print(df_multi.xs('barrel', level='detector'))  # All barrel data
    ```

??? tip "Advanced: Memory Optimization with Categorical Data"
    For large datasets with repeating string values, use categorical dtype:

    ```python
    # Convert string columns to categorical
    df['detector'] = df['detector'].astype('category')
    df['particle_type'] = df['particle_type'].astype('category')

    # Can reduce memory usage by 90%+ for string columns
    print(df.memory_usage(deep=True))
    ```

### Exercise 2.2 (45 min)

📓 **Continue in the notebook:** [day2_morning_exercises.ipynb](https://github.com/OpenAdalab/course_initiation_scientific_python/blob/main/notebooks/day2_morning_exercises.ipynb)

| Level | Task |
|-------|------|
| **Beginner** | Apply energy calibrations with `map()`, calculate statistics with `groupby()` and `agg()` |
| **Advanced** | Build hierarchical data analysis (run → event → particle), find leading particles |

---

## 3. Visualization with Matplotlib and Seaborn (1h)

### Matplotlib Architecture

Understanding the Matplotlib hierarchy:

```
Figure (container)
└── Axes (actual plot)
    ├── XAxis
    ├── YAxis
    ├── Lines, Patches, Text...
    └── Legend
```
???+ tip "A graphical summary of a figure part"
    [The different part of the figure in matplotib](https://matplotlib.org/stable/users/explain/quick_start.html#parts-of-a-figure)

```python
import matplotlib.pyplot as plt
import numpy as np

# Object-oriented approach (recommended)
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, np.cos(x), label='cos(x)')

# Customize
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Trigonometric Functions')
ax.legend()
ax.grid(True, alpha=0.3)

plt.show()
```

### Multiple Subplots

```python
# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Access individual axes
ax1 = axes[0, 0]  # Top-left
ax2 = axes[0, 1]  # Top-right
ax3 = axes[1, 0]  # Bottom-left
ax4 = axes[1, 1]  # Bottom-right

# Or with flatten
for i, ax in enumerate(axes.flatten()):
    ax.set_title(f'Subplot {i+1}')

plt.tight_layout()
plt.show()

# Unequal subplots with gridspec
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 2])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])  # Span both columns
```

### Common Physics Plots

#### Histograms with Error Bars

```python
# Generate data
data = np.random.normal(91.2, 2.5, 10000)  # Z boson mass

fig, ax = plt.subplots(figsize=(10, 6))

# Create histogram
counts, bins, _ = ax.hist(data, bins=50, range=(80, 100),
                           histtype='step', linewidth=2, label='Data')

# Add error bars
bin_centers = (bins[:-1] + bins[1:]) / 2
errors = np.sqrt(counts)
ax.errorbar(bin_centers, counts, yerr=errors, fmt='none',
            capsize=2, color='black', label='Statistical uncertainty')

ax.set_xlabel(r'$m_{\mu\mu}$ (GeV/c²)', fontsize=12)
ax.set_ylabel('Events / (0.4 GeV/c²)', fontsize=12)
ax.set_title('Dimuon Invariant Mass Distribution', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.show()
```

#### 2D Histograms

```python
# Generate correlated data
eta = np.random.uniform(-2.5, 2.5, 10000)
phi = np.random.uniform(-np.pi, np.pi, 10000)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Method 1: hist2d
h = axes[0].hist2d(eta, phi, bins=30, cmap='viridis')
plt.colorbar(h[3], ax=axes[0], label='Events')
axes[0].set_xlabel('η')
axes[0].set_ylabel('φ')
axes[0].set_title('hist2d')

# Method 2: pcolormesh (more control)
H, xedges, yedges = np.histogram2d(eta, phi, bins=30)
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
im = axes[1].pcolormesh(X, Y, H.T, cmap='viridis')
plt.colorbar(im, ax=axes[1], label='Events')
axes[1].set_xlabel('η')
axes[1].set_ylabel('φ')
axes[1].set_title('pcolormesh')

# Method 3: contour
axes[2].contourf(H.T, levels=20, cmap='viridis',
                  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
axes[2].set_xlabel('η')
axes[2].set_ylabel('φ')
axes[2].set_title('contourf')

plt.tight_layout()
plt.show()
```

### Seaborn for Statistical Plots

```python
import seaborn as sns

# Set style
sns.set_theme(style='whitegrid')

# Create sample DataFrame
np.random.seed(42)
df = pd.DataFrame({
    'pt': np.concatenate([
        np.random.exponential(30, 500),
        np.random.exponential(50, 500)
    ]),
    'particle': ['electron'] * 500 + ['muon'] * 500,
    'detector': np.random.choice(['barrel', 'endcap'], 1000)
})
```

#### Distribution Plots

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram with KDE
sns.histplot(data=df, x='pt', hue='particle', kde=True, ax=axes[0])
axes[0].set_title('Histogram with KDE')

# Box plot
sns.boxplot(data=df, x='particle', y='pt', hue='detector', ax=axes[1])
axes[1].set_title('Box Plot')

# Violin plot
sns.violinplot(data=df, x='particle', y='pt', ax=axes[2])
axes[2].set_title('Violin Plot')

plt.tight_layout()
plt.show()
```

#### Pair Plots and Joint Plots

```python
# Add more variables
df['eta'] = np.random.uniform(-2.5, 2.5, 1000)
df['energy'] = df['pt'] * np.cosh(df['eta'])

# Pair plot: all pairwise relationships
sns.pairplot(df[['pt', 'eta', 'energy', 'particle']],
             hue='particle', diag_kind='kde')
plt.show()

# Joint plot: bivariate distribution
g = sns.jointplot(data=df, x='pt', y='energy', hue='particle',
                  kind='scatter', marginal_kws=dict(fill=True))
plt.show()
```

### Publication-Quality Plots

```python
# Custom style for publications
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
})

fig, ax = plt.subplots()

# Use LaTeX-style labels
ax.set_xlabel(r'Transverse momentum $p_T$ (GeV/c)')
ax.set_ylabel(r'Events / 5 GeV/c')
ax.set_title(r'$p_T$ Distribution: Data vs Monte Carlo')

# Add text annotations
ax.text(0.05, 0.95, 'ATLAS Simulation\n' + r'$\sqrt{s}$ = 13 TeV',
        transform=ax.transAxes, verticalalignment='top',
        fontsize=11, family='sans-serif')

# Save in multiple formats
fig.savefig('plot.pdf', dpi=300, bbox_inches='tight')
fig.savefig('plot.png', dpi=300, bbox_inches='tight')
```

??? tip "To go further with interactive ploting"
    Let's have a look at the package [plotly](https://plotly.com/examples/)

### Exercise 2.3 (30 min)

📓 **Continue in the notebook:** [day2_morning_exercises.ipynb](https://github.com/OpenAdalab/course_initiation_scientific_python/blob/main/notebooks/day2_morning_exercises.ipynb)

| Level | Task |
|-------|------|
| **Beginner** | Create standard analysis plots (mass peaks, pT distributions) with labels |
| **Advanced** | Multi-panel data/MC comparison, custom styles, interactive elements |

---

## Key Takeaways

!!! success "What We Learned"
    - **Advanced NumPy**: Fancy indexing, meshgrid, linear algebra, vectorization
    - **Advanced Pandas**: MultiIndex for hierarchical data, custom functions, memory optimization
    - **Visualization**: Object-oriented Matplotlib, Seaborn for statistical plots
    - **Performance**: Vectorized operations are 10-100x faster than loops

!!! warning "Best Practices"
    - Always use vectorized operations when possible
    - Use categorical dtypes for memory efficiency
    - Create reusable plotting functions for consistency
    - Save figures in vector format (PDF) for publications

??? info "This Afternoon"
    Day 2 Afternoon will cover:

    - Writing reusable analysis functions
    - Object-oriented programming for physics analysis
    - Building a complete analysis workflow

---

## Additional Resources

- [NumPy Advanced Indexing](https://numpy.org/doc/stable/user/basics.indexing.html)
- [Pandas MultiIndex Guide](https://pandas.pydata.org/docs/user_guide/advanced.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Scientific Visualization Book](https://github.com/rougier/scientific-visualization-book)

---

**Navigate:** [← Day 1 Afternoon](day1_afternoon.md) | [Day 2 Afternoon →](day2_afternoon.md)
