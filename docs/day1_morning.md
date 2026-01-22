# Day 1 - Morning Session
## Foundations, Data Structures, and Scientific Python Ecosystem

**Topics:** Programming fundamentals, Python syntax, Scientific Python ecosystem introduction

---

## 1. Systems, Programs and Languages (1 hour)

### Basics of Computers and Systems Architecture

Before writing code, it's essential to understand how a computer works at a fundamental level.

#### The Von Neumann Architecture

Most modern computers follow the **Von Neumann architecture**, which consists of:

```
┌─────────────────────────────────────────────────────────────┐
│                         COMPUTER                            │
│  ┌─────────────┐    ┌─────────────────────────────────┐    │
│  │             │    │           MEMORY (RAM)          │    │
│  │    CPU      │◄──►│  ┌───────────┬───────────────┐  │    │
│  │             │    │  │  Program  │     Data      │  │    │
│  │ ┌─────────┐ │    │  │Instructions│   Variables  │  │    │
│  │ │  ALU    │ │    │  └───────────┴───────────────┘  │    │
│  │ │(compute)│ │    └─────────────────────────────────┘    │
│  │ └─────────┘ │                    ▲                      │
│  │ ┌─────────┐ │                    │                      │
│  │ │ Control │ │                    ▼                      │
│  │ │  Unit   │ │    ┌─────────────────────────────────┐    │
│  │ └─────────┘ │    │     STORAGE (Disk/SSD)          │    │
│  └─────────────┘    │  Files, Databases, Programs     │    │
│                     └─────────────────────────────────┘    │
│         ▲                           ▲                      │
│         │                           │                      │
│         ▼                           ▼                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              INPUT / OUTPUT DEVICES                 │   │
│  │     Keyboard, Mouse, Screen, Network, Sensors...   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Key components:**

| Component | Role | Scientific Example |
|-----------|------|-------------------|
| **CPU** (Central Processing Unit) | Executes instructions, performs calculations | Computing particle trajectories |
| **RAM** (Random Access Memory) | Fast, temporary storage for running programs | Holding event data during analysis |
| **Storage** (Disk/SSD) | Permanent data storage | Storing experimental datasets |
| **I/O Devices** | Communicate with the outside world | Reading detector signals |

#### Binary: The Language of Computers

Computers only understand **binary** (0s and 1s). Everything—numbers, text, images, scientific data—is encoded in binary:

```
Decimal     Binary          What it could represent
-------     ------          ----------------------
0           0000            False, empty, zero
1           0001            True, one particle detected
5           0101            Number of hits in detector
255         11111111        Maximum value in 8 bits
```

??? info "Why Binary?"
    Electronic circuits have two stable states: ON (voltage present) and OFF (no voltage). These map naturally to 1 and 0.

### What is a Programming Language?

A **programming language** is a formal way to communicate instructions to a computer. Since computers only understand binary, we need a bridge between human thinking and machine execution.

#### The Abstraction Ladder

```
┌─────────────────────────────────────────────────────────┐
│  Human Thought                                          │
│  "Calculate the average energy of all particles"       │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│  High-Level Language (Python)                           │
│  average = np.mean(energies)                           │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Low-Level Language (Assembly)                          │
│  LOAD R1, energies_ptr                                 │
│  LOOP: ADD R2, [R1]                                    │
│        INC R1 ...                                      │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Machine Code (Binary)                                  │
│  10110001 00101100 11010010 01001011...               │
└─────────────────────────────────────────────────────────┘
```

**High-level languages** like Python are:

- **Readable**: Closer to human language
- **Portable**: Run on different computers
- **Productive**: Less code to write
- **Abstract**: Hide hardware details

### Algorithms

An **algorithm** is a step-by-step, non ambiguous, procedure to solve a problem. It's a sort of "recipe" that tells us what to do, independent of any programming language.

#### Example: Finding the Maximum Energy

**Problem:** Find the highest energy particle in a detector.

**Algorithm (in plain English):**

1. Start with the first particle's energy as the "current maximum"
2. For each remaining particle:
   - If its energy is greater than the current maximum:
     - Update the current maximum
3. Return the current maximum

**Flowchart representation:**

```
        ┌─────────────┐
        │   START     │
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │ max = E[0]  │
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │ i = 1       │
        └──────┬──────┘
               ▼
      ┌────────┴────────┐
      │  i < n_particles?│
      └────────┬────────┘
          yes  │  no
               ▼  ─────────────────┐
      ┌────────┴────────┐          │
      │  E[i] > max ?   │          │
      └────────┬────────┘          │
          yes  │  no               │
               ▼                   │
        ┌─────────────┐            │
        │ max = E[i]  │            │
        └──────┬──────┘            │
               │◄──────────────────┤
               ▼                   │
        ┌─────────────┐            │
        │   i = i + 1 │            │
        └──────┬──────┘            │
               │───────────────────┘
               ▼
        ┌─────────────┐
        │ return max  │
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │    END      │
        └─────────────┘
```

**Implementation in Python:**

```python
def find_max_energy(energies):
    """Find the maximum energy in a list of particles."""
    max_energy = energies[0]
    for energy in energies[1:]:
        if energy > max_energy:
            max_energy = energy
    return max_energy

# Or simply using NumPy:
max_energy = np.max(energies)
```

!!! tip "Algorithm vs Implementation"
    The same algorithm can be implemented in different programming languages. What matters is that the logic is correct!

### Compilers and Interpreters

How does human-readable code become machine-executable instructions?

#### Compiled Languages (C, C++, Fortran, Rust)

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│ Source Code│────►│  Compiler  │────►│ Executable │
│  (.c, .cpp)│     │            │     │  (binary)  │
└────────────┘     └────────────┘     └────────────┘
                                            │
                                            ▼
                                      ┌────────────┐
                                      │  Execution │
                                      └────────────┘
```

**Characteristics:**

- ✅ Fast execution (optimized machine code)
- ✅ Errors caught before running
- ❌ Slower development cycle (compile → run → debug)
- ❌ Platform-specific executables

#### Interpreted Languages (Python, R, JavaScript)
```
┌────────────┐     ┌────────────┐
│ Source Code│────►│Interpreter │────► Output
│   (.py)    │     │  (python)  │
└────────────┘     └────────────┘
                   Line by line
```

**Characteristics:**

- ✅ Fast development (write → run immediately)
- ✅ Interactive exploration (Jupyter notebooks!)
- ✅ Platform independent
- ❌ Slower execution
- ❌ Some errors only found at runtime

#### Python: A Hybrid Approach

Python actually uses both compilation and interpretation:

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│ Source Code│────►│  Compiler  │────►│  Bytecode  │
│   (.py)    │     │            │     │   (.pyc)   │
└────────────┘     └────────────┘     └────────────┘
                                            │
                                            ▼
                                      ┌────────────┐
                                      │Python VM   │────► Output
                                      │(Interpreter)│
                                      └────────────┘
```

Python compiles your code to **bytecode** (intermediate representation), then the Python Virtual Machine (PVM) interprets this bytecode.

### Typology of Programming Languages

Programming languages can be classified in several ways:

#### By Abstraction Level

| Level | Examples | Characteristics |
|-------|----------|-----------------|
| **Low-level** | Assembly, Machine Code | Direct hardware control, fast, hard to write |
| **Mid-level** | C, C++, Rust | Balance of control and abstraction |
| **High-level** | Python, R, MATLAB | Easy to write, slower execution |

#### By Programming Paradigm

| Paradigm | Description | Example in Python |
|----------|-------------|-------------------|
| **Procedural** | Sequence of instructions | `for i in range(10): print(i)` |
| **Object-Oriented** | Data and behavior in objects | `particle.calculate_momentum()` |
| **Functional** | Functions as first-class citizens | `list(map(np.sqrt, energies))` |

Python supports **multiple paradigms**, making it very flexible!

#### By Typing System

| Type | Description | Example |
|------|-------------|---------|
| **Static typing** | Types checked at compile time | C++, Java, Rust |
| **Dynamic typing** | Types checked at runtime | Python, JavaScript |
| **Strong typing** | No implicit type conversion | Python: `"5" + 5` → Error |
| **Weak typing** | Implicit type conversion | JavaScript: `"5" + 5` → `"55"` |

Python is **dynamically typed** but **strongly typed**.

### Examples in Various Languages

The same task ( calculating kinetic energy $E_k = \frac{1}{2}mv^2$ ) in different languages:

**Python:**
```python
def kinetic_energy(mass, velocity):
    return 0.5 * mass * velocity ** 2

E = kinetic_energy(1.67e-27, 1e6)  # proton at 1 km/s
```

**C:**
```c
double kinetic_energy(double mass, double velocity) {
    return 0.5 * mass * velocity * velocity;
}
```

**Fortran:**
```fortran
REAL FUNCTION KINETIC_ENERGY(MASS, VELOCITY)
    REAL, INTENT(IN) :: MASS, VELOCITY
    KINETIC_ENERGY = 0.5 * MASS * VELOCITY**2
END FUNCTION
```

**R:**
```r
kinetic_energy <- function(mass, velocity) {
    0.5 * mass * velocity^2
}
```

???+ tip "Why Python for Scientific Computing ?"
    Python has become the standard language in scientific computing because:

    - **Multiple paradigms** : it supports OOP, fonctionnal and procedural 
    - Both **dynamically typed** & **strongly typed**
    - **Interpreted langage**: Make code testing more easy
    - **Glue language**: Easy integration with C/C++/Fortran for performance
    - **Readability**: Clear syntax facilitates code writing & collaboration  
    - **Open Source** & **portable**
    - **Community**: Vast ecosystem of packages
    classic **scientific libraries**: (NumPy, SciPy, Matplotlib, ...) for numerical analysis and **domain specific** one's (AstroPy, BioPython...)
    - **Jupyter**: Interactive notebooks for reproducible analysis

## 2. Python composants & set up 

### Virtual Environments

A virtual environment is an **isolated workspace** in which installed packages do not interfere with other packages. This allows for more efficient package management, for example, by **often avoiding conflicts that can arise between different packages installed** on your system.

#### Popular Virtual Environment Tools

| Tool | Description | Best For |
|------|-------------|----------|
| **venv** | Built-in Python module (since Python 3.3) | Simple projects, standard library only |
| **virtualenv** | Third-party, more features than venv | Cross-Python version support |
| **conda** | Package + environment manager from Anaconda | Scientific computing, complex dependencies |
| **mamba** | Fast drop-in replacement for conda | Large environments, faster solving |
| **pipenv** | Combines pip + virtualenv with Pipfile | Application development |
| **poetry** | Modern dependency management + packaging | Publishing packages, reproducibility |
| **uv** | Ultra-fast Python package installer (Rust-based) | Speed-focused workflows |


#### Basic Usage Examples

**venv (built-in):**
```bash
# Create environment
python -m venv myenv

# Activate (Linux/Mac)
source myenv/bin/activate

# Activate (Windows)
myenv\Scripts\activate

# Deactivate
deactivate
```

**conda:**
```bash
# Create environment with specific Python version
conda create -n myenv python=3.11

# Activate
conda activate myenv

# Install packages
conda install numpy pandas matplotlib

# Deactivate
conda deactivate
```

### Package Managers

A **package manager** is a tool that automates the process of installing, updating, configuring, and removing software packages. In Python, packages are libraries or modules that extend Python's functionality.

#### Why Use a Package Manager?

Without a package manager, you would need to:

1. Find the package source code manually
2. Download it
3. Handle all dependencies (other packages it needs)
4. Compile it if necessary
5. Install it in the right location

A package manager does all of this with a single command!

#### Popular Python Package Managers

| Tool | Repository | Command Example | Notes |
|------|------------|-----------------|-------|
| **pip** | PyPI (Python Package Index) | `pip install numpy` | Standard Python package manager |
| **conda** | Anaconda/conda-forge | `conda install numpy` | Handles Python + non-Python dependencies |
| **mamba** | Anaconda/conda-forge | `mamba install numpy` | Faster alternative to conda |
| **uv** | PyPI | `uv pip install numpy` | Ultra-fast pip replacement |

#### pip: The Standard Package Manager

**pip** (Pip Installs Packages) is the default package manager for Python. It installs packages from [PyPI](https://pypi.org/) (Python Package Index), which hosts over 500,000 packages.

```bash
# Install a package
pip install numpy

# Install a specific version
pip install numpy==1.24.0

# Install multiple packages
pip install numpy pandas matplotlib

# Install from a requirements file
pip install -r requirements.txt

# Upgrade a package
pip install --upgrade numpy

# Uninstall a package
pip uninstall numpy

# List installed packages
pip list

# Show package information
pip show numpy
```

#### The requirements.txt File

A `requirements.txt` file lists all packages needed for a project, making it easy to recreate the environment:

```text
# requirements.txt
numpy==1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy
```

Create it automatically from your current environment:
```bash
pip freeze > requirements.txt
```

#### pip vs conda

| Aspect | pip | conda |
|--------|-----|-------|
| **Source** | PyPI only | Anaconda + conda-forge |
| **Language** | Python packages only | Any language (Python, R, C...) |
| **Dependencies** | Python dependencies only | System libraries too |
| **Compilation** | May need compiler | Pre-compiled binaries |
| **Environment** | Needs venv/virtualenv | Built-in environment management |

??? warning "Don't mix pip and conda carelessly"
    When using conda environments, prefer `conda install` when possible. If a package is only on PyPI, use `pip install` but be aware it may cause conflicts. A good practice:

    1. Install as much as possible with conda first
    2. Then use pip only for packages not available in conda

#### PyPI: The Python Package Index

[PyPI](https://pypi.org/) is the official repository for Python packages. Key facts:

- Over **500,000 packages** available
- Anyone can publish packages
- Hosts both source distributions and pre-built wheels
- Search packages at: https://pypi.org/

```
┌─────────────────────────────────────────────────────────────┐
│                        PyPI (pypi.org)                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│  │  NumPy  │ │ Pandas  │ │ SciPy   │ │500k more│            │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘            │
└─────────────────────────┬───────────────────────────────────┘
                          │ pip install
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Your Python Environment                   │
└─────────────────────────────────────────────────────────────┘
```
!!! tip "Recommendation for Scientists"
    For scientific computing, **[conda](https://docs.conda.io/en/latest/)** (or **[mamba](https://mamba.readthedocs.io/en/latest/index.html)**) are often the best choices because they will serve both as **virtual environnement tools** and **package manager tools**. Furthermore, they come with pre-built scientific package

### Installing python via anaconda (recommended)
Anaconda is a company specializing in data science. It publishes an open-source, cross-platform **toolkit** containing numerous Python packages essential science. This solution is **widely used by scientists** to build, distribute, install, and maintain software !

!!! tip "[Anaconda vs miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main#should-i-install-miniconda-or-anaconda-distribution)"
    - Install [anaconda](https://www.anaconda.com/download) if you want a fully operating ecosystem (involving, data tools, visualisation, machine learning, and front-end developpement)
    - Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) if you want just essential scientific packages

??? success "Once installed"
    You can use conda in command line (in a shell) or with a graphical interface (anaconda-navigator)

### Jupyter Notebooks 

The [Jupyter project](https://jupyter.org/) is open-source software based on open standards (e.g., Markdown, Python, etc.) for interactive computing in a dozen programming languages, including Python!

Jupyter notebooks (a key composant of this project) is **web application** that allows to:
- Mix code, text, equations, and plots in one document
- Run code cells individually for testing
- Create reproducible analysis workflows

??? success "Starting a notebook"
    - with command line :
    ``` bash
    jupyter-notebook
    ```

    - with GUI : lauch it with anconda navigator

### Exercise 1.1 (20 min)

📓 **Open the companion notebook:** [day1_morning_exercises.ipynbhttps://github.com/OpenAdalab/course_initiation_scientific_python/blob/main/notebooks/day1_morning_exercises.ipynb)

| Level | Task |
|-------|------|
| **Beginner** | Write an algorithm to calculate an average, then implement it in Python |
| **Advanced** | Create a temperature converter with input validation and error handling |

---

## 3. Python Syntax and Basic Operations (1 hour)

This section covers variables, data types, and operators. **All examples and exercises are in the notebook.**

### Key Concepts

- **Variables**: Use descriptive names (`particle_energy`, not `x`)
- **Numeric types**: `int`, `float`, `complex`
- **Operators**: Arithmetic (`+`, `-`, `*`, `/`, `**`), comparison (`>`, `==`, `<=`)
- **Math module**: `math.sqrt()`, `math.cos()`, `math.exp()`

### Exercise 1.2 (25 min)

📓 **Continue in the notebook:** [day1_morning_exercises.ipynb](https://github.com/OpenAdalab/course_initiation_scientific_python/blob/main/notebooks/day1_morning_exercises.ipynb)

| Level | Task |
|-------|------|
| **Beginner** | Calculate invariant mass: $M = \sqrt{(E_1 + E_2)^2 - (p_1 + p_2)^2}$ |
| **Advanced** | Build a relativistic kinematics calculator class with β, γ, and validation |

---

## 4. Introduction to the Scientific Python Ecosystem (1h 45min)

### The Scientific Stack

```
┌─────────────────────────────────────┐
│     Your Analysis Code              │
├─────────────────────────────────────┤
│  Pandas  │ Matplotlib │  SciPy      │
├──────────┴────────────┴─────────────┤
│          NumPy (Foundation)         │
├─────────────────────────────────────┤
│      Python Standard Library        │
└─────────────────────────────────────┘
```

### Overview of key packages

| Package | Purpose | Key Features |
|---------|---------|--------------|
| **[NumPy](https://numpy.org/)** | Numerical arrays | Fast arrays, vectorization, broadcasting |
| **[SciPy](https://scipy.org/)** | Scientific algorithms | Fitting, optimization, statistics, integration |
| **[Pandas](https://pandas.pydata.org/)** | Tabular data | DataFrames, data loading, grouping, merging |
| **[Matplotlib](https://matplotlib.org/)** | Visualization | Plots, publication-quality figures |
| **[Seaborn](https://seaborn.pydata.org/)** | Statistical visualization | Panda's optimized plots for publication-quality figures |


### Exercise 1.3 (45 min)

📓 **Continue in the notebook:** [day1_morning_exercises.ipynb](https://github.com/OpenAdalab/course_initiation_scientific_python/blob/main/notebooks/day1_morning_exercises.ipynb)

| Level | Task |
|-------|------|
| **Beginner** | Generate random data, calculate statistics, create histogram |
| **Advanced** | Fit Gaussian distribution, create publication-quality plot with residuals |

---

## Key Takeaways

!!! success "What We Learned"
    - Python is the standard for scientific analysis
    - **NumPy arrays** are the foundation for numerical computing
    - The **scientific stack** (NumPy, SciPy, Matplotlib, Pandas) provides all tools needed
    - **Jupyter notebooks** enable interactive, reproducible analysis
    - Vectorized operations are much faster than loops

---

## Additional Resources

- [Python official Documentation](https://docs.python.org/3/)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [SciPy Tutorial](https://docs.scipy.org/doc/scipy/tutorial/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [Scikit-HEP Project](https://scikit-hep.org/)
- [Python for Particle Physics (Tutorial)](https://github.com/hsf-training/PyHEP-resources)

---

**Navigate:** [Day 1 Afternoon →](day1_afternoon.md)
