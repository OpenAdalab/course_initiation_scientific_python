# Introduction to Programming using Python

A 3-day training course for particle physics researchers with heterogeneous programming backgrounds.

## Overview

This course provides a hands-on introduction to Python programming with a focus on scientific computing and data analysis for particle physics applications.

**Duration:** 3 days (21 hours)
**Format:** Theory (30%) + Practice (70%)
**Language:** English

## Course Structure

### Day 1: Foundations and Scientific Python Ecosystem

| Session | Topics |
|---------|--------|
| **Morning** | Programming fundamentals, Python syntax, Scientific Python ecosystem (NumPy, Pandas, Matplotlib) |
| **Afternoon** | Lists and data structures, NumPy arrays, Pandas DataFrames |

### Day 2: Advanced Data Manipulation and Code Structuring

| Session | Topics |
|---------|--------|
| **Morning** | Advanced NumPy (vectorization, broadcasting), Advanced Pandas (MultiIndex, groupby), Visualization with Matplotlib/Seaborn |
| **Afternoon** | Functions for code organization, Object-Oriented Programming, Integration exercises |

### Day 3: Error Handling, Testing, and Final Project

| Session | Topics |
|---------|--------|
| **Morning** | Understanding and handling errors, Testing scientific code with pytest, Debugging tools |
| **Afternoon** | **Final Project:** Discovering a resonance in dimuon events |

## Getting Started

### Prerequisites

- Basic familiarity with any programming language (optional)
- Anaconda or Miniconda installed
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone https://github.com/OpenAdalab/course_initiation_scientific_python.git
cd course_initiation_scientific_python

# Create conda environment (recommended)
conda create -n python-physics python=3.10
conda activate python-physics

# Install dependencies
pip install numpy pandas matplotlib seaborn scipy pytest jupyter
```

### Running the Course Website

```bash
# Install mkdocs
pip install mkdocs mkdocs-material

# Serve locally
mkdocs serve

# Build static site
mkdocs build
```

## Repository Structure

```
.
├── docs/                          # Course lesson pages (Markdown)
│   ├── index.md                   # Course home page
│   ├── day1_morning.md            # Day 1 morning lesson
│   ├── day1_afternoon.md          # Day 1 afternoon lesson
│   ├── day2_morning.md            # Day 2 morning lesson
│   ├── day2_afternoon.md          # Day 2 afternoon lesson
│   ├── day3_morning.md            # Day 3 morning lesson
│   └── day3_afternoon.md          # Day 3 afternoon (final project)
├── notebooks/                     # Jupyter exercise notebooks
│   ├── day1_morning_exercises.ipynb
│   ├── day1_afternoon_exercises.ipynb
│   ├── day2_morning_exercises.ipynb
│   ├── day2_afternoon_exercises.ipynb
│   ├── day3_morning_exercises.ipynb
│   └── day3_afternoon_exercises.ipynb
├── assets/                        # Course materials and syllabus
├── mkdocs.yml                     # MkDocs configuration
└── README.md                      # This file
```

## Learning Outcomes

By the end of this training, participants will be able to:

1. **Work efficiently with scientific data**
   - Manipulate large datasets using NumPy arrays
   - Process tabular physics data with Pandas
   - Visualize results with Matplotlib/Seaborn

2. **Structure analysis code professionally**
   - Write modular, reusable functions
   - Design classes for physics objects (Particle, Event)
   - Organize complex analysis workflows

3. **Ensure code reliability**
   - Handle errors gracefully in production pipelines
   - Write effective tests for numerical code
   - Debug and profile performance issues

4. **Apply Python to real particle physics problems**
   - Load and process detector data
   - Implement event selection and kinematic calculations
   - Perform statistical analysis and visualization

## License

This course is distributed under a [Creative Commons CC-BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en) (Attribution of the author, non commercial use and sharing under the same conditions), please respect these distribution conditions 🙏 

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
