# CEHSNF

## Complexity-Aware Explainable Hybrid Symbolic-Neural Framework

CEHSNF is a research framework for learning interpretable mathematical models of dynamical systems by combining **neural networks, symbolic regression, physics-informed learning, explainability, and complexity analysis**.

The framework follows a hybrid workflow in which neural models provide flexible function approximation while symbolic regression extracts interpretable mathematical relationships from the learned behavior or residual errors.

---

## Overview

Traditional neural networks can achieve strong predictive performance but often provide limited mathematical interpretability.

Symbolic regression can produce interpretable equations, but finding accurate expressions directly from complex nonlinear systems can be difficult.

CEHSNF combines these approaches:

```text
Physical System
      │
      ▼
Numerical Simulation
      │
      ▼
Synthetic Dataset
      │
      ├───────────────┐
      ▼               ▼
 Neural Model     Symbolic Regression
      │               │
      └───────┬───────┘
              ▼
       Hybrid Correction
              │
              ▼
   Prediction + Equation
              │
       ┌──────┴──────┐
       ▼             ▼
 Explainability   Complexity
       │             │
       └──────┬──────┘
              ▼
       Benchmark Analysis
```

---

## Key Components

### 1. Physical-System Simulation

The framework currently provides canonical dynamical-system datasets for:

* Harmonic oscillator
* Nonlinear pendulum
* Damped oscillator
* Coupled double pendulum
* Lorenz system

Each system produces physically meaningful input features and target dynamics using numerical simulation.

---

### 2. Neural Model

The neural component uses a fully connected feed-forward network with:

* Three hidden layers
* Tanh activations
* 64 hidden units
* Mean Squared Error loss
* Adam optimization
* Early stopping
* Best-model checkpoint restoration

The neural model provides a flexible baseline for learning nonlinear system dynamics.

---

### 3. Symbolic Regression

CEHSNF uses **PySR** for symbolic regression.

The symbolic component searches for mathematical expressions using configurable:

* Binary operators
* Unary operators
* Maximum expression size
* Number of evolutionary iterations
* Parsimony pressure
* Model-selection strategy

Example discovered equations can be represented in forms such as:

```text
x0 * -1.0
```

or more complex expressions involving trigonometric and arithmetic operations.

---

### 4. Hybrid Neural-Symbolic Model

The hybrid model learns a symbolic correction to the neural prediction.

Given:

```text
Neural prediction = f_neural(X)
```

the residual is defined as:

```text
Residual = y - f_neural(X)
```

Symbolic regression then learns:

```text
f_symbolic(X) ≈ Residual
```

The final prediction becomes:

```text
f_hybrid(X) =
    f_neural(X) + f_symbolic(X)
```

This allows the neural network to model complicated behavior while symbolic regression captures interpretable corrections.

---

### 5. Physics-Informed Neural Network

CEHSNF also contains a physics-informed neural-network component.

The PINN combines:

```text
Total Loss =
    Data Loss
    + λ × Physics Loss
```

The current implementation demonstrates this approach on the coupled double-pendulum system.

---

### 6. Explainability

The framework includes model-agnostic sensitivity analysis.

For each input feature, the method perturbs the feature slightly and measures the resulting change in model output.

This produces:

```text
Feature
   │
   ▼
Perturb input
   │
   ▼
Evaluate model
   │
   ▼
Measure output change
   │
   ▼
Sensitivity score
```

Features can then be ranked according to their influence on the prediction.

---

### 7. Complexity Analysis

Symbolic expressions are analyzed using structural properties such as:

* PySR complexity
* Operator count
* Variable count
* Constant count
* Expression length

The purpose is to study the trade-off between:

```text
Prediction Accuracy
        ↕
Equation Complexity
```

The framework preserves PySR's native complexity as the primary symbolic complexity measure when available.

---

## Repository Structure

```text
CEHSNF/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   └── cehsnf/
│       ├── __init__.py
│       │
│       ├── systems/
│       │   ├── harmonic.py
│       │   ├── pendulum.py
│       │   ├── damped.py
│       │   ├── double_pendulum.py
│       │   └── lorenz.py
│       │
│       ├── models/
│       │   ├── neural.py
│       │   ├── symbolic.py
│       │   ├── hybrid.py
│       │   └── pinn.py
│       │
│       ├── explainability/
│       │   └── sensitivity.py
│       │
│       ├── evaluation/
│       │   ├── metrics.py
│       │   └── complexity.py
│       │
│       └── data/
│           ├── synthetic.py
│           └── nasa_fd001.py
│
├── experiments/
│   ├── run_benchmarks.py
│   ├── run_double_pendulum.py
│   ├── run_lorenz.py
│   └── run_fd001.py
│
├── analysis/
│   ├── generate_graphs.py
│   └── generate_tables.py
│
├── results/
│   └── .gitkeep
│
├── docs/
│   ├── methodology.md
│   ├── benchmark_protocol.md
│   └── nasa_fd001.md
│
└── tests/
```

> **Note:** The NASA FD001 module is planned as an extension and is not part of the current validated benchmark workflow.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/CEHSNF.git
cd CEHSNF
```

Create a virtual environment:

### Windows

```cmd
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## PySR / Julia

CEHSNF uses PySR for symbolic regression.

PySR requires a Julia runtime. After installing the Python dependencies, initialize the PySR environment if necessary:

```bash
python -c "import pysr; pysr.install()"
```

Verify the installation:

```bash
python -c "import pysr; print('PySR installed successfully')"
```

---

## Running the Framework

Set the source directory on Windows:

```cmd
set PYTHONPATH=%CD%\src
```

On Linux/macOS:

```bash
export PYTHONPATH=$PWD/src
```

The individual modules can then be tested independently.

For example:

```bash
python -m cehsnf.data.synthetic
```

Neural model:

```bash
python -m cehsnf.models.neural
```

Symbolic regression:

```bash
python -m cehsnf.models.symbolic
```

Hybrid model:

```bash
python -m cehsnf.models.hybrid
```

PINN:

```bash
python -m cehsnf.models.pinn
```

Sensitivity analysis:

```bash
python -m cehsnf.explainability.sensitivity
```

Metrics:

```bash
python -m cehsnf.evaluation.metrics
```

Complexity analysis:

```bash
python -m cehsnf.evaluation.complexity
```

---

## Evaluation Metrics

The framework currently supports:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R² Score
* Residual statistics

Example:

```text
MSE
RMSE
MAE
R²
```

These metrics can be used to compare:

```text
Neural
Symbolic
Hybrid
PINN
```

models on the same dataset.

---

## Benchmark Systems

### Harmonic Oscillator

A simple dynamical system used as a baseline for validating the framework.

### Nonlinear Pendulum

A nonlinear pendulum system that introduces trigonometric dynamics.

### Damped Oscillator

A damped dynamical system used to evaluate learning under dissipative behavior.

### Coupled Double Pendulum

A nonlinear coupled system with interaction between two pendulum states.

The canonical feature representation is:

```text
theta1
omega1
theta2
omega2
sin(theta2 - theta1)
cos(theta2 - theta1)
```

### Lorenz System

A nonlinear chaotic dynamical system used to evaluate the framework on more complex dynamics.

---

## Research Workflow

The intended research workflow is:

### Step 1 — Generate Physical Data

Generate numerical observations from the governing dynamical system.

### Step 2 — Train Neural Model

Train the neural network to approximate the target dynamics.

### Step 3 — Perform Symbolic Regression

Search for interpretable mathematical expressions.

### Step 4 — Learn Hybrid Correction

Fit symbolic regression to the neural residual.

### Step 5 — Evaluate

Compare predictive performance using standardized metrics.

### Step 6 — Explain

Analyze feature sensitivity and model behavior.

### Step 7 — Analyze Complexity

Study the relationship between symbolic expression complexity and predictive accuracy.

---

## Current Validation

The individual modules have been tested independently during development.

Examples include:

```text
Synthetic benchmark generation
✓ Harmonic oscillator
✓ Nonlinear pendulum
✓ Damped oscillator
✓ Coupled double pendulum
✓ Lorenz system

Neural model
✓ Training
✓ Prediction
✓ Metrics
✓ Residual generation

Symbolic regression
✓ PySR integration
✓ Equation extraction
✓ Prediction
✓ Complexity extraction

Hybrid model
✓ Neural residual generation
✓ Symbolic correction
✓ Hybrid prediction

PINN
✓ Training
✓ Physics loss
✓ Prediction
✓ Sensitivity analysis

Explainability
✓ Feature sensitivity
✓ Sensitivity ranking

Evaluation
✓ MSE
✓ RMSE
✓ MAE
✓ R²
✓ Residual statistics

Complexity
✓ Equation extraction
✓ Operator analysis
✓ Variable analysis
✓ Constant analysis
✓ Expression length
```

---

## Reproducibility

The framework is designed so that physical-system simulation, model training, symbolic regression, evaluation, and explainability can be run as independent components.

Random seeds are used where applicable to improve reproducibility.

Experimental results should be stored under:

```text
results/
```

Generated outputs and local datasets should not be committed to version control.

---

## Future Work

Planned extensions include:

* NASA C-MAPSS FD001 predictive-maintenance validation
* Larger benchmark experiments
* Improved physics-informed loss formulations
* Automated benchmark orchestration
* More detailed equation-complexity analysis
* Accuracy-versus-complexity visualization
* Additional dynamical systems
* Extended symbolic search strategies
* Automated experiment reporting

---

## Project Status

**Current stage:** Research framework reconstruction and modularization.

The repository is being rebuilt from the original CEHSNF research implementation into a cleaner, reproducible, modular architecture while preserving the core methodology.

---

## License

This project is released under the MIT License.

See `LICENSE` for details.

---

## Author

**Sarath B Raam**

Artificial Intelligence and Machine Learning
SRM Institute of Science and Technology, Ramapuram

---

## Citation

If you use CEHSNF in academic work, please cite the associated research publication when available.
