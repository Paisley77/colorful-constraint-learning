# The Color of Constraints: A Geo-Semantic HSV Interface for Inverse Constrained Reinforcement Learning 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"The Color of Constraints"** — a novel neurosymbolic framework for Inverse Constrained Reinforcement Learning (ICRL) that learns interpretable spatio-temporal constraints by projecting high-dimensional states into a structured HSV color manifold.

## Overview

A fundamental limitation in current neurosymbolic AI is the *broken interface* between neural perception and symbolic reasoning. Existing ICRL methods use flat predicate vectors that:
- **Entangle semantic concepts** without clear separation
- **Lack temporal coherence** and robustness to noise
- **Rely on rigid logical formalisms** with limited expressivity

This project introduces a **geometric-semantic interface** that transforms raw state trajectories into a structured HSV color space where constraint manifolds become visually apparent and temporally rich patterns can be learned efficiently.

![HSV Manifold Visualization](assets/manifold_2d.png)
*Expert (blue) and violator (red) trajectories naturally separate in the learned HSV semantic space*

## Methodology

### Core Innovation: The HSV Semantic Interface

We bridge the neurosymbolic gap through a principled geometric approach:

#### 1. **Symbolic Concept Projection**
A parameterized neural network `g_θ` maps raw states to concept activation vectors:
```math
𝐚_t = g_θ(s_t) = MLP_θ(CNN(s_t))
```

#### 2. **Structured Color Embedding**
We project concept vectors into an HSV-inspired latent space using PCA-based transformations:

- **Hue**: Primary semantic category (circular, continuous)
- **Saturation**: Confidence of semantic assignment  
- **Value**: Semantic prominence or significance

Let $\mathbf{a}_t \in \mathbb{R}^k$ be the concept vector at time $t$. After PCA decomposition $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$, we project:

$$
\mathbf{p}_t = \mathbf{V}^T(\mathbf{a}_t - \boldsymbol{\mu})
$$

The complete mapping $\phi: \mathbb{R}^k \to \mathbb{H} \subset \mathbb{R}^3$ is defined as:

$$
\phi(\mathbf{a}_t) = \begin{pmatrix}
\dfrac{\text{atan2}(p_{t,2}, p_{t,1}) + 2\pi}{2\pi} \mod 1 \\[1em]
\sqrt{\sum_{i=1}^k \dfrac{p_{t,i}^2}{\lambda_i}} \\[1em]
\tanh\left(\dfrac{\|\mathbf{a}_t\|_2}{\sigma}\right)
\end{pmatrix}
$$

#### 3. **Temporal Coherence via Smoothing**
We maintain temporal continuity through recurrent color smoothing:
```math
𝐜_t = α·𝐜_{t-1} + (1-α)·𝐜_t^{target}
```

### Multi-Stage Constraint Learning

#### Stage 1: Foundation Learning
- **Objective**: Learn basic semantic separation
- **Method**: Contrastive loss that clusters expert trajectories while separating violators
- **Result**: Establishes the fundamental semantic coordinate system

#### Stage 2: Weak Temporal Pattern Learning  
- **Objective**: Capture simple temporal constraints
- **Method**: Smooth temporal logic operators applied to the color manifold
- **Result**: Learns robust "always/eventually" style constraints

#### Stage 3: Complex Temporal Pattern Learning
- **Objective**: Capture elaborate, long-horizon patterns
- **Method**: Temporal Convolutional Networks (TCNs) operating on color trajectories
- **Result**: Learns sophisticated multi-scale temporal dependencies

![Learning Pipeline](assets/pipeline.png)
*The three-stage learning process: from semantic foundation to complex temporal patterns*

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Paisley77/colorful-constraint-learning.git
cd colorful-constraint-learning

# Create virtual environment
python -m venv hsv
hsv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
# Generate demonstration data
python scripts/01_collect_demos.py --config configs/cartpole.yaml

# Train the foundation embedding
python scripts/02_train_foundation.py --config configs/default.yaml

# Visualize the learned manifold
python scripts/03_visualize_manifold.py --env cartpole
```

### Experiment with Custom Constraints

```python
from src.environments.constrained_cartpole import ConstrainedCartPole
from src.learning.staged_training import ColorfulConstraintLearner

# Create environment with constraint
env = ConstrainedCartPole(constraint_type="left_half")

# Initialize and train the framework
learner = ColorfulConstraintLearner(env)
manifold = learner.learn_constraints(expert_demos, violator_demos)
```

## Project Structure

```
colorful-constraint-learning/
├── src/                    # Core framework code
│   ├── environments/       # Custom constraint environments
│   ├── perception/         # Concept learning networks
│   ├── embedding/          # HSV color transformation
│   └── learning/           # Multi-stage training algorithm
├── configs/               # Experiment configurations
├── scripts/               # Execution pipelines
└── assets/                # Visualizations & demos
```

## Results & Visualizations

This framework produces intuitive geometric representations of learned constraints:

- **2D Semantic Maps**: Hue-Saturation projections showing natural separation
- **3D Temporal Manifolds**: Interactive trajectory visualizations
- **Constraint Evolution**: Learning progress across training stages

![3D Trajectory Visualization](assets/trajectory_3d.gif)
*Interactive 3D visualization of temporal manifolds in HSV space*

## Citation

If you use this work in your research, please cite:

```bibtex
@software{hou2025colorconstraints,
  title = {The Color of Constraints: A Geo-Semantic Interface for Inverse Constrained Reinforcement Learning},
  author = {Hou, Jingxuan},
  year = {2025},
  url = {https://github.com/Paisley77/colorful-constraint-learning}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

---

*This research represents ongoing work in neurosymbolic AI and inverse constrained reinforcement learning at the University of Pennsylvania.*
```