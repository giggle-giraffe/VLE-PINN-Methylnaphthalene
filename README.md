# VLE-PINN-Methylnaphthalene

An implementation of Physics-Informed Neural Networks (PINNs) for modeling the reaction
kinetics of catalytic 1-methylnaphthalene (MN) hydrogenation. This project combines deep
learning with chemical engineering principles to predict reaction outcomes and recover
intrinsic kinetic parameters from sparse reactor data.

What sets this framework apart is that it is **phase-aware**: a neural phase-equilibrium
surrogate — trained on rigorous flash calculations — is embedded inside the PINN, so kinetic
parameters are inferred against the phase-correct liquid/vapor compositions at every point
along the reactor trajectory, not against raw feed compositions.

```
A (1-methylnaphthalene) ──r1──▶ B (1-methyltetralin)  ──r2──▶ D (methyldecalin)
A (1-methylnaphthalene) ──r3──▶ C (5-methyltetralin)  ──r4──▶ D (methyldecalin)
```

Reaction rates follow a Langmuir–Hinshelwood (LHHW) form with competitive adsorption of
H₂ and species A–D. The PINN learns 18 physical parameters — four (A, E) reaction pairs
and five (A, ΔH) adsorption pairs — jointly with a neural trajectory model, by minimizing
data mismatch and the ODE residual `dX/d(m/F) = r(X, T, p_H2; θ)` at continuously sampled
collocation points.

## 🔬 Features

- **Phase-Aware PINN Architecture**: A differentiable neural VLE surrogate (phase
  classifier + equilibrium-composition regressors) frozen inside the PINN, keeping
  gradients flowing through the flash calculation
- **Rigorous Synthetic VLE Data**: Peng–Robinson flash calculations over wide
  temperature / pressure / composition ranges to train the surrogate
- **Curriculum Learning**: Multi-stage training with staged loss weights, initial-condition
  pre-training, and strategic parameter freezing
- **GradNorm Optimization**: Multi-objective loss balancing with one-time auto-calibration,
  target training rates, weight caps, and explosion rollback
- **Adaptive Training**: Epoch-based learning-rate scheduling, non-uniform PDE collocation
  sampling, and an optional late switch from Adam to L-BFGS

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.x (pinned in `requirements.txt`)
- CUDA-compatible GPU (recommended)

### Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `thermo` + `fluids` (flash calculations), `h5py`, `pandas`,
`scikit-learn`, `loguru`, `matplotlib`.

## 📁 Project Structure

```
pinn_phase_os_dev/
├── thermo_gen/                  # Synthetic VLE data generation
│   ├── src/thermo_gen.py        #   Rigorous PT-flash sampling (Peng–Robinson EOS)
│   └── config/                  #   Component list (CAS) and T/P/composition ranges
├── phase/                       # Neural phase-equilibrium surrogate
│   ├── src/
│   │   ├── clf_nn.py            #   Phase classifier (two-phase vs. single-phase)
│   │   ├── reg_nn.py            #   Regressors (vapor fraction + equilibrium compositions)
│   │   ├── train_clf.py         #   Training scripts for the three networks
│   │   ├── train_reg_x.py
│   │   ├── train_reg_y.py
│   │   └── data_util_hdf5.py    #   HDF5 datasets and feature scaling
│   └── config/                  #   Surrogate training / inference configurations
├── pinn/                        # The phase-aware PINN
│   ├── src/
│   │   ├── model_pinn_mn.py     #   PINN architecture (encoders, kinetic parameters)
│   │   ├── model_phase_mn.py    #   Embedded differentiable phase surrogate
│   │   ├── loss.py              #   LHHW PDE residual and physics-informed losses
│   │   ├── train.py             #   Training loop (curriculum + GradNorm + L-BFGS)
│   │   ├── training/            #   Checkpointing, config parsing, loss assembly,
│   │   │                        #   adaptive weighting
│   │   └── plot.py              #   Visualization utilities
│   └── config/train_pinn_mn.yaml
├── requirements.txt
└── LICENSE
```

## 🔧 Configuration

The project uses YAML configuration files to control all stages:

- `thermo_gen/config/gen_synthetics_mn.yaml` — chemical components (CAS numbers) and the
  temperature / pressure / composition ranges sampled by the flash calculations
- `phase/config/training_*.yaml` — architecture and training hyperparameters for the
  phase classifier and the two composition regressors
- `pinn/config/train_pinn_mn.yaml` — PINN training: parameter bounds for the kinetic
  transforms, curriculum stages, GradNorm settings, learning-rate schedule, and the
  L-BFGS switchover

## 📊 Key Algorithms

### Phase-Equilibrium Surrogate
Three networks approximate the flash calculation: a classifier decides two-phase vs.
single-phase, and two regressors predict vapor fraction and equilibrium mole fractions
for each regime. At inference the classifier's softmax probabilities blend the regressor
outputs, keeping the phase decision differentiable so gradients propagate through the
VLE calculation into the kinetic parameters.

### Physics-Informed Training
- **Data-driven learning**: Outlet flow rates from steady-state reactor experiments
- **Physics-based constraints**: LHHW ODE residuals evaluated on flash-corrected
  compositions at continuously sampled collocation points
- **Mass conservation**: Total-flow and pointwise carbon-balance penalties ensure
  physical consistency

### Curriculum Learning
- **Multi-stage training**: Progressive rebalancing from data-dominant to physics-dominant
  objectives, with smooth interpolation between stages
- **Parameter freezing**: Kinetic parameters stay frozen while the trajectory network
  first learns the initial conditions
- **Adaptive loss weighting**: Dynamic balancing of data, PDE, initial-condition,
  conservation, and constraint losses

### GradNorm Optimization
- **Multi-objective balancing**: Automatic loss-weight adjustment from gradient magnitudes
- **Auto-calibration**: One-time normalization that puts all loss terms on a common scale
- **Stability safeguards**: Weight caps, conservative early-phase updates, and rollback
  on detected weight explosions

## 📈 Results and Visualization

The training process generates comprehensive visualizations:

- Loss evolution curves (raw and normalized, per component)
- Adaptive weight dynamics and gradient-magnitude analysis
- Predicted vs. measured outlet flow rates with calibration metrics
- Species trajectories along the reactor coordinate
- Learned-parameter histories and recovered kinetic parameters in engineering units

## 📚 Citation

If you use this code in your research, please cite this repository:

```bibtex
@article{pinn_phase_mn,
  author = {Yafei Pan, Wenbin Chen, Xiaoqian Dang, Tao Li, Wei Zhang, Chen Zhang, Cuiqing Li, Yong Luo, Feng Liu and Mingfeng Li},
  title  = {Integrating Hydrogenation Reaction Kinetics with Phase Equilibrium Using Physics-Informed Neural Network},
  year   = {2026},
  note   = {Under Review}
}
```

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🤝 Contact

- Tao Li: tao.li@gigglegiraffe.com
- Xiaoqian Dang: dangxiaoqian@gigglegiraffe.com
- Chen Zhang: zhangc@bipt.edu.cn

For questions about the methodology or collaboration opportunities, please open an issue or contact the maintainers directly.

## 🔗 Related Work

This project builds upon advances in:
- Physics-Informed Neural Networks (PINNs)
- Multi-objective optimization in deep learning
- Vapor–liquid equilibrium and equation-of-state modeling
- Heterogeneous catalytic kinetics (Langmuir–Hinshelwood models)
- Curriculum learning strategies

---

