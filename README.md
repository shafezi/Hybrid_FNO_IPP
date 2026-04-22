# A Modular Framework for Multi-Robot Adaptive Ocean Monitoring with Gaussian-Process Residual Correction

> Closed-loop framework that combines any dynamical prior (e.g., FNO, persistence) with Gaussian-process residual correction and Voronoi-partitioned multi-robot uncertainty-driven sampling. We additionally show that standard RMSE evaluation systematically misranks methods in this domain, and propose 1D wavenumber-spectral diagnostics and structural metrics (ACC, FSS) as the proper evaluation tools for IPP.

**Author:** Shayesteh Hafezi · Department of Naval Architecture and Marine Engineering, University of Michigan · `shafezi@umich.edu`

This repository contains the **code** for the framework, all experiments, and all analysis/figure-generation scripts. The accompanying ICRA paper, conference poster, and full result tree are not in the repo (the paper is under submission; the result tree is ~17 GB). See *Data and trained weights* and *Citation* below.

---

## What this repo contains

| Folder | What's inside |
|---|---|
| [`ipp/`](ipp/) | Core library — Gaussian-process residual model, acquisition functions (uncertainty, GP-UCB, mutual information, hybrid), Voronoi partitioning, and multi-robot coordination logic. |
| [`dynamic_ipp/`](dynamic_ipp/) | Episode rollout engine — wires the FNO/persistence prior together with the GP correction loop and multi-robot acquisition for a full assimilation horizon. |
| [`scripts/`](scripts/) | All experiment, sweep, aggregation, and figure-generation scripts. The headline scripts are listed below. |
| [`configs/`](configs/) | YAML configs for all experiment families (single-robot baseline, dynamic rollout, residual-GP). |
| [`training/`](training/) | OceanNet (FNO) training scripts and SLURM launch wrappers. |
| [`docs/`](docs/) | Quick start, training info, and HPC cluster setup notes. |
| [`data_loader_SSH*.py`](.) | ROMS sea-surface-height data loaders for FNO training. |

---

## The closed-loop framework

At every assimilation step:

1. **Dynamical prior** produces a forecast `ŷ_prior = f(ŷ)`. Plug in any model — we evaluate FNO, persistence, and a no-prior ablation.
2. **GP residual** fits `e(x) = y_true(x) − ŷ_prior(x)` on the accumulated robot observations.
3. **Multi-robot Voronoi acquisition** assigns each robot a partition cell and picks a target site within that cell maximizing an acquisition function (uncertainty / GP-UCB / mutual information).
4. Robots execute under a glider-speed budget (0.5 m/s) and observe `y_true(x_i)`; the GP refits.
5. Corrected estimate `ŷ = ŷ_prior + μ_GP` feeds back into step 1.

---

## Headline finding

Standard RMSE evaluation systematically rewards methods that predict the climatological mean. A no-prior GP attains the lowest RMSE while having near-zero high-wavenumber spectral content — i.e., it loses all the eddies and fronts the monitoring is supposed to capture.

| Method (n=20, day 40) | RMSE ↓ | HF (ideal = 1) |
|---|---|---|
| GP-only                | **0.78** | 0.10 |
| **FNO+GP**             | 0.94 | **0.90** |
| Persist+GP             | 1.15 | 0.85 |

The corresponding 1D wavenumber spectra and qualitative fields make the failure mode explicit: GP-only resembles the climatological mean and has spectra that collapse 4–5 orders of magnitude at high wavenumbers; FNO+GP and Persist+GP recover mesoscale eddies and track the ground-truth spectrum. Run `python scripts/make_paper_figures.py` after a sweep to regenerate the headline figures locally.

---

## Reproducing the experiments

### Install

```bash
# clone, then
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Data and trained weights

Two artifacts are too large to ship in this repo:

- **ROMS reanalysis SSH (≈ 5 GB)** — preprocessed northwest-Pacific SSH from the ROMS regional ocean model used to train OceanNet and to evaluate every experiment. Available on request to the author.
- **Trained OceanNet (FNO) weights (≈ 400 MB)** — the 4-day single-step SSH predictor trained on 1993–2019 ROMS reanalysis. Available on request, or retrain with the scripts in [`training/`](training/).

Place the data under `extracted_data/` and the model weights under `Models/`, matching the paths the configs expect.

### Run a single rollout

```bash
python scripts/run_dynamic_rollout_ipp.py --config configs/dynamic_rollout_ipp.yaml
```

### Run the full headline sweep

```bash
bash scripts/run_full_sweep_5bots.sh
bash scripts/run_full_sweep_20bots.sh
bash scripts/run_full_sweep_40bots.sh
```

Each runs all combinations of `{Matérn ν=0.5, 1.5, 2.5; RBF}` × `{uncertainty, GP-UCB, MI}` × 5 seeds × 5 methods at the chosen robot count — about 300 episodes per robot count. Results are written under `results/dynamic_ipp/final/`.

### Aggregate metrics and produce figures

```bash
python scripts/aggregate_final_metrics.py     # builds master_metrics.csv
python scripts/make_paper_figures.py          # bar charts, scaling, Pareto
python scripts/make_system_diagram.py         # closed-loop schematic
python scripts/run_final_psd.py --n_robots 20 --kernel matern --nu 1.5 \
    --t0 30 --seed 42 --acquisition uncertainty_only
python scripts/make_talk_slides.py            # 7-min video deck (.pptx)
```

---

## Repository map at a glance

```
release/
├── README.md                    # this file
├── LICENSE                      # MIT
├── requirements.txt             # Python dependencies
├── ipp/                         # GP, acquisition, Voronoi
├── dynamic_ipp/                 # episode rollout
├── scripts/                     # experiments, sweeps, figures, video deck
├── configs/                     # YAML experiment configs
├── training/                    # OceanNet (FNO) training scripts
├── docs/                        # quick start + cluster setup notes
└── data_loader_SSH.py           # ROMS data loaders for FNO training
    data_loader_SSH_two_step.py
```

---


## Acknowledgments

ROMS team for the regional ocean reanalyses, and the OceanNet authors for the public model architecture and training recipe.

---

## License

MIT — see [`LICENSE`](LICENSE).
