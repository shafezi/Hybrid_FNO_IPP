# A Modular Framework for Multi-Robot Adaptive Ocean Monitoring with Gaussian-Process Residual Correction

> Closed-loop framework that combines any dynamical prior (e.g., FNO, persistence) with Gaussian-process residual correction and Voronoi-partitioned multi-robot uncertainty-driven sampling. We additionally show that standard RMSE evaluation systematically misranks methods in this domain, and propose 1D wavenumber-spectral diagnostics and structural metrics (ACC, FSS) as the proper evaluation tools for IPP.

**Author:** Shayesteh Hafezi · Department of Naval Architecture and Marine Engineering, University of Michigan · `shafezi@umich.edu`

**Paper:** [`paper/main.tex`](paper/main.tex) (compiles to a 6-page IEEE conference manuscript; submitted to ICRA 2026).
**Poster:** [`paper/poster.tex`](paper/poster.tex) (A0 landscape, tikzposter).

---

## What this repo contains

| Folder | What's inside |
|---|---|
| [`ipp/`](ipp/) | Core library — Gaussian-process residual model, acquisition functions (uncertainty, GP-UCB, mutual information, hybrid), Voronoi partitioning, and multi-robot coordination logic. |
| [`dynamic_ipp/`](dynamic_ipp/) | Episode rollout engine — wires the FNO/persistence prior together with the GP correction loop and multi-robot acquisition for a full assimilation horizon. |
| [`scripts/`](scripts/) | All experiment, sweep, aggregation, and figure-generation scripts. The headline scripts are listed below. |
| [`configs/`](configs/) | YAML configs for all experiment families (single-robot baseline, dynamic rollout, residual-GP). |
| [`paper/`](paper/) | LaTeX sources for the ICRA submission and the A0 conference poster (tikzposter), plus the BibTeX file. |
| [`training/`](training/) | OceanNet (FNO) training scripts and SLURM launch wrappers. |
| [`docs/`](docs/) | Quick start, training info, and HPC cluster setup notes. |
| [`results_sample/`](results_sample/) | Aggregated metrics (`master_metrics.csv`, `summary_final_step.csv`) and the headline figures used in the paper and poster. The full result tree (~17 GB of per-episode metrics, individual timestep figures, and videos) is **not** in the repo — see *Data and trained weights* below. |

---

## The closed-loop framework, in one diagram

![System diagram](results_sample/system_diagram.png)

At every assimilation step:

1. **Dynamical prior** produces a forecast `ŷ_prior = f(ŷ)`. Plug in any model — we evaluate FNO, persistence, and a no-prior ablation.
2. **GP residual** fits `e(x) = y_true(x) − ŷ_prior(x)` on the accumulated robot observations.
3. **Multi-robot Voronoi acquisition** assigns each robot a partition cell and picks a target site within that cell maximizing an acquisition function (uncertainty / GP-UCB / mutual information).
4. Robots execute under a glider-speed budget (0.5 m/s) and observe `y_true(x_i)`; the GP refits.
5. Corrected estimate `ŷ = ŷ_prior + μ_GP` feeds back into step 1.

The framework treats `f` as a black box — the same code runs FNO, persistence, or no-prior.

---

## Headline finding

Standard RMSE evaluation systematically rewards methods that predict the climatological mean. A no-prior GP attains the lowest RMSE while having near-zero high-wavenumber spectral content — i.e., it loses all the eddies and fronts the monitoring is supposed to capture.

| Method (n=20, day 40) | RMSE ↓ | HF (ideal = 1) |
|---|---|---|
| GP-only                | **0.78** | 0.10 |
| **FNO+GP**             | 0.94 | **0.90** |
| Persist+GP             | 1.15 | 0.85 |

The corresponding 1D wavenumber spectra and qualitative fields make the failure mode explicit:

| Qualitative comparison | Power spectra |
|---|---|
| GP-only resembles the climatological mean; corrected methods recover mesoscale eddies. | GP-only collapses 4–5 orders of magnitude at high wavenumbers; corrected methods track ground truth across all $k$. |

See [`results_sample/figures_paper/`](results_sample/figures_paper/) for the publication figures.

---

## Reproducing the headline experiments

### Install

```bash
# clone, then
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Data and trained weights

Two artifacts are too large to ship in this repo:

- **ROMS reanalysis SSH (≈ 5 GB)** — preprocessed northwest-Pacific SSH from the ROMS regional ocean model used to train OceanNet and to evaluate every experiment. Available on request to the author.
- **Trained OceanNet (FNO) weights (≈ 400 MB)** — the 4-day single-step SSH predictor trained on 1993–2019 ROMS reanalysis. Available on request, or retrain with [`training/`](training/).

Place the data under `extracted_data/` and the model weights under `Models/`, matching the paths the configs expect.

### Run a single rollout

```bash
python scripts/run_dynamic_rollout_ipp.py --config configs/dynamic_rollout_ipp.yaml
```

### Run the full headline sweep (20 robots)

```bash
bash scripts/run_full_sweep_20bots.sh
```

This runs all combinations of `{Matérn ν=0.5, 1.5, 2.5; RBF}` × `{uncertainty, GP-UCB, MI}` × 5 seeds × 5 methods at 20 robots — about 300 episodes per robot count. Repeat with `run_full_sweep_5bots.sh`, `run_full_sweep_40bots.sh`, etc.

### Aggregate and produce paper figures

```bash
python scripts/aggregate_final_metrics.py     # builds master_metrics.csv
python scripts/make_paper_figures.py          # bar charts, scaling, Pareto
python scripts/make_system_diagram.py         # Figure 1
python scripts/run_final_psd.py --n_robots 20 --kernel matern --nu 1.5 \
    --t0 30 --seed 42 --acquisition uncertainty_only
python scripts/make_talk_slides.py            # the 7-min video deck
```

Each script writes into `results/dynamic_ipp/final/` by default.

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
├── paper/                       # main.tex, poster.tex, references.bib
├── training/                    # OceanNet (FNO) training scripts
├── docs/                        # quick start + cluster setup notes
├── results_sample/              # aggregated metrics + headline figures
│   ├── master_metrics.csv
│   ├── summary_final_step.csv
│   ├── system_diagram.png
│   └── figures_paper/
└── data_loader_SSH.py           # ROMS data loaders for FNO training
    data_loader_SSH_two_step.py
```

---

## Citation

If you use this framework or evaluation methodology, please cite:

```bibtex
@inproceedings{hafezi2026modular,
  title     = {A Modular Framework for Multi-Robot Adaptive Ocean Monitoring
               with Gaussian-Process Residual Correction},
  author    = {Hafezi, Shayesteh},
  booktitle = {Proceedings of the IEEE International Conference on Robotics
               and Automation (ICRA)},
  year      = {2026},
  note      = {Submitted}
}
```

---

## Acknowledgments

ROMS team for the regional ocean reanalyses, and the OceanNet authors for the public model architecture and training recipe.

---

## License

MIT — see [`LICENSE`](LICENSE).
