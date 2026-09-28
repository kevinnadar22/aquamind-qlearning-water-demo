# AquaMind — Jowitt & Xu Q-Learning Water Demo

Tabular Q-learning pressure management on the **Jowitt & Xu (1990)** benchmark network, hydraulics via **[WNTR](https://github.com/USEPA/WNTR)** (EPANET 2.2), aligned with **Negm (2024)** (*Water Pressure Optimisation for Leakage Management Using Deep Reinforcement Learning*, Lancaster University, [eprint](https://eprints.lancs.ac.uk/id/eprint/217610/)).

Public mirror: [github.com/kevinnadar22/aquamind-qlearning-water-demo](https://github.com/kevinnadar22/aquamind-qlearning-water-demo)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8765
```

Open `http://127.0.0.1:8765` — choose **Background leakage** or **Burst (nodes 4, 9, 11)** and click **Run**.

## Demo script (~7 steps)

1. Start the server (above).
2. Run **Background leakage** — note KPI cards (leakage reduction %, violations, pressures, total leakage).
3. Open **Benchmark comparison** — compare **AquaMind Q-learning** row to Negm Table 5-4 (same scenario).
4. Scrub **Live network** playback — toggle **Baseline 40 m** vs **Q-learning**; watch PRV setpoints and pipe colours (10–70 m limits).
5. Read the **Critical node** footnote (node **5** = assumed high-demand / hospital proxy, not in the benchmark).
6. Switch to **Burst** — emitters on nodes **4, 9, 11** at **3** L·s⁻¹·m^−0.5, others **0** (Negm §6.2).
7. Compare burst results to Negm Table 6-2 in the same comparison table.

## Files

| File | Role |
|------|------|
| `jowitt_negm.inp` | Jowitt & Xu + PRVs on **P01, P31, P25** @ 40 m baseline; 22 background emitters (exp **1.18**); pattern **Fc**; 1 h steps |
| `network_sim.py` | WNTR hourly snapshots, reward, episode roll-out |
| `main.py` | Q-learning training at import (seed **42**), FastAPI `/api/simulate` |
| `index.html` | Dashboard, network map, benchmark table |

## Citations

- Jowitt, P., & Xu, C. (1990). *Predictive management of water distribution networks.*
- Negm, M. (2024). *Water Pressure Optimisation for Leakage Management Using Deep Reinforcement Learning.* Lancaster University. https://eprints.lancs.ac.uk/id/eprint/217610/
- Koşucu, G., & Demirel, S. (2022). EPANET models (CC-BY-4.0): https://doi.org/10.5281/zenodo.6243078
- WNTR: https://github.com/USEPA/WNTR

## Differences from Negm (2024)

| Topic | Negm (2024) | This demo |
|--------|-------------|-----------|
| Optimiser | DE, PSO, NM, PPO, A2C, etc. (tuned DRL, 20k timesteps) | **Tabular Q-learning**, 180 episodes, ε-greedy |
| State | Nodal pressures / hydraulic features | **Hour of day only** (24 states) |
| Actions | Continuous or fine discrete PRV settings in [0, 70] m | **3 levels per PRV: 25, 40, 55 m** → 27 joint actions |
| Reward scales | Tuned 3:1 leakage:violations (Jowitt); no tanh on Jowitt | Fixed **3:1** on per-node leakage fraction + violation delta |
| Episode metric | 3 test episodes averaged in thesis | **Single 24 h episode** per API run |
| PRV pipe **P25** | Literature pipe P25 (13→12) | Same topology; Koşucu file used pipe **P37** + valve **26→12** — rebuilt as **P25_u / PRV25** per Araujo/Negm |
| Koşucu PRV legs **27→15**, **28→21** | Modeled as PRV links in `JX_3PRV_epanet22.inp` | Restored as open pipes **P44**, **P45** (same diameters as former valves); **P40**, **P41** retained |
| Burst training | Trained on **random** burst locations, tested on fixed 4/9/11 | Q-table trained **on the same fixed burst** scenario |
| Critical node | Not part of benchmark | Node **5** labelled as assumed critical (highest base demand) |
| Comparison | — | Labelled **indicative** where the above applies |

## Reproducibility

All randomness uses `SEED = 42` in `main.py` (`random`, `numpy`).
