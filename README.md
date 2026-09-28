# AquaMind — Autonomous Water Infrastructure (Q-Learning Demo)

Minimal demo web app for the mini-project **Autonomous Water Infrastructure Management via Q-Learning** (K.J. Somaiya — Group 24). It simulates hourly mass-balance control of a small multi-zone network (including a **Hospital** priority zone) and compares a **static SCADA baseline** to a **tabular Q-learning agent**.

## Stack

- **Backend:** one FastAPI file (`main.py`) — simulation, Q-learning training on startup, JSON API
- **Frontend:** one static page (`index.html`) — Tailwind + Chart.js via CDN
- **No** React, npm, or EPANET/WNTR

## Quick start

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or:

```bash
pip install -r requirements.txt && python3 main.py
```

Open **http://localhost:8000**. The first startup trains Q-tables for all scenarios (a few seconds).

## Demo script (~6 steps)

1. Start the server and open the dashboard. Point out the **AquaMind** header and four KPI cards (live metrics, not slides).
2. Leave **Normal operations** selected and click **Run simulation**. Note **water loss reduction** and that **hospital reliability** for the agent meets or beats the baseline.
3. Open the **Reservoir storage** chart: the agent curve stays higher over the 30-day horizon than the fixed high-aperture baseline.
4. Switch to **Drought (reduced inflow)** and run again. Highlight **hospital supply reliability** (baseline drops; agent prioritizes the Hospital zone via reward shaping).
5. Switch to **Pipe leak (Industrial zone)** and run. Show lower **leakage** time series and improved **reservoir autonomy** vs baseline.
6. Briefly show **Q-learning training reward** (convergence on startup) and **mean off-peak pressure** under 25 m for the agent — tying back to FAVAD leakage control and pressure management objectives.

## API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard UI |
| `GET /api/simulate?scenario=normal\|drought\|leak` | Full KPIs + chart series (JSON) |
| `GET /api/health` | Health check |

## Model (summary)

- 1 reservoir, 5 zones (Hospital included), valve openings low/med/high
- Demand: daily peaks + noise; leakage `L = k · P^1.18`
- Q-learning state: reservoir level, time-of-day, demand buckets; actions: hospital + network valve presets
- Scenarios adjust inflow (drought) or leak coefficient (pipe leak)

All displayed numbers come from the running simulation.

## Team context

Maria Kevin, Sahil Khot, Krishna Modi — Guide: Prof. Pravin Patil.
