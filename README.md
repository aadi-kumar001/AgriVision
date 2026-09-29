# AgriVision — Field Intelligence Console

**Theme:** AgriTech — Technology for Food Security (SDG 2, 12, 13, 15)

A full-stack hackathon prototype covering the brief's problem areas:

| Problem area | Module |
|---|---|
| AI-powered crop disease & pest detection | **Disease Detection** — image classifier |
| Crop-yield prediction & farm advisory | **Yield Prediction** — regression model |
| Soil health and sustainable farming | **Soil Health** — NPK/pH scoring engine |
| Precision farming and smart irrigation | **Irrigation Advisory** — ET₀ water budget |
| Food distribution & supply-chain optimization | **Market Insights** — price trend + nearby mandis |
| (cross-cutting) | **Dashboard** + persistent **Analysis Log** (SQLite) |

Everything runs **fully offline** — no API keys, no external model downloads, no GPU. All models
train in-process in under a couple of seconds on synthetic-but-domain-grounded data (see
"How the intelligence works" below for exactly what is real ML vs. a rule engine, so you can
describe it accurately to judges).

---

## Project structure

```
agritech-hackathon/
├── backend/
│   ├── app.py            Flask API (all routes)
│   ├── ml_engine.py       models: disease classifier, yield regressor, soil/irrigation/market logic
│   ├── db.py              SQLite persistence (analysis history + dashboard stats)
│   └── requirements.txt
└── frontend/
    ├── index.html         single-page app shell (all module views)
    ├── css/style.css       design system
    └── js/app.js           API calls + rendering, vanilla JS (no build step)
```

## Run it

**1. Backend (Python 3.9+)**

```bash
cd backend
pip install -r requirements.txt
python app.py
```

This starts the API at `http://localhost:5000` and prints the supported crop list.
A SQLite file `agrivision.db` is created automatically on first run.

**2. Frontend**

The frontend is static HTML/CSS/JS — no build step. Two easy ways to run it:

- **Simplest:** just double-click `frontend/index.html` to open it in your browser.
- **Recommended** (avoids browser file:// quirks): serve it locally —
  ```bash
  cd frontend
  python -m http.server 8080
  ```
  then open `http://localhost:8080`.

The frontend talks to the API at `http://localhost:5000` by default. If your backend
runs elsewhere, change `API_BASE` at the top of `frontend/js/app.js`.

The sidebar footer shows a live green/red dot for API connectivity.

---

## API reference

All responses are JSON. CORS is open (`*`) so the static frontend can call it from any origin.

| Method & path | Body / query | Returns |
|---|---|---|
| `GET /api/health` | — | service status + supported crops |
| `POST /api/disease/detect` | multipart form, field `image` | disease label, confidence, severity, advisory |
| `POST /api/yield/predict` | JSON: `crop, area_ha, avg_temp_c, seasonal_rainfall_mm, soil_ph, fertilizer_kg_per_ha, irrigation_type` | predicted yield (t/ha), total production, advisory |
| `POST /api/soil/health` | JSON: `nitrogen_mg_kg, phosphorus_mg_kg, potassium_mg_kg, ph, organic_carbon_pct, crop` | 0–100 health score, category, deficiencies, recommendations |
| `POST /api/irrigation/advisory` | JSON: `soil_moisture_pct, temperature_c, humidity_pct, crop, growth_stage` | recommended water (L/m²), urgency, advisory |
| `GET /api/market/insights?crop=` | — | current price, 30-day trend, nearby mandi comparison |
| `GET /api/dashboard/summary` | — | totals for the dashboard tiles |
| `GET /api/history?limit=&module=` | — | recent logged analyses |

Supported crops: `rice, wheat, maize, cotton, sugarcane, soybean`.

### Example

```bash
curl -X POST http://localhost:5000/api/yield/predict \
  -H "Content-Type: application/json" \
  -d '{"crop":"rice","area_ha":2,"avg_temp_c":27,"seasonal_rainfall_mm":1150,
       "soil_ph":6.1,"fertilizer_kg_per_ha":80,"irrigation_type":"drip"}'
```

---

## How the intelligence works (be accurate with judges)

- **Disease detection** — a real trained `RandomForestClassifier` (scikit-learn) over an
  8-dimensional HSV color/texture feature vector extracted from the uploaded image
  (green/yellow/brown-orange/white-pale pixel fractions, dark-spot fraction, saturation,
  brightness variance = texture proxy). The training set is synthetically generated to match
  each disease class's known color signature. This is a genuine, fast, fully-offline
  classifier — it is **not** a deep CNN, and the API response says so explicitly
  (`"model"` field) so the UI never overstates it. Swapping in a CNN (e.g. a fine-tuned
  MobileNet on PlantVillage) is the natural next step and the API contract (`/api/disease/detect`)
  would not need to change.
- **Yield prediction** — a `RandomForestRegressor` trained on ~2,400 synthetic samples built
  from agronomic response curves (Gaussian temperature/rainfall/pH optima per crop, a
  logistic fertilizer response, and an irrigation-method multiplier), with noise added.
  It generalizes sensibly between the sampled points, which is why the API also returns a
  simple uncertainty estimate from the tree ensemble's spread.
- **Soil health, irrigation and market modules** are transparent rule/formula engines
  (NPK band scoring, a simplified Hargreaves-style reference-evapotranspiration water
  budget, and a deterministic synthetic price series) rather than trained ML — this is the
  right tool for these problems at this fidelity and is easy to defend in Q&A.

## Extending this for a real deployment

- Swap the disease classifier for a CNN fine-tuned on PlantVillage / a farm's own labeled photos.
- Replace the synthetic yield-training data with real historical district/mandal yield + weather + soil records.
- Feed the irrigation module from a live soil-moisture IoT sensor instead of a manual form field.
- Point the market module at a live mandi price API (e.g. India's Agmarknet) instead of the synthetic series.
- Add auth + per-farmer accounts on top of the existing SQLite schema (swap for Postgres at scale).
