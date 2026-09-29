"""
ml_engine.py
------------
All "intelligence" for AgriVision lives here:

  1. Crop disease detector   -> image color/texture features + RandomForestClassifier
  2. Yield predictor         -> RandomForestRegressor trained on agronomic synthetic data
  3. Soil health scorer      -> rule engine grounded in standard NPK/pH bands
  4. Irrigation advisor      -> simplified reference-evapotranspiration (ET0) water budget
  5. Market insight engine   -> deterministic synthetic mandi price series

Everything is trained/generated at process start so the API works fully
offline with no external downloads, no API keys, and no GPU.  The disease
detector is a genuine trained classifier (features -> RandomForest), not a
hand-coded if/else chain, but it is intentionally a *lightweight* model
suitable for a hackathon demo rather than a deep CNN — this is stated
explicitly in the API responses so it is never presented as more than it is.
"""

import io
import math
import random
import hashlib
from datetime import date, timedelta

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


# --------------------------------------------------------------------------
# 1. CROP DISEASE DETECTION
# --------------------------------------------------------------------------

DISEASE_CLASSES = [
    "healthy",
    "leaf_blight",
    "leaf_rust",
    "powdery_mildew",
    "nutrient_deficiency",
]

DISEASE_INFO = {
    "healthy": {
        "label": "Healthy leaf",
        "severity": "none",
        "advisory": [
            "No signs of stress detected. Continue routine monitoring every 5-7 days.",
            "Maintain current irrigation and fertilization schedule.",
        ],
    },
    "leaf_blight": {
        "label": "Leaf blight (fungal)",
        "severity": "high",
        "advisory": [
            "Remove and destroy visibly infected leaves to slow spread.",
            "Apply a copper-based fungicide at label rate within 48 hours.",
            "Avoid overhead irrigation; water at the base to keep foliage dry.",
        ],
    },
    "leaf_rust": {
        "label": "Leaf rust (fungal)",
        "severity": "medium",
        "advisory": [
            "Apply a triazole or strobilurin fungicide at first sign of pustules.",
            "Improve field air circulation by widening plant spacing next season.",
            "Scout weekly — rust spreads fast in humid, warm conditions.",
        ],
    },
    "powdery_mildew": {
        "label": "Powdery mildew (fungal)",
        "severity": "medium",
        "advisory": [
            "Apply sulfur-based or potassium-bicarbonate spray in the early evening.",
            "Increase spacing/pruning to raise airflow around the canopy.",
            "Avoid high-nitrogen feeding, which increases susceptibility.",
        ],
    },
    "nutrient_deficiency": {
        "label": "Nutrient deficiency (likely nitrogen)",
        "severity": "low",
        "advisory": [
            "Soil-test for N-P-K before applying fertilizer.",
            "Apply a balanced or nitrogen-leaning fertilizer split over 2 doses.",
            "Re-photograph the same leaves in 10-14 days to confirm greening.",
        ],
    },
}


def _extract_leaf_features(img: Image.Image) -> np.ndarray:
    """Convert an image into an 8-dim color/texture feature vector.

    Features are simple, explainable HSV summary statistics computed over
    likely-leaf pixels (green/yellow/brown hue range), which is exactly the
    kind of feature a plant-pathology teaching model uses before reaching
    for a deep CNN.
    """
    img = img.convert("RGB").resize((256, 256))
    arr = np.asarray(img).astype(np.float32) / 255.0

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.max(arr, axis=-1)
    minc = np.min(arr, axis=-1)
    v = maxc
    s = np.where(maxc == 0, 0, (maxc - minc) / (maxc + 1e-6))

    # hue
    delta = (maxc - minc) + 1e-6
    hue = np.zeros_like(maxc)
    mask_r = (maxc == r)
    mask_g = (maxc == g)
    mask_b = (maxc == b)
    hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r]) + 360) % 360
    hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120) % 360
    hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240) % 360

    leaf_mask = s > 0.15  # ignore near-grey background/glare
    if leaf_mask.sum() < 200:
        leaf_mask = np.ones_like(s, dtype=bool)

    h_leaf = hue[leaf_mask]
    s_leaf = s[leaf_mask]
    v_leaf = v[leaf_mask]

    def frac_in(lo, hi):
        return float(np.mean((h_leaf >= lo) & (h_leaf < hi)))

    green_frac = frac_in(70, 170)
    yellow_frac = frac_in(40, 70)
    brown_orange_frac = frac_in(10, 40)
    white_pale_frac = float(np.mean((s_leaf < 0.18) & (v_leaf > 0.6)))
    dark_spot_frac = float(np.mean(v_leaf < 0.28))
    sat_mean = float(np.mean(s_leaf))
    val_std = float(np.std(v_leaf))  # texture proxy: blotchiness
    val_mean = float(np.mean(v_leaf))

    return np.array([
        green_frac, yellow_frac, brown_orange_frac, white_pale_frac,
        dark_spot_frac, sat_mean, val_std, val_mean,
    ], dtype=np.float32)


def _synth_disease_sample(label: str) -> np.ndarray:
    """Generate one synthetic 8-dim feature vector typical of a class."""
    n = lambda mu, sd: float(np.clip(rng.normal(mu, sd), 0, 1))
    if label == "healthy":
        return np.array([n(0.80, 0.06), n(0.05, 0.03), n(0.02, 0.02),
                          n(0.01, 0.01), n(0.02, 0.02), n(0.55, 0.08),
                          n(0.07, 0.02), n(0.55, 0.08)])
    if label == "leaf_blight":
        return np.array([n(0.35, 0.10), n(0.10, 0.05), n(0.35, 0.10),
                          n(0.02, 0.02), n(0.25, 0.08), n(0.45, 0.10),
                          n(0.16, 0.04), n(0.40, 0.08)])
    if label == "leaf_rust":
        return np.array([n(0.45, 0.10), n(0.15, 0.06), n(0.28, 0.09),
                          n(0.02, 0.02), n(0.10, 0.05), n(0.50, 0.09),
                          n(0.13, 0.03), n(0.48, 0.08)])
    if label == "powdery_mildew":
        return np.array([n(0.40, 0.10), n(0.08, 0.04), n(0.05, 0.03),
                          n(0.35, 0.10), n(0.03, 0.02), n(0.30, 0.08),
                          n(0.14, 0.03), n(0.62, 0.07)])
    if label == "nutrient_deficiency":
        return np.array([n(0.35, 0.09), n(0.45, 0.10), n(0.05, 0.03),
                          n(0.01, 0.01), n(0.02, 0.02), n(0.40, 0.09),
                          n(0.08, 0.02), n(0.52, 0.07)])
    raise ValueError(label)


def _train_disease_model():
    X, y = [], []
    for label in DISEASE_CLASSES:
        for _ in range(300):
            X.append(_synth_disease_sample(label))
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=8, random_state=RANDOM_SEED
    )
    clf.fit(X, y)
    return clf


_disease_model = _train_disease_model()


def predict_disease(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes))
    feats = _extract_leaf_features(img).reshape(1, -1)
    proba = _disease_model.predict_proba(feats)[0]
    classes = _disease_model.classes_
    order = np.argsort(proba)[::-1]
    top_label = classes[order[0]]
    info = DISEASE_INFO[top_label]
    ranked = [
        {"disease": classes[i], "label": DISEASE_INFO[classes[i]]["label"],
         "confidence": round(float(proba[i]) * 100, 1)}
        for i in order
    ]
    return {
        "prediction": top_label,
        "label": info["label"],
        "severity": info["severity"],
        "confidence": round(float(proba[order[0]]) * 100, 1),
        "advisory": info["advisory"],
        "top_candidates": ranked[:3],
        "model": "RandomForest (8-feature HSV color/texture) — lightweight demo classifier",
    }


# --------------------------------------------------------------------------
# 2. YIELD PREDICTION
# --------------------------------------------------------------------------

CROP_BASELINE = {
    # crop: (base t/ha, temp_opt, rain_opt_mm_per_season, ph_opt)
    "rice":      (4.2, 27, 1200, 6.2),
    "wheat":     (3.5, 20, 550, 6.8),
    "maize":     (5.0, 24, 650, 6.3),
    "cotton":    (1.8, 28, 700, 6.8),
    "sugarcane": (70.0, 26, 1500, 6.5),
    "soybean":   (2.6, 25, 600, 6.4),
}


def _yield_formula(crop, area_ha, temp, rain, ph, fert, irrigation):
    base, t_opt, r_opt, ph_opt = CROP_BASELINE[crop]
    temp_factor = math.exp(-((temp - t_opt) ** 2) / (2 * 6 ** 2))
    rain_factor = math.exp(-((rain - r_opt) ** 2) / (2 * (r_opt * 0.35) ** 2))
    ph_factor = math.exp(-((ph - ph_opt) ** 2) / (2 * 0.6 ** 2))
    fert_factor = 1 / (1 + math.exp(-(fert - 60) / 25))  # logistic response
    irrigation_bonus = {"drip": 1.12, "sprinkler": 1.06, "canal": 1.0, "rainfed": 0.85}.get(irrigation, 1.0)
    yield_per_ha = base * (0.25 + 0.75 * temp_factor) * (0.25 + 0.75 * rain_factor) \
        * (0.4 + 0.6 * ph_factor) * (0.6 + 0.4 * fert_factor) * irrigation_bonus
    return max(yield_per_ha, base * 0.1)


def _build_yield_dataset(n_per_crop=400):
    rows, targets = [], []
    for crop in CROP_BASELINE:
        for _ in range(n_per_crop):
            temp = rng.normal(CROP_BASELINE[crop][1], 5)
            rain = max(50, rng.normal(CROP_BASELINE[crop][2], CROP_BASELINE[crop][2] * 0.3))
            ph = np.clip(rng.normal(CROP_BASELINE[crop][3], 0.7), 4.5, 8.5)
            fert = np.clip(rng.normal(70, 30), 0, 200)
            irrigation = rng.choice(["drip", "sprinkler", "canal", "rainfed"])
            true_yield = _yield_formula(crop, 1, temp, rain, ph, fert, irrigation)
            noisy_yield = max(0.1, true_yield * rng.normal(1.0, 0.06))
            rows.append([
                list(CROP_BASELINE).index(crop), temp, rain, ph, fert,
                ["drip", "sprinkler", "canal", "rainfed"].index(irrigation),
            ])
            targets.append(noisy_yield)
    return np.array(rows, dtype=np.float32), np.array(targets, dtype=np.float32)


def _train_yield_model():
    X, y = _build_yield_dataset()
    reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_SEED)
    reg.fit(X, y)
    return reg


_yield_model = _train_yield_model()
_crop_index = list(CROP_BASELINE)
_irrigation_index = ["drip", "sprinkler", "canal", "rainfed"]


def predict_yield(crop, area_ha, temp, rain, ph, fert, irrigation):
    crop = crop.lower()
    irrigation = irrigation.lower()
    if crop not in CROP_BASELINE:
        raise ValueError(f"Unsupported crop '{crop}'. Supported: {list(CROP_BASELINE)}")
    if irrigation not in _irrigation_index:
        irrigation = "canal"

    x = np.array([[
        _crop_index.index(crop), temp, rain, ph, fert,
        _irrigation_index.index(irrigation),
    ]], dtype=np.float32)

    tree_preds = np.array([t.predict(x)[0] for t in _yield_model.estimators_])
    per_ha = float(np.mean(tree_preds))
    std = float(np.std(tree_preds))
    total = per_ha * area_ha

    base, t_opt, r_opt, ph_opt = CROP_BASELINE[crop]
    tips = []
    if abs(temp - t_opt) > 6:
        tips.append(f"Season temperature is {abs(temp - t_opt):.1f}°C from the {crop} optimum ({t_opt}°C) — consider adjusting sowing dates.")
    if rain < r_opt * 0.7:
        tips.append("Rainfall is well below optimum — plan supplemental irrigation to avoid water stress.")
    elif rain > r_opt * 1.4:
        tips.append("Rainfall is well above optimum — ensure field drainage to prevent waterlogging.")
    if abs(ph - ph_opt) > 0.8:
        tips.append(f"Soil pH {ph} is off the {crop} optimum ({ph_opt}) — a lime or gypsum amendment may help.")
    if fert < 30:
        tips.append("Fertilizer input looks low relative to typical requirement — a soil test can confirm a safe increase.")
    if irrigation == "rainfed":
        tips.append("Switching to drip/sprinkler irrigation could meaningfully raise yield and water-use efficiency.")
    if not tips:
        tips.append("Conditions are close to optimal for this crop — maintain current practices.")

    return {
        "crop": crop,
        "predicted_yield_t_per_ha": round(per_ha, 2),
        "yield_uncertainty_t_per_ha": round(std, 2),
        "predicted_total_production_t": round(total, 2),
        "area_ha": area_ha,
        "advisory": tips,
        "model": "RandomForestRegressor (200 trees) trained on agronomic synthetic data",
    }


# --------------------------------------------------------------------------
# 3. SOIL HEALTH
# --------------------------------------------------------------------------

# Standard-ish extractable soil nutrient bands (mg/kg) used for teaching purposes
NPK_BANDS = {
    "N": {"low": 280, "high": 560},
    "P": {"low": 10, "high": 25},
    "K": {"low": 110, "high": 280},
}


def _band_score(value, low, high):
    if value < low:
        return max(0, 100 * value / low) * 0.6  # deficient -> capped lower
    if value > high:
        excess = (value - high) / high
        return max(40, 100 - excess * 60)
    # inside optimal band -> scaled 70-100
    span = high - low
    return 70 + 30 * (value - low) / span


def soil_health(n, p, k, ph, organic_carbon, crop):
    n_score = _band_score(n, NPK_BANDS["N"]["low"], NPK_BANDS["N"]["high"])
    p_score = _band_score(p, NPK_BANDS["P"]["low"], NPK_BANDS["P"]["high"])
    k_score = _band_score(k, NPK_BANDS["K"]["low"], NPK_BANDS["K"]["high"])

    ph_opt = CROP_BASELINE.get(crop.lower(), (None, None, None, 6.5))[3]
    ph_score = max(0, 100 - abs(ph - ph_opt) * 35)

    oc_score = min(100, max(0, (organic_carbon / 0.75) * 100))  # 0.75% ~ good

    overall = round(0.28 * n_score + 0.22 * p_score + 0.22 * k_score + 0.16 * ph_score + 0.12 * oc_score, 1)

    if overall >= 80:
        category = "Excellent"
    elif overall >= 65:
        category = "Good"
    elif overall >= 45:
        category = "Fair — needs attention"
    else:
        category = "Poor — intervention needed"

    deficiencies = []
    recs = []
    if n < NPK_BANDS["N"]["low"]:
        deficiencies.append("Nitrogen (N)")
        recs.append("Apply urea or a nitrogen-rich organic amendment (compost/FYM) in split doses.")
    if p < NPK_BANDS["P"]["low"]:
        deficiencies.append("Phosphorus (P)")
        recs.append("Apply DAP or rock phosphate at sowing to support root development.")
    if k < NPK_BANDS["K"]["low"]:
        deficiencies.append("Potassium (K)")
        recs.append("Apply muriate of potash (MOP) to improve stress and disease resistance.")
    if abs(ph - ph_opt) > 0.8:
        if ph < ph_opt:
            recs.append(f"Soil is acidic for {crop} (pH {ph} vs optimum {ph_opt}) — apply agricultural lime.")
        else:
            recs.append(f"Soil is alkaline for {crop} (pH {ph} vs optimum {ph_opt}) — apply gypsum or elemental sulfur.")
    if organic_carbon < 0.5:
        recs.append("Organic carbon is low — incorporate crop residue, compost, or green manure to build soil structure.")
    if not recs:
        recs.append("Soil parameters are within healthy ranges — maintain current nutrient management plan.")

    return {
        "overall_score": overall,
        "category": category,
        "component_scores": {
            "nitrogen": round(n_score, 1),
            "phosphorus": round(p_score, 1),
            "potassium": round(k_score, 1),
            "ph": round(ph_score, 1),
            "organic_carbon": round(oc_score, 1),
        },
        "deficiencies": deficiencies or ["None detected"],
        "recommendations": recs,
    }


# --------------------------------------------------------------------------
# 4. IRRIGATION ADVISORY  (simplified reference-ET water budget)
# --------------------------------------------------------------------------

CROP_KC = {  # simplified crop coefficient (mid-season)
    "rice": 1.15, "wheat": 1.05, "maize": 1.15, "cotton": 1.15,
    "sugarcane": 1.25, "soybean": 1.05,
}

GROWTH_STAGE_FACTOR = {
    "seedling": 0.5, "vegetative": 0.85, "flowering": 1.15, "maturity": 0.6,
}


def irrigation_advisory(soil_moisture_pct, temperature, humidity, crop, growth_stage):
    crop = crop.lower()
    growth_stage = growth_stage.lower()
    kc = CROP_KC.get(crop, 1.05)
    stage_f = GROWTH_STAGE_FACTOR.get(growth_stage, 0.85)

    # Very simplified Hargreaves-like ET0 proxy (mm/day), offline-safe
    et0 = max(0.5, 0.0135 * (temperature + 17.8) * math.sqrt(max(1, 40 - humidity)))
    etc = et0 * kc * stage_f  # crop water demand, mm/day

    moisture_deficit = max(0, 65 - soil_moisture_pct)  # target field capacity ~65%
    water_mm = round(etc + moisture_deficit * 0.15, 1)
    liters_per_sqm = round(water_mm, 1)  # 1 mm over 1 sqm = 1 liter

    if soil_moisture_pct >= 60:
        urgency = "low"
        next_irrigation_days = 3
    elif soil_moisture_pct >= 35:
        urgency = "medium"
        next_irrigation_days = 1
    else:
        urgency = "high"
        next_irrigation_days = 0

    tips = []
    if urgency == "high":
        tips.append("Soil moisture is critically low — irrigate today, preferably early morning or evening.")
    elif urgency == "medium":
        tips.append("Soil moisture is trending low — plan irrigation within 24 hours.")
    else:
        tips.append("Soil moisture is adequate — hold irrigation and recheck in a couple of days.")
    if temperature > 34:
        tips.append("High temperature increases evapotranspiration — consider mulching to reduce water loss.")
    tips.append("Drip irrigation can cut water use by 30-50% versus flood irrigation for this crop.")

    return {
        "crop": crop,
        "growth_stage": growth_stage,
        "reference_et0_mm_day": round(et0, 2),
        "crop_water_demand_mm_day": round(etc, 2),
        "recommended_water_liters_per_sqm": liters_per_sqm,
        "urgency": urgency,
        "next_irrigation_in_days": next_irrigation_days,
        "advisory": tips,
    }


# --------------------------------------------------------------------------
# 5. MARKET / SUPPLY-CHAIN INSIGHTS  (deterministic synthetic series)
# --------------------------------------------------------------------------

MARKET_BASE_PRICE = {  # INR per quintal (100 kg), illustrative
    "rice": 2100, "wheat": 2250, "maize": 1950, "cotton": 6800,
    "sugarcane": 340, "soybean": 4300,
}


def _seeded_random(seed_str, i):
    h = hashlib.sha256(f"{seed_str}-{i}".encode()).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) * 2 - 1  # in [-1, 1]


def market_insights(crop):
    crop = crop.lower()
    base = MARKET_BASE_PRICE.get(crop, 2500)
    today = date.today()
    series = []
    price = base
    for i in range(30, 0, -1):
        drift = _seeded_random(crop, i) * base * 0.012
        price = max(base * 0.7, price + drift)
        d = today - timedelta(days=i)
        series.append({"date": d.isoformat(), "price_per_quintal": round(price, 1)})

    recent = [p["price_per_quintal"] for p in series[-7:]]
    trend = "rising" if recent[-1] > recent[0] * 1.02 else "falling" if recent[-1] < recent[0] * 0.98 else "stable"

    mandis = []
    for i, name in enumerate(["District Mandi A", "Regional Hub B", "Wholesale Market C"]):
        offset = _seeded_random(crop + name, 1) * base * 0.05
        mandis.append({"mandi": name, "price_per_quintal": round(price + offset, 1),
                        "distance_km": round(8 + i * 14 + abs(offset) / 50, 1)})
    mandis.sort(key=lambda m: -m["price_per_quintal"])

    if trend == "rising":
        advisory = "Prices are trending up — holding for a few more days may improve returns if storage allows."
    elif trend == "falling":
        advisory = "Prices are trending down — selling soon may lock in better returns than waiting."
    else:
        advisory = "Prices are stable — sell based on storage cost and cash-flow needs."

    return {
        "crop": crop,
        "current_price_per_quintal": round(price, 1),
        "trend_7d": trend,
        "history_30d": series,
        "nearby_mandis": mandis,
        "best_mandi": mandis[0],
        "advisory": advisory,
    }
