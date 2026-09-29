"""
AgriVision API
==============
A Flask backend for an AgriTech hackathon project covering:

  - AI-assisted crop disease detection (image upload)
  - Crop yield prediction
  - Soil health scoring
  - Irrigation advisory (water budgeting)
  - Market / supply-chain price insights
  - A dashboard summary + analysis history (SQLite)

Run:
    pip install -r requirements.txt
    python app.py
Then open frontend/index.html in a browser (see project README).
"""

from flask import Flask, request, jsonify
import traceback

import ml_engine
import db

app = Flask(__name__)
db.init_db()

SUPPORTED_CROPS = list(ml_engine.CROP_BASELINE.keys())


# --------------------------------------------------------------------------
# CORS (no flask-cors dependency needed — handled manually so the static
# frontend can call this API from a file:// or different-port origin)
# --------------------------------------------------------------------------
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def cors_preflight(_any):
    return "", 204


def error_response(message, status=400):
    return jsonify({"error": message}), status


# --------------------------------------------------------------------------
# Meta
# --------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "AgriVision API",
        "supported_crops": SUPPORTED_CROPS,
    })


@app.route("/api/dashboard/summary", methods=["GET"])
def dashboard_summary():
    return jsonify(db.get_dashboard_stats())


@app.route("/api/history", methods=["GET"])
def history():
    module = request.args.get("module")
    limit = int(request.args.get("limit", 50))
    return jsonify(db.get_history(limit=limit, module=module))


# --------------------------------------------------------------------------
# 1. Disease detection
# --------------------------------------------------------------------------
@app.route("/api/disease/detect", methods=["POST"])
def disease_detect():
    if "image" not in request.files:
        return error_response("Upload an image file under form field 'image'.")
    file = request.files["image"]
    image_bytes = file.read()
    if not image_bytes:
        return error_response("Empty image upload.")
    try:
        result = ml_engine.predict_disease(image_bytes)
    except Exception as exc:
        return error_response(f"Could not process image: {exc}", 422)

    db.log_analysis(
        "disease",
        result["prediction"],
        {"filename": file.filename, **result},
    )
    return jsonify(result)


# --------------------------------------------------------------------------
# 2. Yield prediction
# --------------------------------------------------------------------------
@app.route("/api/yield/predict", methods=["POST"])
def yield_predict():
    data = request.get_json(silent=True) or {}
    try:
        crop = str(data["crop"]).lower()
        area_ha = float(data.get("area_ha", 1))
        temp = float(data["avg_temp_c"])
        rain = float(data["seasonal_rainfall_mm"])
        ph = float(data["soil_ph"])
        fert = float(data.get("fertilizer_kg_per_ha", 60))
        irrigation = str(data.get("irrigation_type", "canal")).lower()
    except (KeyError, ValueError, TypeError) as exc:
        return error_response(f"Missing/invalid field: {exc}")

    if crop not in SUPPORTED_CROPS:
        return error_response(f"Unsupported crop '{crop}'. Supported: {SUPPORTED_CROPS}")

    try:
        result = ml_engine.predict_yield(crop, area_ha, temp, rain, ph, fert, irrigation)
    except Exception as exc:
        traceback.print_exc()
        return error_response(str(exc), 422)

    db.log_analysis(
        "yield",
        f"{crop}: {result['predicted_yield_t_per_ha']} t/ha",
        {"inputs": data, **result},
    )
    return jsonify(result)


# --------------------------------------------------------------------------
# 3. Soil health
# --------------------------------------------------------------------------
@app.route("/api/soil/health", methods=["POST"])
def soil_health_route():
    data = request.get_json(silent=True) or {}
    try:
        n = float(data["nitrogen_mg_kg"])
        p = float(data["phosphorus_mg_kg"])
        k = float(data["potassium_mg_kg"])
        ph = float(data["ph"])
        oc = float(data.get("organic_carbon_pct", 0.5))
        crop = str(data.get("crop", "wheat")).lower()
    except (KeyError, ValueError, TypeError) as exc:
        return error_response(f"Missing/invalid field: {exc}")

    result = ml_engine.soil_health(n, p, k, ph, oc, crop)
    db.log_analysis(
        "soil",
        f"{result['category']} ({result['overall_score']})",
        {"inputs": data, **result},
    )
    return jsonify(result)


# --------------------------------------------------------------------------
# 4. Irrigation advisory
# --------------------------------------------------------------------------
@app.route("/api/irrigation/advisory", methods=["POST"])
def irrigation_route():
    data = request.get_json(silent=True) or {}
    try:
        moisture = float(data["soil_moisture_pct"])
        temp = float(data["temperature_c"])
        humidity = float(data["humidity_pct"])
        crop = str(data.get("crop", "wheat")).lower()
        stage = str(data.get("growth_stage", "vegetative")).lower()
    except (KeyError, ValueError, TypeError) as exc:
        return error_response(f"Missing/invalid field: {exc}")

    result = ml_engine.irrigation_advisory(moisture, temp, humidity, crop, stage)
    db.log_analysis(
        "irrigation",
        f"{crop}: {result['urgency']} urgency",
        {"inputs": data, **result},
    )
    return jsonify(result)


# --------------------------------------------------------------------------
# 5. Market insights
# --------------------------------------------------------------------------
@app.route("/api/market/insights", methods=["GET"])
def market_route():
    crop = request.args.get("crop", "wheat").lower()
    result = ml_engine.market_insights(crop)
    db.log_analysis("market", f"{crop}: {result['trend_7d']}", result)
    return jsonify(result)


if __name__ == "__main__":
    print("AgriVision API starting on http://localhost:5000")
    print(f"Supported crops: {SUPPORTED_CROPS}")
    app.run(host="0.0.0.0", port=5000, debug=True)
