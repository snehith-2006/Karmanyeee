from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Create Flask app (explicit template/static folders)
app = Flask(__name__, template_folder='Templates', static_folder='static')

# Load trained ML model (fall back to mock if missing)
try:
    model = joblib.load("budget_model.pkl")
    MODEL_AVAILABLE = True
except Exception as e:
    print("Warning: could not load model; running in mock/dev mode:", e)
    model = None
    MODEL_AVAILABLE = False


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# PREDICTION ROUTE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    prior_visits = float(request.form.get("prior_visits", 0))
    prior_spend = float(request.form.get("prior_spend", 0))
    treatment = float(request.form.get("treatment", 0))
    impressions = float(request.form.get("impressions", 0))
    clicks = float(request.form.get("clicks", 0))
    conversion = float(request.form.get("conversion", 0))
    revenue = float(request.form.get("revenue", 0))

    # Feature engineering (same as training)
    CTR = clicks / (impressions + 1)
    CPC = 1
    CONV_RATE = conversion / (clicks + 1)

    features = np.array([[ 
        prior_visits,
        prior_spend,
        treatment,
        impressions,
        clicks,
        conversion,
        revenue,
        CTR,
        CPC,
        CONV_RATE
    ]])

    use_mock = request.form.get('use_mock') == '1' or (not MODEL_AVAILABLE)

    if use_mock:
        # Simple deterministic mock formula for development / demos
        prediction = (revenue * 0.05) + (prior_spend * 0.02) + (conversion * 12) + (clicks * 0.01) + (impressions * 0.0001)
        confidence_val = 0.60
    else:
        prediction = model.predict(features)[0] * 1000
        confidence_val = 0.85

    low = prediction * 0.9
    high = prediction * 1.1

    # If the client expects JSON (AJAX), return JSON; otherwise render HTML
    accept = request.headers.get('Accept', '')
    if 'application/json' in accept or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            'prediction': round(prediction, 2),
            'confidence': 0.85,
            'message': f'Suggested range: ${round(low,2)}–${round(high,2)}'
        })

    return render_template(
        "result.html",
        budget=round(prediction, 2),
        low=round(low, 2),
        high=round(high, 2)
    )


# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
