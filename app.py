from flask import Flask, request, render_template, url_for
import numpy as np
import pickle
import sklearn

# Initialize Flask app
app = Flask(__name__)

# --- Load Models for Crop Recommendation ---
try:
    crop_model = pickle.load(open('model.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
except Exception as e:
    print(f"Error loading Crop Recommendation models: {e}")

# --- Load Models for Fertilizer Recommendation ---
try:
    fertilizer_model = pickle.load(open('classifier.pkl', 'rb'))
    fertilizer_encoder = pickle.load(open('fertilizer.pkl', 'rb'))
except Exception as e:
    print(f"Error loading Fertilizer Recommendation models: {e}")

# ---------------- ROUTES ----------------

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/crop')
def crop_recommendation():
    return render_template("crop.html")

@app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template("fertilizer.html")

# =================================================
# ✅ ✅ ✅ NEW ROUTE ADDED (THIS IS THE FIX)
# =================================================
@app.route('/detail')                      # ← NEW
def detail():                             # ← NEW
    return render_template("Detail.html") # ← NEW
# =================================================

# =================================================
# Existing crop info route
# =================================================
@app.route('/crop-info')
def crop_info():
    return render_template("information/crops.html")

@app.route("/predict_crop", methods=['POST'])
def predict_crop():
    try:
        N = request.form['Nitrogen']
        P = request.form['Phosporus']
        K = request.form['Potassium']
        temp = request.form['Temperature']
        humidity = request.form['Humidity']
        ph = request.form['Ph']
        rainfall = request.form['Rainfall']

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = crop_model.predict(final_features)

        if len(prediction) > 0:
            crop = prediction[0]
            result = "{} is the best crop to be cultivated.".format(crop.capitalize())
        else:
            result = "Sorry, we could not determine the best crop to be cultivated."

    except Exception as e:
        result = f"An error occurred: {str(e)}"

    return render_template('crop.html', result=result)

@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        temp = request.form.get('temp')
        humi = request.form.get('humid')
        mois = request.form.get('mois')
        soil = request.form.get('soil')
        crop = request.form.get('crop')
        nitro = request.form.get('nitro')
        pota = request.form.get('pota')
        phosp = request.form.get('phos')

        if None in (temp, humi, mois, soil, crop, nitro, pota, phosp):
            return render_template('fertilizer.html', result='Invalid input. Please provide all fields.')

        if fertilizer_model is None or fertilizer_encoder is None:
            return render_template('fertilizer.html', result='Error: Fertilizer model not loaded properly.')

        input_data = [
            int(temp), int(humi), int(mois),
            int(soil), int(crop), int(nitro),
            int(pota), int(phosp)
        ]

        prediction_index = fertilizer_model.predict([input_data])
        res = fertilizer_encoder.classes_[prediction_index][0]

        result = f"The recommended fertilizer is: {res}"

    except Exception as e:
        result = f"An error occurred: {str(e)}"

    return render_template('fertilizer.html', result=result)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
