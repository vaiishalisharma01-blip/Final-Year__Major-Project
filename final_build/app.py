"""
CropSense AI — Intelligent Agriculture Assistant
Main Flask Application
Features:
  - Crop Recommendation  (Random Forest ML model)
  - Fertilizer Recommendation (N/P/K soil analysis)
  - Plant Disease Detection  (ResNet9 deep learning model)
  - Agriculture Chatbot
"""

import os
import io
import pickle
import warnings

import numpy as np
import pandas as pd
import requests
import torch
from flask import Flask, jsonify, render_template, request
from markupsafe import Markup
from PIL import Image
from torchvision import transforms

from utils.fertilizer_disc import fertilizer_disc
from utils.model import ResNet9

warnings.filterwarnings('ignore')

# ── Disease classes (38 categories from PlantVillage dataset) ────────────────
DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

# ── Model paths (use os.path.join for cross-platform compatibility) ───────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'plant_disease_model.pth')
FERTILIZER_CSV     = os.path.join(BASE_DIR, 'data', 'fertilizer.csv')

# ── Load models at startup ───────────────────────────────────────────────────
disease_model = ResNet9(3, len(DISEASE_CLASSES))
disease_model.load_state_dict(
    torch.load(DISEASE_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
)
disease_model.eval()

# Load crop model once at startup (avoids per-request overhead and version issues)
RF_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'RandomForest.pkl')
with open(RF_MODEL_PATH, 'rb') as f:
    rf_model = pickle.load(f)


def predict_image(img_bytes, model=disease_model):
    """
    Transforms image bytes to tensor and predicts plant disease label.

    Args:
        img_bytes: Raw image bytes from uploaded file.
        model: Loaded PyTorch ResNet9 model.

    Returns:
        prediction (str): Predicted disease class name.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_tensor = transform(image)
    img_batch  = torch.unsqueeze(img_tensor, 0)

    with torch.no_grad():
        outputs = model(img_batch)
        _, preds = torch.max(outputs, dim=1)

    return DISEASE_CLASSES[preds[0].item()]


def weather_fetch(city_name):
    """
    Fetches current temperature (°C) and humidity (%) for a city
    using the OpenWeatherMap API.

    Args:
        city_name (str): Name of the city.

    Returns:
        (temperature, humidity) tuple, or None if city not found.

    NOTE: Replace the API key below with your own from openweathermap.org
          Store it in an environment variable in production — never hardcode.
    """
    api_key  = os.environ.get('OPENWEATHER_API_KEY', 'YOUR_API_KEY_HERE')
    base_url = 'http://api.openweathermap.org/data/2.5/weather'

    params = {
        'q':     city_name,
        'appid': api_key,
        'units': 'metric',   # Returns temperature in Celsius directly
    }

    try:
        response = requests.get(base_url, params=params, timeout=5)
        data = response.json()
        # API returns cod=200 (int) on success, cod='404' (string) on city not found
        if str(data.get('cod')) != '404' and 'main' in data:
            main = data['main']
            temperature = round(main['temp'], 2)
            humidity    = main['humidity']
            return temperature, humidity
    except requests.RequestException:
        pass

    return None


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='template')


@app.route('/')
def index():
    """Render the main landing page."""
    return render_template('index.html')


@app.route('/crop_recom.html')
def crop_recom():
    """Render the crop recommendation page."""
    return render_template('crop_recom.html')


@app.route('/fertilizer.html')
def fert_recom():
    """Render the fertilizer recommendation page."""
    return render_template('fertilizer.html')


@app.route('/disease.html')
def dis_detect():
    """Render the plant disease detection page."""
    return render_template('disease.html')


@app.route('/chatbot2.html')
def chatbot2():
    """Render the AI chatbot page."""
    return render_template('chatbot2.html')


@app.route('/predict', methods=['POST'])
def predict_crop():
    """
    Predict the best crop to grow based on soil nutrients,
    weather conditions, pH, and rainfall.

    Form inputs: nitrogen, phosphorus, potassium, ph-level, rainfall, city
    Returns: JSON with { prediction: <crop_name> }
    """
    try:
        N        = int(request.form['nitrogen'])
        P        = int(request.form['phosphorus'])
        K        = int(request.form['potassium'])
        ph       = float(request.form['ph-level'])
        rainfall = float(request.form['rainfall'])
        city     = request.form.get('city', '').strip()
    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Invalid input: {str(e)}. Please fill all fields correctly.'}), 400

    weather = weather_fetch(city)
    if weather is None:
        # Fallback: use city-based seasonal averages so the app works
        # even without an API key set. Values are reasonable Indian averages.
        city_defaults = {
            'Mumbai': (30, 75), 'Delhi': (28, 55), 'Bengaluru': (24, 65),
            'Hyderabad': (28, 60), 'Ahmedabad': (32, 50), 'Chennai': (31, 72),
            'Kolkata': (29, 78), 'Pune': (26, 62), 'Jaipur': (30, 48),
            'Lucknow': (27, 60), 'Kanpur': (28, 58), 'Nagpur': (30, 55),
            'Indore': (27, 58), 'Bhopal': (27, 60), 'Patna': (28, 68),
            'Ludhiana': (25, 55), 'Agra': (29, 52), 'Varanasi': (28, 62),
            'Nashik': (26, 60), 'Coimbatore': (27, 68),
        }
        temperature, humidity = city_defaults.get(city, (27, 60))
    else:
        temperature, humidity = weather
    # Use DataFrame to preserve feature names matching the training data
    features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                             columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    prediction = rf_model.predict(features)[0]

    return jsonify({'prediction': prediction})


@app.route('/recommend_fertilizer', methods=['POST'])
def recommend_fert():
    """
    Recommend fertilizer adjustments based on the difference between
    current soil N/P/K levels and the ideal values for the chosen crop.

    Form inputs: nitrogen, phosphorus, potassium, crop
    Returns: JSON with { prediction: <html_advice> }
    """
    N    = float(request.form['nitrogen'])
    P    = float(request.form['phosphorus'])
    K    = float(request.form['potassium'])
    crop = request.form['crop']

    df = pd.read_csv(FERTILIZER_CSV)

    # Get ideal N/P/K values for the selected crop
    crop_row = df[df['Crop'] == crop]
    if crop_row.empty:
        return jsonify({'error': f'Crop "{crop}" not found in database.'}), 400

    ideal_n = crop_row['N'].iloc[0]
    ideal_p = crop_row['P'].iloc[0]
    ideal_k = crop_row['K'].iloc[0]

    diff_n = ideal_n - N
    diff_p = ideal_p - P
    diff_k = ideal_k - K

    # Find the nutrient with the largest deviation (list avoids dict key collision)
    deviations = [
        (abs(diff_n), 'N', diff_n),
        (abs(diff_p), 'P', diff_p),
        (abs(diff_k), 'K', diff_k),
    ]
    _, nutrient, diff = max(deviations, key=lambda x: x[0])

    key_map = {
        'N': 'NHigh' if diff < 0 else 'Nlow',
        'P': 'PHigh' if diff < 0 else 'Plow',
        'K': 'KHigh' if diff < 0 else 'Klow',
    }
    key = key_map[nutrient]

    return jsonify({'prediction': Markup(fertilizer_disc[key])})


@app.route('/detect_disease', methods=['POST'])
def detect_disease():
    """
    Detect plant disease from an uploaded leaf image using ResNet9.

    Form data: image (file upload)
    Returns: JSON with { result: <disease_class> } or { error: <message> }
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    try:
        img_bytes = file.read()
        prediction = predict_image(img_bytes)
        return jsonify({'result': prediction})
    except Exception as e:
        return jsonify({'error': f'Image processing failed: {str(e)}'}), 500


if __name__ == '__main__':
    # Set OPENWEATHER_API_KEY as an environment variable before running:
    #   Windows:  set OPENWEATHER_API_KEY=your_key_here
    #   Mac/Linux: export OPENWEATHER_API_KEY=your_key_here
    app.run(debug=True)

# For Vercel deployment
try:
    from vercel_wsgi import make_handler
    handler = make_handler(app)
except ImportError:
    pass
